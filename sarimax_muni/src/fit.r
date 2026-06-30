source("sarimax/src/utils.r")

#' Convert a YYYYWW epiweek integer to the Date of its opening Sunday (MMWR)
#'
#' Anchors on Jan 4, which is always in MMWR epidemiological week 1.
#'
#' @param yw  Integer (or vector) epiweek in YYYYWW format, e.g. 202541.
#'
#' @return Date (or vector of Dates): the Sunday opening that epiweek.
# ── Helper: get the Sunday date opening a given YYYYWW epiweek (MMWR) ──────
# Anchors on Jan 4, which is always in MMWR week 1.
epiweek_to_date <- function(yw) {
  yr   <- yw %/% 100L
  wk   <- yw  %% 100L
  jan4 <- as.Date(paste0(yr, "-01-04"))
  dow_jan4  <- as.integer(format(jan4, "%w"))  # %w: 0 = Sunday
  sunday_w1 <- jan4 - dow_jan4                 # Sunday on or before Jan 4
  sunday_w1 + (wk - 1L) * 7L
}

#' Check whether a given year has 53 epiweeks (MMWR/Brazilian calendar)
#'
#' Derived from `epiweek_to_date()` for consistency: a year has 53 weeks iff
#' the Sunday that would open week 53 falls before week 1 of the next year.
#'
#' @param year  Integer year (e.g. 2025).
#'
#' @return Logical: TRUE if `year` has 53 epiweeks, FALSE if it has 52.
# ── Helper: does a given year have 53 epiweeks? (MMWR/Brazilian calendar) ──
# Derived from epiweek_to_date for consistency: year has 53 weeks iff the
# Sunday that would open week 53 falls before week 1 of the next year.
has_53_weeks <- function(year) {
  epiweek_to_date(year * 100L + 53L) < epiweek_to_date((year + 1L) * 100L + 1L)
}

#' Enumerate all epiweeks (inclusive) between two YYYYWW integers
#'
#' Correctly rolls over year boundaries and accounts for 52- vs 53-week years
#' via `has_53_weeks()`.
#'
#' @param start  Integer epiweek in YYYYWW format marking the start (inclusive).
#' @param end    Integer epiweek in YYYYWW format marking the end (inclusive).
#'
#' @return Integer vector of YYYYWW epiweeks from `start` to `end`.
# ── Helper: enumerate all epiweeks between two YYYYWW integers ─────────────
enumerate_epiweeks <- function(start, end) {
  start_year <- start %/% 100L
  start_week <- start  %% 100L
  end_year   <- end   %/% 100L
  end_week   <- end    %% 100L

  epiweeks <- integer(0)
  yr <- start_year
  wk <- start_week

  repeat {
    epiweeks <- c(epiweeks, yr * 100L + wk)
    if (yr == end_year && wk == end_week) break
    n_weeks <- if (has_53_weeks(yr)) 53L else 52L
    if (wk < n_weeks) {
      wk <- wk + 1L
    } else {
      yr <- yr + 1L
      wk <- 1L
    }
  }
  epiweeks
}

#' Map an epiweek to the same calendar week one year earlier
#'
#' Used to source a seasonal-naive covariate substitute for forecast horizons
#' where future covariates are not yet observed. If the previous year does
#' not have a week 53 (i.e. it is a 52-week year), falls back to week 52 of
#' the previous year — the closest available epiweek.
#'
#' @param yw  Integer epiweek in YYYYWW format.
#'
#' @return A list with `epiweek` (the previous year\'s YYYYWW epiweek) and
#'         `adjusted` (logical, TRUE if a week-53 fallback was applied).
# ── Helper: given YYYYWW, return the previous year's equivalent epiweek ────
# If the previous year does not have week 53 (52-week year), fall back to
# week 52 of the previous year — the closest available epiweek.
prev_year_epiweek <- function(yw) {
  yr <- yw %/% 100L
  wk <- yw  %% 100L
  prev_yr      <- yr - 1L
  prev_n_weeks <- if (has_53_weeks(prev_yr)) 53L else 52L

  if (wk <= prev_n_weeks) {
    list(epiweek = prev_yr * 100L + wk, adjusted = FALSE)
  } else {
    # week 53 doesn't exist in prev year → use last week of prev year
    list(epiweek = prev_yr * 100L + prev_n_weeks, adjusted = TRUE)
  }
}


#' Fit a SARIMAX model over an explicit epiweek range and forecast a future epiweek range
#'
#' Like `fit_sarimax()`, but splits train/forecast rows by epiweek range
#' rather than by train_*/target_* indicator columns, and substitutes
#' unobserved future covariates with the previous year\'s values at the same
#' epiweek (via `prev_year_epiweek()`) — a seasonal-naive covariate forecast
#' used because the real future-covariate values are not yet available at
#' submission time.
#'
#' @param data            Data frame with `epiweek`, `cases`, and all
#'                        covariates referenced in `formula`.
#' @param formula         Formula or character vector of covariate names
#'                        (RHS only; `cases` is the implicit response).
#' @param train_start     Integer YYYYWW: first training epiweek (inclusive).
#' @param train_end       Integer YYYYWW: last training epiweek (inclusive).
#' @param forecast_start  Integer YYYYWW: first forecast epiweek (inclusive).
#' @param forecast_end    Integer YYYYWW: last forecast epiweek (inclusive).
#' @param method          Estimation method passed to `Arima()` (default
#'                        "CSS-ML").
#' @param lambda          Box-Cox lambda; NULL (default) uses log1p/expm1
#'                        transform instead.
#' @param optim.control   List passed to `Arima()`\'s optimizer (default
#'                        list(maxit = 500)).
#' @param optim.method    Optimization method passed to `Arima()` (default
#'                        "BFGS").
#' @param levels          Numeric vector of prediction-interval coverage
#'                        levels (default c(50, 80, 90, 95)).
#' @param order           ARIMA (p,d,q) order (default c(1,1,1)).
#' @param seasonal        Seasonal ARIMA list, e.g.
#'                        list(order = c(1,0,1), period = 52).
#' @param bootstrap       Logical: use bootstrapped simulation paths for
#'                        prediction intervals instead of Gaussian
#'                        normal-theory intervals (default TRUE).
#' @param npaths          Number of bootstrap simulation paths (default 1000).
#'
#' @return A tibble with columns `date`, `pred`, and `lower_*`/`upper_*` for
#'         each level in `levels`, with `warnings` and `fit` attributes
#'         attached (as in `fit_sarimax()`).
fit_sarimax_epiweek <- function(data,
                        formula,
                        train_start,
                        train_end,
                        forecast_start,
                        forecast_end,
                        method        = "CSS-ML",
                        lambda        = NULL,
                        optim.control = list(maxit = 500),
                        optim.method  = "BFGS",
                        levels        = c(50, 80, 90, 95),
                        order         = c(1, 1, 1),
                        seasonal      = list(order = c(1, 0, 1), period = 52),
                        bootstrap     = TRUE,
                        npaths        = 1000) {

  stopifnot(is.data.frame(data))
  stopifnot("epiweek" %in% names(data))
  stopifnot("cases"   %in% names(data))

  # ── 1. Training rows ────────────────────────────────────────────────────────
  train_rows <- data[data$epiweek >= train_start & data$epiweek <= train_end, ]
  if (nrow(train_rows) == 0) {
    stop("No training rows found for epiweeks ", train_start, "–", train_end)
  }

  # ── 2. Enumerate forecast epiweeks & build forecast dates/prev-year info ───
  fc_epiweeks <- enumerate_epiweeks(forecast_start, forecast_end)
  n_fc        <- length(fc_epiweeks)

  prev_epiweeks  <- integer(n_fc)
  forecast_dates <- as.Date(rep(NA, n_fc))
  adjusted_idx   <- logical(n_fc)

  for (i in seq_len(n_fc)) {
    res               <- prev_year_epiweek(fc_epiweeks[i])
    prev_epiweeks[i]  <- res$epiweek
    adjusted_idx[i]   <- res$adjusted
    forecast_dates[i] <- epiweek_to_date(fc_epiweeks[i])  # already Sunday
  }

  if (any(adjusted_idx)) {
    adj_info <- paste0(
      fc_epiweeks[adjusted_idx], " (prev: ", prev_epiweeks[adjusted_idx], ")",
      collapse = "; "
    )
    warning(
      "53-week year mismatch: the following forecast epiweeks have no direct ",
      "previous-year equivalent and were shifted forward by one week:\n  ",
      adj_info,
      call. = FALSE
    )
  }

  # ── 3. Log-transform response ───────────────────────────────────────────────
  y <- if (is.null(lambda)) log1p(train_rows$cases) else train_rows$cases

  # ── 4. Build & standardize regressor matrices ───────────────────────────────
  rhs_terms <- {
    if (is.null(formula)) {
      character(0)
    } else if (is.character(formula)) {
      if (length(formula) == 0) character(0)
      else {
        fo <- stats::as.formula(paste("~", paste(formula, collapse = "+")))
        attr(stats::terms(fo), "term.labels")
      }
    } else {
      attr(stats::terms(formula), "term.labels")
    }
  }

  if (length(rhs_terms) == 0) {
    xreg_train  <- NULL
    xreg_future <- NULL
  } else {
    raw_train <- as.matrix(train_rows[, rhs_terms, drop = FALSE])

    col_means <- colMeans(raw_train, na.rm = TRUE)
    col_sds   <- apply(raw_train, 2, sd, na.rm = TRUE)
    col_sds[col_sds == 0] <- 1

    xreg_train <- scale(raw_train, center = col_means, scale = col_sds)

    # Look up previous-year rows by epiweek (vectorised, preserving order)
    prev_rows <- data[match(prev_epiweeks, data$epiweek), rhs_terms, drop = FALSE]

    if (any(is.na(prev_rows))) {
      missing_ew <- prev_epiweeks[apply(is.na(prev_rows), 1, any)]
      stop(
        "Could not find previous-year covariate data for epiweeks: ",
        paste(missing_ew, collapse = ", ")
      )
    }

    raw_future  <- as.matrix(prev_rows)
    xreg_future <- scale(raw_future, center = col_means, scale = col_sds)
  }

  # ── 5. Fit SARIMAX ──────────────────────────────────────────────────────────
  y_ts         <- ts(y, frequency = seasonal$period)
  fit_warnings <- character(0)

  fit <- withCallingHandlers(
    Arima(y_ts,
          order    = order,
          seasonal = seasonal,
          xreg     = xreg_train,
          method   = method,
          lambda   = lambda,
          optim.control = optim.control,
          optim.method  = optim.method
        ),
    error = function(e) stop("Arima() failed: ", conditionMessage(e)),
    warning = function(w) {
      fit_warnings <<- c(fit_warnings, conditionMessage(w))
      invokeRestart("muffleWarning")
    }
  )

  # ── 6. Detect NaN standard errors ──────────────────────────────────────────
  se_warnings <- character(0)
  ses <- withCallingHandlers(
    sqrt(diag(fit$var.coef)),
    warning = function(w) {
      se_warnings <<- c(se_warnings, conditionMessage(w))
      invokeRestart("muffleWarning")
    }
  )
  nan_se_params <- names(which(is.nan(ses)))
  if (length(nan_se_params) > 0) {
    fit_warnings <- c(
      fit_warnings,
      se_warnings,
      paste0(
        "NaN standard errors for: ",
        paste(nan_se_params, collapse = ", "),
        ". Prediction intervals may be unreliable. ",
        "Consider reducing model order or using auto.arima()."
      )
    )
  }

  # ── 7. Forecast & back-transform ────────────────────────────────────────────
  levels      <- sort(unique(levels))
  fc_warnings <- character(0)
  bt          <- if (is.null(lambda)) expm1 else identity

  if (bootstrap) {
    # ── Bootstrap-simulated point forecast + intervals ──────────────────────
    # Point forecast and interval bounds come from the same simulated future
    # paths (median and quantiles respectively) so they stay mutually
    # consistent -- avoids the degenerate-forecast failure mode where the
    # deterministic log-scale mean back-transforms to 0 (clipped by pmax)
    # while the upper bound, derived separately via bootstrap, stays
    # positive. See is_degenerate_forecast() below and simulate_forecast()
    # in sarimax/src/utils.r.
    sim <- withCallingHandlers(
      simulate_forecast(fit, h = n_fc, xreg = xreg_future, levels = levels,
                        npaths = npaths, bt = bt),
      warning = function(w) {
        fc_warnings <<- c(fc_warnings, conditionMessage(w))
        invokeRestart("muffleWarning")
      }
    )
    fc_warnings <- c(fc_warnings, sim$warnings)

    out <- tibble::tibble(date = forecast_dates, pred = sim$pred)
    for (lv in levels) {
      out[[paste0("lower_", lv)]] <- sim[[paste0("lower_", lv)]]
      out[[paste0("upper_", lv)]] <- sim[[paste0("upper_", lv)]]
    }
  } else {
    # ── Gaussian normal-theory point forecast + intervals (legacy path) ────
    fc <- withCallingHandlers(
      forecast::forecast(
        fit,
        h         = n_fc,
        xreg      = xreg_future,
        level     = levels,
        bootstrap = bootstrap,
        npaths    = npaths
      ),
      warning = function(w) {
        fc_warnings <<- c(fc_warnings, conditionMessage(w))
        invokeRestart("muffleWarning")
      }
    )

    out <- tibble::tibble(date = forecast_dates, pred = bt(as.numeric(fc$mean)))
    for (lv in levels) {
      lv_char <- paste0(lv, "%")
      out[[paste0("lower_", lv)]] <- bt(as.numeric(fc$lower[, lv_char]))
      out[[paste0("upper_", lv)]] <- bt(as.numeric(fc$upper[, lv_char]))
    }
  }

  out <- out |>
    dplyr::mutate(dplyr::across(
      c(pred, dplyr::starts_with("lower_"), dplyr::starts_with("upper_")),
      \(x) pmax(x, 0)
    ))

  # ── 9. Attach metadata ──────────────────────────────────────────────────────
  all_warnings <- c(fit_warnings, fc_warnings)
  attr(out, "warnings") <- if (length(all_warnings) > 0) all_warnings else NULL
  attr(out, "fit")      <- fit

  out
}

# ── Final forecasts for the sel_cities municipalities (same city set used in
#    sarimax/src/model_sel.r's cities_dengue / cities_chikungunya) ──────────

#' Flatten a per-city named list of warning/error messages into a tibble
#'
#' `warn_list` is a list keyed by city code, where each element is a named
#' character vector (names = target_id, values = warning/error message), as
#' accumulated in `city_warnings_*`/`attempt_warnings` below. Used to persist
#' both real Arima() warnings and caught Arima() errors (e.g. "non-finite
#' finite-difference value") to a single CSV log per disease.
to_warnings_df <- function(warn_list, phase) {
  if (length(warn_list) == 0) {
    return(tibble(phase = character(), city = character(), target_id = character(), message = character()))
  }
  bind_rows(lapply(names(warn_list), function(ct) {
    w <- warn_list[[ct]]
    if (length(w) == 0) return(NULL)
    tibble(phase = phase, city = ct, target_id = names(w), message = unname(w))
  }))
}

#' Detect a degenerate (all-zero) point forecast
#'
#' A SARIMAX fit on a sparse, intermittent case series (common for
#' lower-incidence cities/diseases) can converge cleanly — no Arima()
#' warnings, no NaN standard errors — and still produce a central point
#' forecast (`pred`) that is exactly zero for the entire horizon, because the
#' log1p(cases)-scale forecast drifts to <= 0 and gets clipped by
#' `pmax(x, 0)` inside `fit_sarimax()`/`fit_sarimax_epiweek()`. That failure
#' mode is silent under the warning/error-based retry logic above, so it
#' gets its own explicit check, treated the same way as an actionable
#' warning: it triggers a retry with the next-best formula/order.
#'
#' @param fit  The forecast tibble returned by `fit_sarimax()`/
#'             `fit_sarimax_epiweek()` (or NULL, if the fit failed outright).
#'
#' @return Logical: TRUE if `fit` is non-NULL and every value in its `pred`
#'         column is exactly 0.
is_degenerate_forecast <- function(fit) {
  !is.null(fit) && all(fit$pred == 0)
}

best_wis_df_dengue_cities <- read_csv("sarimax/results/metrics/best_wis_dengue_all_cities.csv", show_col_types = FALSE)
city_warnings_dengue <- list()

preds_dengue_cities <- lapply(best_wis_df_dengue_cities$city, function(ct) {
  file_name <- paste0("processed_data/dengue/sel_cities/dengue_", ct, "_agg.csv.gz")
  d <- read_csv(file_name, show_col_types = FALSE)
  train_rows <- d$train_1 == 1

  candidates <- get_candidates(d)
  candidates <- filter_low_variance(d, candidates, threshold = 0.01)
  candidates <- filter_by_correlation(
    d[train_rows, ],
    candidates,
    min_cor = 0.1
  )

  if (length(candidates) == 0) {
    candidates <- get_candidates(dengue_state)
    candidates <- filter_low_variance(
      d[
        d[[train_ids[1]]] == 1, 
      ],
      candidates, 
      threshold = threshold_low_variance
    )
  }

  pca_result <- pca_all(
    data = d,
    candidates = candidates[!grepl("enso|iod|pdo", candidates)],
    var_threshold = 0.9
  )
  d <- pca_result$data

  best_order <- best_wis_df_dengue_cities |> filter(city == ct) |> pull(order)
  ord <- parse_order(best_order)
  best_formula <- best_wis_df_dengue_cities |> filter(city == ct) |> pull(formula_id)
  train_ids  <- paste0("train_", 1:4)
  target_ids <- paste0("target_", 1:4)
  pred_target <- lapply(seq_along(train_ids), function(i) {
    train_id <- train_ids[i]
    target_id <- target_ids[i]
    # ── Catch hard Arima() errors (e.g. "non-finite finite-difference value")
    #    so a single bad fit doesn't crash the whole run; treat it like a
    #    warning that needs a retry with a different formula/order.
    fit <- tryCatch({
      if (train_id == "train_4") {
        fit_sarimax_epiweek(
          data = d,
          formula = best_formula,
          train_start = 201001,
          train_end = 202525,
          forecast_start = 202541,
          forecast_end = 202640,
          order = c(ord$order[1], ord$order[2], ord$order[3]),
          seasonal = list(order = c(ord$seasonal_order[1], ord$seasonal_order[2], ord$seasonal_order[3]), period = 52)
        )
      } else {
        fit_sarimax(
          data = d,
          formula = best_formula,
          order = c(ord$order[1], ord$order[2], ord$order[3]),
          seasonal = list(order = c(ord$seasonal_order[1], ord$seasonal_order[2], ord$seasonal_order[3]), period = 52),
          train_id = train_id
        )
      }
    }, error = function(e) {
      err_msg <- paste0("Arima() failed: ", conditionMessage(e))
      city_warnings_dengue[[as.character(ct)]] <<- c(city_warnings_dengue[[as.character(ct)]],
        setNames(err_msg, target_id))
      message(sprintf("[%s | %s] ERROR: %s", ct, target_id, err_msg))
      NULL
    })
    if (is.null(fit)) return(NULL)

    # ── Report warnings with city/target context ───────────────────────────
    w <- attr(fit, "warnings")
    if (!is.null(w)) {
      city_warnings_dengue[[as.character(ct)]] <<- c(city_warnings_dengue[[as.character(ct)]],
        setNames(w, rep(target_id, length(w))))
      message(sprintf("[%s | %s] %d warning(s):\n%s",
                      ct, target_id, length(w),
                      paste0("  - ", w, collapse = "\n")))
    }

    # ── Flag degenerate (all-zero) point forecasts so this city is retried
    #    with the next-best formula/order, same as a real Arima() warning ──
    if (is_degenerate_forecast(fit)) {
      city_warnings_dengue[[as.character(ct)]] <<- c(city_warnings_dengue[[as.character(ct)]],
        setNames("Degenerate forecast: all point predictions are zero", target_id))
      message(sprintf("[%s | %s] Degenerate forecast: all point predictions are zero", ct, target_id))
    }
    write_csv(fit, file.path("sarimax/results/preds/", paste0("pred_dengue_", ct, "_", target_id, ".csv")))
    fit
  })
  names(pred_target) <- target_ids
  bind_rows(pred_target, .id = "target_id") |>
    mutate(city = ct)
})

cities_to_retry_dengue <- names(city_warnings_dengue)
# cities_to_retry_dengue <- c(3549805, 5215231)
warnings_dengue_initial <- city_warnings_dengue  # snapshot before reset, so it can be saved below
city_warnings_dengue <- list()
preds_dengue_retry   <- list()
resolved_formulas_dengue <- tibble(city = character(), formula_id = character(), order = character(), row = integer())

for (ct in cities_to_retry_dengue) {
  file_name <- paste0("processed_data/dengue/sel_cities/dengue_", ct, "_agg.csv.gz")
  d <- read_csv(file_name, show_col_types = FALSE)
  train_rows <- d$train_1 == 1

  candidates <- get_candidates(d)
  candidates <- filter_low_variance(d, candidates, threshold = 0.01)
  candidates <- filter_by_correlation(
    d[train_rows, ],
    candidates,
    min_cor = 0.1
  )

  pca_result <- pca_all(
    data       = d,
    candidates = candidates[!grepl("enso|iod|pdo", candidates)],
    var_threshold = 0.9
  )
  d <- pca_result$data

  metrics_file <- paste0("sarimax/results/metrics/metrics_all_formulas_dengue_", ct, ".csv")
  metrics_df   <- read_csv(metrics_file, show_col_types = FALSE)

  i          <- 2
  resolved   <- FALSE

  while (i <= nrow(metrics_df) && !resolved) {
    message(sprintf("[%s] Trying formula row %d of %d: %s",
                    ct, i, nrow(metrics_df), metrics_df$formula_id[i]))

    best_order   <- metrics_df$order[i]
    ord          <- parse_order(best_order)
    best_formula <- metrics_df$formula_id[i]
    train_ids    <- paste0("train_", 1:4)
    target_ids   <- paste0("target_", 1:4)

    # Reset warnings for this attempt
    attempt_warnings <- list()

    pred_target <- lapply(seq_along(train_ids), function(j) {
      train_id  <- train_ids[j]
      target_id <- target_ids[j]

      # ── Catch hard Arima() errors (e.g. "non-finite finite-difference
      #    value") so a single bad fit doesn't crash the run; it counts as
      #    actionable and the while-loop just tries the next formula row.
      fit <- tryCatch({
        if (train_id == "train_4") {
          fit_sarimax_epiweek(
            data           = d,
            formula        = best_formula,
            train_start    = 201001,
            train_end      = 202525,
            forecast_start = 202541,
            forecast_end   = 202640,
            order          = c(ord$order[1], ord$order[2], ord$order[3]),
            seasonal       = list(order = c(ord$seasonal_order[1], ord$seasonal_order[2], ord$seasonal_order[3]), period = 52)
          )
        } else {
          fit_sarimax(
            data     = d,
            formula  = best_formula,
            order    = c(ord$order[1], ord$order[2], ord$order[3]),
            seasonal = list(order = c(ord$seasonal_order[1], ord$seasonal_order[2], ord$seasonal_order[3]), period = 52),
            train_id = train_id
          )
        }
      }, error = function(e) {
        err_msg <- paste0("Arima() failed: ", conditionMessage(e))
        attempt_warnings[[target_id]] <<- c(attempt_warnings[[target_id]], err_msg)
        message(sprintf("[%s | %s] ERROR: %s", ct, target_id, err_msg))
        NULL
      })
      if (is.null(fit)) return(NULL)

      w <- attr(fit, "warnings")
      if (!is.null(w)) {
        attempt_warnings[[target_id]] <<- w
        message(sprintf("[%s | %s] %d warning(s):\n%s",
                        ct, target_id, length(w),
                        paste0("  - ", w, collapse = "\n")))
      }

      # ── Flag degenerate (all-zero) point forecasts as a reason to keep
      #    trying the next formula row, same as a real Arima() warning ──
      if (is_degenerate_forecast(fit)) {
        attempt_warnings[[target_id]] <<- c(attempt_warnings[[target_id]],
          "Degenerate forecast: all point predictions are zero")
        message(sprintf("[%s | %s] Degenerate forecast: all point predictions are zero", ct, target_id))
      }

      fit
    })
    names(pred_target) <- target_ids

    # Check if this attempt is clean (no actionable warnings, no hard errors,
    # no degenerate all-zero forecasts)
    actionable <- unlist(lapply(attempt_warnings, function(w) {
      any(grepl("auto.arima|unreliable|failed|degenerate", w, ignore.case = TRUE))
    }))
    has_error <- any(vapply(pred_target, is.null, logical(1)))

    if ((length(actionable) == 0 || !any(actionable)) && !has_error) {
      # Clean fit — save results and move on
      resolved <- TRUE
      city_warnings_dengue[[ct]] <- attempt_warnings

      for (j in seq_along(train_ids)) {
        write_csv(
          pred_target[[j]],
          file.path("sarimax/results/preds/",
                    paste0("pred_dengue_", ct, "_", target_ids[j], ".csv"))
        )
      }

      preds_dengue_retry[[ct]] <- bind_rows(pred_target, .id = "target_id") |>
        mutate(city = ct)

      message(sprintf("[%s] Resolved with formula row %d: %s", ct, i, best_formula))
      resolved_formulas_dengue <- resolved_formulas_dengue |>
        bind_rows(tibble(city = ct, formula_id = best_formula, order = best_order, row = i))
    } else {
      message(sprintf("[%s] Formula row %d still has warnings; trying next.", ct, i))
    }

    i <- i + 1
  }

  if (!resolved) {
    message(sprintf("[%s] All %d formulas exhausted; could not resolve warnings.", ct, nrow(metrics_df)))
    city_warnings_dengue[[ct]] <- attempt_warnings
  }
}

write_csv(resolved_formulas_dengue, "sarimax/results/metrics/resolved_formulas_dengue.csv")

# ── Persist dengue warnings AND caught Arima() errors (both passes) ───────
warnings_dengue_df <- bind_rows(
  to_warnings_df(warnings_dengue_initial, "initial"),
  to_warnings_df(city_warnings_dengue, "retry")
)
write_csv(warnings_dengue_df, "sarimax/results/metrics/warnings_dengue.csv")

best_wis_df_chikungunya_cities <- read_csv("sarimax/results/metrics/best_wis_chikungunya_all_cities.csv", show_col_types = FALSE)
city_warnings_chikungunya <- list()

preds_chikungunya_cities <- lapply(best_wis_df_chikungunya_cities$city, function(ct) {
  if (ct != 4219507) return(NULL)
  file_name <- paste0("processed_data/chikungunya/sel_cities/chikungunya_", ct, "_agg.csv.gz")
  d <- read_csv(file_name, show_col_types = FALSE)
  train_rows <- d$train_1 == 1

  candidates <- get_candidates(d)
  candidates <- filter_low_variance(d, candidates, threshold = 0.01)
  candidates <- filter_by_correlation(
    d[train_rows, ],
    candidates,
    min_cor = 0.1
  )

  if (length(candidates) == 0) {
    candidates <- get_candidates(dengue_state)
    candidates <- filter_low_variance(
      d[
        d[[train_ids[1]]] == 1, 
      ],
      candidates, 
      threshold = threshold_low_variance
    )
  }

  pca_result <- pca_all(
    data = d,
    candidates = candidates[!grepl("enso|iod|pdo", candidates)],
    var_threshold = 0.9
  )
  d <- pca_result$data

  best_order <- best_wis_df_chikungunya_cities |> filter(city == ct) |> pull(order)
  ord <- parse_order(best_order)
  best_formula <- best_wis_df_chikungunya_cities |> filter(city == ct) |> pull(formula_id)
  train_ids  <- paste0("train_", 1:4)
  target_ids <- paste0("target_", 1:4)
  pred_target <- lapply(seq_along(train_ids), function(i) {
    train_id <- train_ids[i]
    target_id <- target_ids[i]
    # ── Catch hard Arima() errors (e.g. "non-finite finite-difference value")
    #    so a single bad fit doesn't crash the whole run; treat it like a
    #    warning that needs a retry with a different formula/order.
    fit <- tryCatch({
      if (train_id == "train_4") {
        fit_sarimax_epiweek(
          data = d,
          formula = best_formula,
          train_start = 201001,
          train_end = 202525,
          forecast_start = 202541,
          forecast_end = 202640,
          order = c(ord$order[1], ord$order[2], ord$order[3]),
          seasonal = list(order = c(ord$seasonal_order[1], ord$seasonal_order[2], ord$seasonal_order[3]), period = 52),
          optim.method = "BFGS"
        )
      } else {
        fit_sarimax(
          data = d,
          formula = best_formula,
          order = c(ord$order[1], ord$order[2], ord$order[3]),
          seasonal = list(order = c(ord$seasonal_order[1], ord$seasonal_order[2], ord$seasonal_order[3]), period = 52),
          train_id = train_id
        )
      }
    }, error = function(e) {
      err_msg <- paste0("Arima() failed: ", conditionMessage(e))
      city_warnings_chikungunya[[as.character(ct)]] <<- c(city_warnings_chikungunya[[as.character(ct)]],
        setNames(err_msg, target_id))
      message(sprintf("[%s | %s] ERROR: %s", ct, target_id, err_msg))
      NULL
    })
    if (is.null(fit)) return(NULL)

    # ── Report warnings with city/target context ───────────────────────────
    w <- attr(fit, "warnings")
    if (!is.null(w)) {
      city_warnings_chikungunya[[as.character(ct)]] <<- c(city_warnings_chikungunya[[as.character(ct)]],
        setNames(w, rep(target_id, length(w))))
      message(sprintf("[%s | %s] %d warning(s):\n%s",
                      ct, target_id, length(w),
                      paste0("  - ", w, collapse = "\n")))
    }

    # ── Flag degenerate (all-zero) point forecasts so this city is retried
    #    with the next-best formula/order, same as a real Arima() warning ──
    if (is_degenerate_forecast(fit)) {
      city_warnings_chikungunya[[as.character(ct)]] <<- c(city_warnings_chikungunya[[as.character(ct)]],
        setNames("Degenerate forecast: all point predictions are zero", target_id))
      message(sprintf("[%s | %s] Degenerate forecast: all point predictions are zero", ct, target_id))
    }
    write_csv(fit, file.path("sarimax/results/preds/", paste0("pred_chikungunya_", ct, "_", target_id, ".csv")))
    fit
  })
  names(pred_target) <- target_ids
  bind_rows(pred_target, .id = "target_id") |>
    mutate(city = ct)
})

cities_to_retry_chikungunya <- names(city_warnings_chikungunya)
# cities_to_retry_chikungunya <- c("4104808", "4219507")
warnings_chikungunya_initial <- city_warnings_chikungunya  # snapshot before reset, so it can be saved below
city_warnings_chikungunya <- list()
preds_chikungunya_retry   <- list()
resolved_formulas_chikungunya <- tibble(city = character(), formula_id = character(), order = character(), row = integer())

for (ct in cities_to_retry_chikungunya) {
  file_name <- paste0("processed_data/chikungunya/sel_cities/chikungunya_", ct, "_agg.csv.gz")
  d <- read_csv(file_name, show_col_types = FALSE)
  train_rows <- d$train_1 == 1

  candidates <- get_candidates(d)
  candidates <- filter_low_variance(d, candidates, threshold = 0.01)
  candidates <- filter_by_correlation(
    d[train_rows, ],
    candidates,
    min_cor = 0.1
  )

  pca_result <- pca_all(
    data       = d,
    candidates = candidates[!grepl("enso|iod|pdo", candidates)],
    var_threshold = 0.9
  )
  d <- pca_result$data

  metrics_file <- paste0("sarimax/results/metrics/metrics_all_formulas_chikungunya_", ct, ".csv")
  metrics_df   <- read_csv(metrics_file, show_col_types = FALSE)

  i          <- 2
  resolved   <- FALSE

  while (i <= nrow(metrics_df) && !resolved) {
    message(sprintf("[%s] Trying formula row %d of %d: %s",
                    ct, i, nrow(metrics_df), metrics_df$formula_id[i]))

    best_order   <- metrics_df$order[i]
    ord          <- parse_order(best_order)
    best_formula <- metrics_df$formula_id[i]
    train_ids    <- paste0("train_", 1:4)
    target_ids   <- paste0("target_", 1:4)

    # Reset warnings for this attempt
    attempt_warnings <- list()

    pred_target <- lapply(seq_along(train_ids), function(j) {
      train_id  <- train_ids[j]
      target_id <- target_ids[j]

      # ── Catch hard Arima() errors (e.g. "non-finite finite-difference
      #    value") so a single bad fit doesn't crash the run; it counts as
      #    actionable and the while-loop just tries the next formula row.
      fit <- tryCatch({
        if (train_id == "train_4") {
          fit_sarimax_epiweek(
            data           = d,
            formula        = best_formula,
            train_start    = 201001,
            train_end      = 202525,
            forecast_start = 202541,
            forecast_end   = 202640,
            order          = c(ord$order[1], ord$order[2], ord$order[3]),
            seasonal       = list(order = c(ord$seasonal_order[1], ord$seasonal_order[2], ord$seasonal_order[3]), period = 52),
            optim.method   = "BFGS"
          )
        } else {
          fit_sarimax(
            data     = d,
            formula  = best_formula,
            order    = c(ord$order[1], ord$order[2], ord$order[3]),
            seasonal = list(order = c(ord$seasonal_order[1], ord$seasonal_order[2], ord$seasonal_order[3]), period = 52),
            train_id = train_id
          )
        }
      }, error = function(e) {
        err_msg <- paste0("Arima() failed: ", conditionMessage(e))
        attempt_warnings[[target_id]] <<- c(attempt_warnings[[target_id]], err_msg)
        message(sprintf("[%s | %s] ERROR: %s", ct, target_id, err_msg))
        NULL
      })
      if (is.null(fit)) return(NULL)

      w <- attr(fit, "warnings")
      if (!is.null(w)) {
        attempt_warnings[[target_id]] <<- w
        message(sprintf("[%s | %s] %d warning(s):\n%s",
                        ct, target_id, length(w),
                        paste0("  - ", w, collapse = "\n")))
      }

      # ── Flag degenerate (all-zero) point forecasts as a reason to keep
      #    trying the next formula row, same as a real Arima() warning ──
      if (is_degenerate_forecast(fit)) {
        attempt_warnings[[target_id]] <<- c(attempt_warnings[[target_id]],
          "Degenerate forecast: all point predictions are zero")
        message(sprintf("[%s | %s] Degenerate forecast: all point predictions are zero", ct, target_id))
      }

      fit
    })
    names(pred_target) <- target_ids

    # Check if this attempt is clean (no actionable warnings, no hard errors,
    # no degenerate all-zero forecasts)
    actionable <- unlist(lapply(attempt_warnings, function(w) {
      any(grepl("auto.arima|unreliable|failed|degenerate", w, ignore.case = TRUE))
    }))
    has_error <- any(vapply(pred_target, is.null, logical(1)))

    if ((length(actionable) == 0 || !any(actionable)) && !has_error) {
      # Clean fit — save results and move on
      resolved <- TRUE
      city_warnings_chikungunya[[ct]] <- attempt_warnings

      for (j in seq_along(train_ids)) {
        write_csv(
          pred_target[[j]],
          file.path("sarimax/results/preds/",
                    paste0("pred_chikungunya_", ct, "_", target_ids[j], ".csv"))
        )
      }

      preds_chikungunya_retry[[ct]] <- bind_rows(pred_target, .id = "target_id") |>
        mutate(city = ct)

      message(sprintf("[%s] Resolved with formula row %d: %s", ct, i, best_formula))
      resolved_formulas_chikungunya <- resolved_formulas_chikungunya |>
        bind_rows(tibble(city = ct, formula_id = best_formula, order = best_order, row = i))
    } else {
      message(sprintf("[%s] Formula row %d still has warnings; trying next.", ct, i))
    }

    i <- i + 1
  }

  if (!resolved) {
    message(sprintf("[%s] All %d formulas exhausted; could not resolve warnings.", ct, nrow(metrics_df)))
    city_warnings_chikungunya[[ct]] <- attempt_warnings
  }
}

write_csv(resolved_formulas_chikungunya, "sarimax/results/metrics/resolved_formulas_chikungunya.csv")

# ── Persist chikungunya warnings AND caught Arima() errors (both passes) ──
warnings_chikungunya_df <- bind_rows(
  to_warnings_df(warnings_chikungunya_initial, "initial"),
  to_warnings_df(city_warnings_chikungunya, "retry")
)
write_csv(warnings_chikungunya_df, "sarimax/results/metrics/warnings_chikungunya.csv")


# Solve degenerate forecasts 
city <- 4219507
id <- 3
preds_chikungunya_cities <- lapply(best_wis_df_chikungunya_cities$city, function(ct) {
  if (ct != city) return(NULL)
  file_name <- paste0("processed_data/chikungunya/sel_cities/chikungunya_", ct, "_agg.csv.gz")
  d <- read_csv(file_name, show_col_types = FALSE)
  train_rows <- d$train_1 == 1

  candidates <- get_candidates(d)
  candidates <- filter_low_variance(d, candidates, threshold = 0.01)
  candidates <- filter_by_correlation(
    d[train_rows, ],
    candidates,
    min_cor = 0.1
  )

  if (length(candidates) == 0) {
    candidates <- get_candidates(dengue_state)
    candidates <- filter_low_variance(
      d[
        d[[train_ids[1]]] == 1, 
      ],
      candidates, 
      threshold = threshold_low_variance
    )
  }

  pca_result <- pca_all(
    data = d,
    candidates = candidates[!grepl("enso|iod|pdo", candidates)],
    var_threshold = 0.9
  )
  d <- pca_result$data

  metrics_file <- paste0("sarimax/results/metrics/metrics_all_formulas_chikungunya_", ct, ".csv")
  metrics_df   <- read_csv(metrics_file, show_col_types = FALSE)
  best_order <- metrics_df$order[id]
  ord <- parse_order(best_order)
  best_formula <- metrics_df$formula_id[id]
  train_ids  <- paste0("train_", 1:4)
  target_ids <- paste0("target_", 1:4)
  pred_target <- lapply(seq_along(train_ids), function(i) {
    train_id <- train_ids[i]
    target_id <- target_ids[i]
    # ── Catch hard Arima() errors (e.g. "non-finite finite-difference value")
    #    so a single bad fit doesn't crash the whole run; treat it like a
    #    warning that needs a retry with a different formula/order.
    fit <- tryCatch({
      if (train_id == "train_4") {
        fit_sarimax_epiweek(
          data = d,
          formula = best_formula,
          train_start = 201001,
          train_end = 202525,
          forecast_start = 202541,
          forecast_end = 202640,
          order = c(ord$order[1], ord$order[2], ord$order[3]),
          seasonal = list(order = c(ord$seasonal_order[1], ord$seasonal_order[2], ord$seasonal_order[3]), period = 52),
          optim.method = "BFGS"
        )
      } else {
        fit_sarimax(
          data = d,
          formula = best_formula,
          order = c(ord$order[1], ord$order[2], ord$order[3]),
          seasonal = list(order = c(ord$seasonal_order[1], ord$seasonal_order[2], ord$seasonal_order[3]), period = 52),
          train_id = train_id
        )
      }
    }, error = function(e) {
      err_msg <- paste0("Arima() failed: ", conditionMessage(e))
      city_warnings_chikungunya[[as.character(ct)]] <<- c(city_warnings_chikungunya[[as.character(ct)]],
        setNames(err_msg, target_id))
      message(sprintf("[%s | %s] ERROR: %s", ct, target_id, err_msg))
      NULL
    })
    if (is.null(fit)) return(NULL)

    # ── Report warnings with city/target context ───────────────────────────
    w <- attr(fit, "warnings")
    if (!is.null(w)) {
      city_warnings_chikungunya[[as.character(ct)]] <<- c(city_warnings_chikungunya[[as.character(ct)]],
        setNames(w, rep(target_id, length(w))))
      message(sprintf("[%s | %s] %d warning(s):\n%s",
                      ct, target_id, length(w),
                      paste0("  - ", w, collapse = "\n")))
    }

    # ── Flag degenerate (all-zero) point forecasts so this city is retried
    #    with the next-best formula/order, same as a real Arima() warning ──
    if (is_degenerate_forecast(fit)) {
      city_warnings_chikungunya[[as.character(ct)]] <<- c(city_warnings_chikungunya[[as.character(ct)]],
        setNames("Degenerate forecast: all point predictions are zero", target_id))
      message(sprintf("[%s | %s] Degenerate forecast: all point predictions are zero", ct, target_id))
    }
    write_csv(fit, file.path("sarimax/results/preds/", paste0("pred_chikungunya_", ct, "_", target_id, ".csv")))
    fit
  })
  names(pred_target) <- target_ids
  bind_rows(pred_target, .id = "target_id") |>
    mutate(city = ct)
})
