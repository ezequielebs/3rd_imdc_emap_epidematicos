library(forecast)
library(dplyr)
library(tidyverse)
library(tidyr)
library(purrr)
library(lubridate)
library(scoringutils)
library(parallel)
library(pbapply)

# AR-model counterpart of sarimax/src/utils.r. fit_ar() is adapted from that
# file's fit_sarimax() with the unused `formula`/xreg machinery stripped out
# (AR models have no exogenous regressors), and is always called here with
# order = c(p, 0, 0) and seasonal = c(0, 0, 0) -- i.e. no differencing, no MA
# term, no seasonal component at all. That makes this a genuine AR(p) model,
# not an ARIMA/SARIMA model fit through the same engine.
#
# compute_metrics() is identical to sarimax's version. run_ar_grid_search()
# replaces run_grid_search()/run_order_grid_search(): it grid-searches only
# over p (no d, q, P, D, Q knobs at all).

#' Fit an AR(p) model (no exogenous regressors, no differencing/MA/seasonal
#' terms) and return forecasts with prediction intervals.
#'
#' @param data        A data frame containing `cases`, `date`, and train/target
#'                    indicator columns.
#' @param train_id    Character string: one of "train_1","train_2","train_3","train_4"
#' @param levels      Numeric vector of coverage levels, e.g. c(50, 80, 95).
#' @param order       ARIMA (p,d,q) order -- pass c(p, 0, 0) for AR(p).
#' @param seasonal    Seasonal ARIMA list -- pass list(order = c(0,0,0), period = 1)
#'                    for a non-seasonal AR(p).
fit_ar <- function(data,
                   train_id,
                   method = "CSS-ML",
                   lambda = NULL,
                   optim.control = list(maxit = 500),   # more iterations
                   optim.method  = "BFGS",
                   levels = c(50, 80, 90, 95),
                   order      = c(1, 0, 0),
                   seasonal   = list(order = c(0, 0, 0), period = 52),
                   bootstrap  = TRUE,   # simulate forecast paths instead of
                   npaths     = 1000) { # assuming Gaussian normal-theory intervals

  stopifnot(is.data.frame(data))
  stopifnot(train_id %in% paste0("train_", 1:4))

  target_id <- sub("train_", "target_", train_id)
  stopifnot(target_id %in% names(data))

  # ── 1. Split train / target rows ─────────────────────────────────────────
  train_rows  <- data[data[[train_id]]  == 1, ]
  target_rows <- data[data[[target_id]] == 1, ]

  if (nrow(train_rows)  == 0) stop("No training rows found for ", train_id)
  if (nrow(target_rows) == 0) stop("No target rows found for ",  target_id)

  # ── 2. Log-transform response (log1p to handle zero counts) ──────────────
  if (is.null(lambda)) {
    y <- log1p(train_rows$cases)
  } else {
    y <- train_rows$cases
  }

  # ── 4. Fit AR(p) — capture warnings ───────────────────────────────────────
  y_ts <- ts(y, frequency = seasonal$period)
  fit_warnings <- character(0)

  fit <- withCallingHandlers(
    Arima(y_ts,
          order    = order,
          seasonal = seasonal,
          xreg     = NULL,
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

  # ── 5. Detect NaN standard errors ─────────────────────────────────────────
  nan_se_params <- names(which(is.nan(sqrt(diag(fit$var.coef)))))
  if (length(nan_se_params) > 0) {
    fit_warnings <- c(
      fit_warnings,
      paste0(
        "NaN standard errors for: ",
        paste(nan_se_params, collapse = ", "),
        ". Prediction intervals may be unreliable. ",
        "Consider reducing model order or using auto.arima()."
      )
    )
  }

  # ── 6. Forecast & back-transform ──────────────────────────────────────────
  h      <- nrow(target_rows)
  levels <- sort(unique(levels))

  fc_warnings <- character(0)

  fc <- withCallingHandlers(
    forecast(fit, h = h, xreg = NULL, level = levels,
             bootstrap = bootstrap, npaths = npaths),
    warning = function(w) {
      fc_warnings <<- c(fc_warnings, conditionMessage(w))
      invokeRestart("muffleWarning")
    }
  )

  # Back-transform from log1p scale: expm1(x) = exp(x) - 1
  bt <- expm1  # shorthand

  # ── 7. Access date ─────────────────────────────────────────────
  dates <- data$date[data[[target_id]] == 1]

  # ── 8. Assemble output tibble ─────────────────────────────────────────────
  if (is.null(lambda)) {
    out <- tibble(
      date = dates,
      pred = bt(as.numeric(fc$mean))
    )
  } else {
    out <- tibble(
      date = dates,
      pred = as.numeric(fc$mean)
    )
  }

  for (lv in levels) {
    lv_char <- paste0(lv, "%")
    if (is.null(lambda)) {
      out[[paste0("lower_", lv)]] <- bt(as.numeric(fc$lower[, lv_char]))
      out[[paste0("upper_", lv)]] <- bt(as.numeric(fc$upper[, lv_char]))
    } else {
      out[[paste0("lower_", lv)]] <- as.numeric(fc$lower[, lv_char])
      out[[paste0("upper_", lv)]] <- as.numeric(fc$upper[, lv_char])
    }
  }

  # Clip to zero (back-transform should already ensure this, but be safe)
  out <- out |>
    mutate(across(c(pred, starts_with("lower_"), starts_with("upper_")),
                  \(x) pmax(x, 0)))

  # ── 9. Attach warnings as attribute ───────────────────────────────────────
  all_warnings <- c(fit_warnings, fc_warnings)

  attr(out, "warnings") <- if (length(all_warnings) > 0) all_warnings else NULL
  attr(out, "fit")      <- fit

  out
}

# Scoringutils WIS helper — identical to sarimax/src/utils.r::compute_metrics(),
# just labelled "ar" instead of "sarimax" for the scoringutils model column.
compute_metrics <- function(
  pred_df,
  actual,
  levels = c(50, 80, 90, 95)
) {
  levels     <- sort(levels)
  lower_cols <- paste0("lower_", levels)
  upper_cols <- paste0("upper_", levels)
  alphas     <- 1 - levels / 100

  n <- nrow(pred_df)

  make_quantile_rows <- function(predicted, quantile_level) {
    tibble(
      time           = seq_len(n),
      observed       = actual,
      predicted      = predicted,
      quantile_level = quantile_level,
      model          = "ar"
    )
  }

  su <- bind_rows(
    lapply(seq_along(levels), function(i)
      make_quantile_rows(pred_df[[lower_cols[i]]], alphas[i] / 2)),
    lapply(seq_along(levels), function(i)
      make_quantile_rows(pred_df[[upper_cols[i]]], 1 - alphas[i] / 2)),
    make_quantile_rows(pred_df$pred, 0.5)
  ) |>
    arrange(time, quantile_level) |>
    as_forecast_quantile(
      observed       = "observed",
      predicted      = "predicted",
      quantile_level = "quantile_level",
      model          = "model"
    )

  sc <- withCallingHandlers(
    score(su) |> summarise_scores(by = "model"),
    warning = function(w) {
      if (grepl("interval_coverage", conditionMessage(w)))
        invokeRestart("muffleWarning")
    }
  )

  coverage <- setNames(
    sapply(seq_along(levels), function(i) {
      mean(actual >= pred_df[[lower_cols[i]]] &
           actual <= pred_df[[upper_cols[i]]])
    }),
    paste0("coverage_", levels)
  )

  y    <- actual
  yhat <- pred_df$pred

  c(
    list(
      wis  = sc$wis,
      mae  = mean(abs(y - yhat)),
      mse  = mean((y - yhat)^2),
      rmse = sqrt(mean((y - yhat)^2)),
      mape = mean(abs((y - yhat) / pmax(y, 1))) * 100
    ),
    as.list(coverage)
  )
}

# ── Main function ─────────────────────────────────────────────────────────────

#' Grid search over AR(p) lag order with CV evaluation. No d, q, or seasonal
#' (P,D,Q) terms exist at all in this search -- order is always (p, 0, 0) and
#' seasonal is always (0, 0, 0). This is the AR-model counterpart of
#' sarimax/src/utils.r::run_grid_search(), with both the covariate dimension
#' and every non-AR order component removed.
#'
#' @param data       Data frame with `cases` and train/target indicators.
#' @param levels     Numeric vector of coverage levels, e.g. c(50, 80, 95).
#' @param max_p      Largest AR lag order to try. Grid is p = 1..max_p.
#' @param fixed_p    FALSE (default) to search p = 1..max_p, or a single
#'                   integer to evaluate just that one lag order.
#' @param n_cores    Number of parallel workers. Defaults to cores - 1.
#'
#' @return A list with two tibbles:
#'   $predictions : one row per p x train_id x date
#'   $metrics     : one row per p x train_id, with all scores
run_ar_grid_search <- function(data,
                               levels = c(50, 80, 90, 95),
                               method = "CSS-ML",
                               lambda = NULL,
                               optim.control = list(maxit = 500),
                               optim.method  = "BFGS",
                               max_p   = 10,
                               fixed_p = FALSE,
                               bootstrap = TRUE,
                               npaths    = 1000,
                               n_cores   = max(1L, detectCores() - 1L)) {

  p_grid <- if (!isFALSE(fixed_p)) fixed_p else seq(1, max_p)

  train_ids <- paste0("train_", 1:4)
  n_orders  <- length(p_grid)

  message(sprintf(
    "Grid: %d AR order(s) x %d splits = %d total fits.",
    n_orders, length(train_ids), n_orders * length(train_ids)
  ))

  # ── Parallel backend ────────────────────────────────────────────────────────
  cl <- makeCluster(n_cores)
  on.exit(stopCluster(cl), add = TRUE)

  clusterExport(cl, varlist = c(
    "fit_ar", "compute_metrics",
    "data", "p_grid",
    "levels", "train_ids",
    "method", "lambda", "optim.control", "optim.method",
    "bootstrap", "npaths"
  ), envir = environment())

  clusterEvalQ(cl, {
    library(forecast)
    library(dplyr)
    library(tidyr)
    library(scoringutils)
  })

  message(sprintf("Running %d jobs on %d core(s)...", n_orders, n_cores))
  start_time <- proc.time()["elapsed"]

  # ── Parallel loop over p ───────────────────────────────────────────────────
  op <- pbapply::pboptions(type = "timer")
  pbapply::pboptions(op)
  all_results <- pbapply::pblapply(
    seq_along(p_grid),
    function(pi) {
      p         <- p_grid[pi]
      order_str <- sprintf("AR(%d)", p)

      split_results <- lapply(train_ids, function(train_id) {
        target_id <- sub("train_", "target_", train_id)
        actual    <- data$cases[data[[target_id]] == 1]

        preds <- tryCatch(
          fit_ar(data,
                 train_id = train_id,
                 levels   = levels,
                 method   = method,
                 order    = c(p, 0, 0),
                 seasonal = list(order = c(0, 0, 0), period = 1),
                 lambda   = lambda,
                 optim.control = optim.control,
                 optim.method  = optim.method,
                 bootstrap = bootstrap,
                 npaths    = npaths),
          error = function(e) NULL
        )

        failed  <- is.null(preds) || nrow(preds) != length(actual)
        metrics <- if (!failed) {
          tryCatch(as_tibble(compute_metrics(preds, actual)), error = function(e) NULL)
        } else NULL

        list(
          predictions = if (!failed) {
            preds |> mutate(order = order_str, train_id = train_id, failed = FALSE)
          } else {
            tibble(order = order_str, train_id = train_id, failed = TRUE)
          },
          metrics = if (!is.null(metrics)) {
            metrics |> mutate(order = order_str, train_id = train_id, failed = FALSE)
          } else {
            tibble(order = order_str, train_id = train_id, failed = TRUE)
          }
        )
      })

      split_results
    },
    cl = cl
  )

  # ── Progress summary ────────────────────────────────────────────────────────
  elapsed_total <- proc.time()["elapsed"] - start_time
  fmt_time <- function(secs) {
    if (secs < 60)    return(sprintf("%.0fs", secs))
    if (secs < 3600)  return(sprintf("%.0fm %.0fs", secs %/% 60, secs %% 60))
    sprintf("%.0fh %.0fm", secs %/% 3600, (secs %% 3600) %/% 60)
  }

  # ── Flatten and bind ────────────────────────────────────────────────────────
  flat <- unlist(all_results, recursive = FALSE)

  predictions <- bind_rows(lapply(flat, `[[`, "predictions"))
  metrics     <- bind_rows(lapply(flat, `[[`, "metrics"))

  n_failed <- sum(metrics$failed, na.rm = TRUE)
  message(sprintf("Done. Total time: %s. Failed fits: %d / %d.",
                  fmt_time(elapsed_total), n_failed,
                  n_orders * length(train_ids)))

  list(predictions = predictions, metrics = metrics)
}
