#!/usr/bin/env Rscript

# Fit SARIMAX models for state-level dengue validation windows.
#
# The script reads processed_data/dengue/dengue_<UF>_agg.csv.gz files, fits one
# model for each train_i/target_i split, and writes validation metrics and
# predictions to sarimax/results.

suppressPackageStartupMessages({
  library(dplyr)
  library(forecast)
  library(purrr)
  library(readr)
  library(stringr)
  library(tibble)
  library(tidyr)
})

splits_default <- 1:4
# default_orders <- list(
#   c(0, 1, 1),
#   c(1, 1, 0),
#   c(1, 1, 1),
#   c(2, 1, 1),
#   c(1, 1, 2),
#   c(2, 1, 2),
#   c(2, 0, 1),
#   c(1, 0, 1),
#   c(2, 0, 2),
#   c(3, 1, 1),
#   c(1, 1, 3)
# )
# default_seasonal_orders <- list(
#   c(0, 0, 0),
#   c(1, 0, 0),
#   c(0, 1, 1),
#   c(1, 0, 1)
# )

flat_grid <- crossing(
  p = 1:5,
  d = 1:2,
  q = 1:5,
  P = 0:2,  
  D = 0:1,  
  Q = 0:2   
)

massive_sarimax_grid <- flat_grid %>%
  rowwise() %>% # Evaluate row by row
  mutate(
    order = list(c(p, d, q)),
    seasonal_order = list(c(P, D, Q))
  ) %>%
  ungroup() %>%
  # Keep only the list columns
  select(order, seasonal_order)

default_orders <- unique(massive_sarimax_grid$order)
default_seasonal_orders <- unique(massive_sarimax_grid$seasonal_order)


exog_prefixes <- c(
  "enso",
  "iod",
  "pdo",
  "temp",
  "precip",
  "pressure",
  "rel_humid",
  "thermal",
  "rainy_days"
)

parse_args <- function(args) {
  parsed <- list(
    data_dir = "processed_data/dengue",
    output_dir = "sarimax/results",
    states = character(),
    splits = splits_default,
    max_covariates = 12,
    covariate_counts = c(0, 4, 8, 12),
    corr_threshold = 0.75,
    maxiter = 100,
    metric = "mean_mae",
    quick = FALSE
  )

  i <- 1
  while (i <= length(args)) {
    arg <- args[[i]]

    if (arg == "--data-dir") {
      parsed$data_dir <- args[[i + 1]]
      i <- i + 2
    } else if (arg == "--output-dir") {
      parsed$output_dir <- args[[i + 1]]
      i <- i + 2
    } else if (arg == "--max-covariates") {
      parsed$max_covariates <- as.integer(args[[i + 1]])
      i <- i + 2
    } else if (arg == "--corr-threshold") {
      parsed$corr_threshold <- as.numeric(args[[i + 1]])
      i <- i + 2
    } else if (arg == "--metric") {
      parsed$metric <- args[[i + 1]]
      i <- i + 2
    } else if (arg == "--maxiter") {
      parsed$maxiter <- as.integer(args[[i + 1]])
      i <- i + 2
    } else if (arg == "--quick") {
      parsed$quick <- TRUE
      i <- i + 1
    } else if (arg == "--states") {
      values <- character()
      i <- i + 1
      while (i <= length(args) && !startsWith(args[[i]], "--")) {
        values <- c(values, toupper(args[[i]]))
        i <- i + 1
      }
      parsed$states <- values
    } else if (arg == "--splits") {
      values <- integer()
      i <- i + 1
      while (i <= length(args) && !startsWith(args[[i]], "--")) {
        values <- c(values, as.integer(args[[i]]))
        i <- i + 1
      }
      parsed$splits <- values
    } else if (arg == "--covariate-counts") {
      values <- integer()
      i <- i + 1
      while (i <= length(args) && !startsWith(args[[i]], "--")) {
        values <- c(values, as.integer(args[[i]]))
        i <- i + 1
      }
      parsed$covariate_counts <- values
    } else {
      stop("Unknown argument: ", arg, call. = FALSE)
    }
  }

  parsed$splits <- parsed$splits[parsed$splits %in% splits_default]
  parsed$covariate_counts <- sort(unique(parsed$covariate_counts[parsed$covariate_counts >= 0]))
  parsed$covariate_counts <- parsed$covariate_counts[parsed$covariate_counts <= parsed$max_covariates]
  parsed$metric <- match.arg(parsed$metric, c("mean_mae", "median_mae", "total_absolute_error"))
  parsed
}

state_from_file <- function(path) {
  str_match(basename(path), "^dengue_([A-Z]{2})_agg\\.csv\\.gz$")[, 2]
}

list_state_files <- function(data_dir, states) {
  files <- list.files(
    data_dir,
    pattern = "^dengue_[A-Z]{2}_agg\\.csv\\.gz$",
    full.names = TRUE
  )
  files <- sort(files)

  if (length(states) == 0) {
    return(files)
  }

  files[state_from_file(files) %in% states]
}

candidate_exog_columns <- function(data) {
  excluded <- c(
    "cases",
    "validation_cases",
    "pop",
    "year",
    "uf_code",
    paste0("train_", splits_default),
    paste0("target_", splits_default)
  )

  numeric_columns <- names(data)[map_lgl(data, is.numeric)]
  numeric_columns[
    !numeric_columns %in% excluded &
      str_detect(numeric_columns, paste0("^(", paste(exog_prefixes, collapse = "|"), ")"))
  ]
}

select_exog_columns <- function(data, train_mask, max_covariates, corr_threshold) {
  candidates <- candidate_exog_columns(data)
  target <- log1p(data$cases[train_mask])

  scores <- map_dfr(candidates, function(column) {
    values <- data[[column]][train_mask]

    if (sum(!is.na(values)) < 20 || isTRUE(sd(values, na.rm = TRUE) == 0)) {
      return(tibble(covariate = character(), score = numeric()))
    }

    score <- suppressWarnings(abs(cor(
      values,
      target,
      use = "complete.obs",
      method = "spearman"
    )))

    if (!is.finite(score)) {
      return(tibble(covariate = character(), score = numeric()))
    }

    tibble(covariate = column, score = score)
  }) %>%
    arrange(desc(score))

  selected <- character()

  for (column in scores$covariate) {
    if (length(selected) >= max_covariates) {
      break
    }

    too_correlated <- any(map_lgl(selected, function(chosen) {
      correlation <- suppressWarnings(cor(
        data[[column]][train_mask],
        data[[chosen]][train_mask],
        use = "complete.obs"
      ))

      is.finite(correlation) && abs(correlation) >= corr_threshold
    }))

    if (!too_correlated) {
      selected <- c(selected, column)
    }
  }

  selected
}

load_state_data <- function(files) {
  map(files, function(path) {
    state <- state_from_file(path)
    data <- read_csv(path, show_col_types = FALSE) %>%
      arrange(epiweek)

    list(state = state, data = data, path = path)
  })
}

global_covariate_scores <- function(state_data, splits) {
  candidates <- reduce(
    map(state_data, ~ candidate_exog_columns(.x$data)),
    intersect
  )

  map_dfr(candidates, function(column) {
    split_scores <- map_dbl(state_data, function(item) {
      state_scores <- map_dbl(splits, function(split) {
        train_mask <- item$data[[paste0("train_", split)]] & !is.na(item$data$cases)
        values <- item$data[[column]][train_mask]
        target <- log1p(item$data$cases[train_mask])

        if (sum(!is.na(values)) < 20 || isTRUE(sd(values, na.rm = TRUE) == 0)) {
          return(NA_real_)
        }

        suppressWarnings(abs(cor(
          values,
          target,
          use = "complete.obs",
          method = "spearman"
        )))
      })

      mean(state_scores, na.rm = TRUE)
    })

    tibble(covariate = column, score = mean(split_scores, na.rm = TRUE))
  }) %>%
    filter(is.finite(score)) %>%
    arrange(desc(score))
}

pooled_training_values <- function(state_data, splits, column) {
  unlist(map(state_data, function(item) {
    unlist(map(splits, function(split) {
      train_mask <- item$data[[paste0("train_", split)]] & !is.na(item$data$cases)
      item$data[[column]][train_mask]
    }))
  }), use.names = FALSE)
}

select_global_covariates <- function(state_data, splits, max_covariates, corr_threshold) {
  scores <- global_covariate_scores(state_data, splits)
  selected <- character()

  for (column in scores$covariate) {
    if (length(selected) >= max_covariates) {
      break
    }

    values <- pooled_training_values(state_data, splits, column)
    too_correlated <- any(map_lgl(selected, function(chosen) {
      chosen_values <- pooled_training_values(state_data, splits, chosen)
      complete <- complete.cases(values, chosen_values)

      if (sum(complete) < 20) {
        return(FALSE)
      }

      correlation <- suppressWarnings(cor(values[complete], chosen_values[complete]))
      is.finite(correlation) && abs(correlation) >= corr_threshold
    }))

    if (!too_correlated) {
      selected <- c(selected, column)
    }
  }

  selected
}

fill_missing <- function(x, replacement) {
  if (all(is.na(x))) {
    return(rep(replacement, length(x)))
  }

  for (i in seq_along(x)) {
    if (is.na(x[[i]]) && i > 1) {
      x[[i]] <- x[[i - 1]]
    }
  }

  for (i in rev(seq_along(x))) {
    if (is.na(x[[i]]) && i < length(x)) {
      x[[i]] <- x[[i + 1]]
    }
  }

  replace(x, is.na(x), replacement)
}

build_exog <- function(data, train_mask, forecast_mask, columns) {
  if (length(columns) == 0) {
    return(list(train = NULL, forecast = NULL))
  }

  train <- as.data.frame(data[train_mask, columns, drop = FALSE])
  forecast_data <- as.data.frame(data[forecast_mask, columns, drop = FALSE])

  medians <- map_dbl(train, median, na.rm = TRUE)
  medians[!is.finite(medians)] <- 0

  train <- map2_dfc(train, medians, fill_missing)
  forecast_data <- map2_dfc(forecast_data, medians, fill_missing)
  names(train) <- columns
  names(forecast_data) <- columns

  means <- map_dbl(train, mean)
  sds <- map_dbl(train, sd)
  sds[!is.finite(sds) | sds == 0] <- 1

  list(
    train = sweep(sweep(as.matrix(train), 2, means, "-"), 2, sds, "/"),
    forecast = sweep(sweep(as.matrix(forecast_data), 2, means, "-"), 2, sds, "/")
  )
}

fit_model <- function(y_train, xreg_train, order, seasonal_order, maxiter) {
  y_ts <- ts(y_train, frequency = 52)

  suppressWarnings(
    Arima(
      y_ts,
      order = order,
      seasonal = seasonal_order,
      xreg = xreg_train,
      include.constant = TRUE,
      method = "ML",
      optim.control = list(maxit = maxiter)
    )
  )
}

format_order <- function(order) {
  paste0("(", paste(order, collapse = ", "), ")")
}

format_seasonal_order <- function(seasonal_order) {
  paste0("(", paste(seasonal_order, collapse = ", "), ", 52)")
}

evaluate_spec_split <- function(data, state, split, args, covariates, order, seasonal_order) {
  train_column <- paste0("train_", split)
  target_column <- paste0("target_", split)
  train_mask <- data[[train_column]] & !is.na(data$cases)
  target_mask <- data[[target_column]] & !is.na(data$cases)

  exog <- build_exog(data, train_mask, target_mask, covariates)
  y_train <- log1p(as.numeric(data$cases[train_mask]))
  observed <- as.numeric(data$cases[target_mask])

  fit <- fit_model(y_train, exog$train, order, seasonal_order, args$maxiter)
  forecast_result <- forecast(
    fit,
    h = length(observed),
    xreg = exog$forecast
  )

  predicted <- pmax(expm1(as.numeric(forecast_result$mean)), 0)
  errors <- predicted - observed

  metrics <- tibble(
    state = state,
    split = split,
    train_column = train_column,
    target_column = target_column,
    n_train = sum(train_mask),
    n_target = sum(target_mask),
    mae = mean(abs(errors), na.rm = TRUE),
    total_absolute_error = sum(abs(errors), na.rm = TRUE),
    rmse = sqrt(mean(errors^2, na.rm = TRUE)),
    mape = mean(abs(errors) / if_else(observed == 0, NA_real_, observed), na.rm = TRUE) * 100,
    bias = mean(errors, na.rm = TRUE),
    aic = fit$aic,
    order = format_order(order),
    seasonal_order = format_seasonal_order(seasonal_order),
    n_covariates = length(covariates),
    covariates = paste(covariates, collapse = ";"),
    response_transform = "log1p(cases)",
    prediction_back_transform = "expm1",
    covariates_standardized = length(covariates) > 0
  )

  predictions <- tibble(
    state = state,
    split = split,
    epiweek = data$epiweek[target_mask],
    observed = observed,
    predicted = predicted,
    error = errors,
    absolute_error = abs(errors)
  )

  list(metrics = metrics, predictions = predictions)
}

aggregate_metric <- function(metrics, metric) {
  if (!"mae" %in% names(metrics) || all(is.na(metrics$mae))) {
    return(Inf)
  }

  if (metric == "mean_mae") {
    mean(metrics$mae, na.rm = TRUE)
  } else if (metric == "median_mae") {
    median(metrics$mae, na.rm = TRUE)
  } else {
    sum(metrics$total_absolute_error, na.rm = TRUE)
  }
}

evaluate_global_spec <- function(state_data, splits, args, covariates, order, seasonal_order) {
  metrics_rows <- list()

  for (item in state_data) {
    for (split in splits) {
      metrics <- tryCatch(
        evaluate_spec_split(
          item$data,
          item$state,
          split,
          args,
          covariates,
          order,
          seasonal_order
        )$metrics,
        error = function(e) {
          tibble(
            state = item$state,
            split = split,
            error = conditionMessage(e)
          )
        }
      )

      metrics_rows <- append(metrics_rows, list(metrics))
    }
  }

  metrics <- bind_rows(metrics_rows)

  tibble(
    aggregate_metric = aggregate_metric(metrics, args$metric),
    mean_mae = if ("mae" %in% names(metrics)) mean(metrics$mae, na.rm = TRUE) else Inf,
    median_mae = if ("mae" %in% names(metrics)) median(metrics$mae, na.rm = TRUE) else Inf,
    total_absolute_error = if ("total_absolute_error" %in% names(metrics)) {
      sum(metrics$total_absolute_error, na.rm = TRUE)
    } else {
      Inf
    },
    failed_fits = if ("error" %in% names(metrics)) sum(!is.na(metrics$error)) else 0L,
    order = format_order(order),
    seasonal_order = format_seasonal_order(seasonal_order),
    n_covariates = length(covariates),
    covariates = paste(covariates, collapse = ";"),
    detail = list(metrics)
  )
}

evaluate_final_spec <- function(state_data, splits, args, covariates, order, seasonal_order) {
  metrics_rows <- list()
  prediction_rows <- list()

  for (item in state_data) {
    cat("State ", item$state, ": ", basename(item$path), "\n", sep = "")

    for (split in splits) {
      cat("  split ", split, "\n", sep = "")

      result <- tryCatch(
        evaluate_spec_split(
          item$data,
          item$state,
          split,
          args,
          covariates,
          order,
          seasonal_order
        ),
        error = function(e) {
          list(
            metrics = tibble(
              state = item$state,
              split = split,
              train_column = paste0("train_", split),
              target_column = paste0("target_", split),
              error = conditionMessage(e)
            ),
            predictions = tibble()
          )
        }
      )

      metrics_rows <- append(metrics_rows, list(result$metrics))
      if (nrow(result$predictions) > 0) {
        prediction_rows <- append(prediction_rows, list(result$predictions))
      }
    }
  }

  list(
    metrics = bind_rows(metrics_rows),
    predictions = bind_rows(prediction_rows)
  )
}

main <- function() {
  args <- parse_args(commandArgs(trailingOnly = TRUE))
  dir.create(args$output_dir, recursive = TRUE, showWarnings = FALSE)

  files <- list_state_files(args$data_dir, args$states)
  state_data <- load_state_data(files)
  orders <- if (args$quick) default_orders[1:3] else default_orders
  seasonal_orders <- if (args$quick) default_seasonal_orders[1] else default_seasonal_orders

  global_covariates <- select_global_covariates(
    state_data,
    args$splits,
    args$max_covariates,
    args$corr_threshold
  )
  covariate_counts <- sort(unique(c(args$covariate_counts, length(global_covariates))))
  covariate_counts <- covariate_counts[covariate_counts <= length(global_covariates)]
  if (args$quick) {
    covariate_counts <- sort(unique(c(
      0,
      min(4, length(global_covariates)),
      length(global_covariates)
    )))
  }

  cat("Global covariate ranking selected:\n")
  print(global_covariates)
  cat("\nEvaluating shared SARIMAX grid using ", args$metric, "\n", sep = "")

  grid_rows <- list()
  spec_id <- 1

  for (covariate_count in covariate_counts) {
    covariates <- head(global_covariates, covariate_count)

    for (order in orders) {
      for (seasonal_order in seasonal_orders) {
        cat(
          "  spec ", spec_id,
          ": order=", format_order(order),
          " seasonal=", format_seasonal_order(seasonal_order),
          " covariates=", covariate_count,
          "\n",
          sep = ""
        )

        grid_rows <- append(grid_rows, list(
          evaluate_global_spec(
            state_data,
            args$splits,
            args,
            covariates,
            order,
            seasonal_order
          ) %>%
            mutate(spec_id = spec_id, .before = 1)
        ))
        spec_id <- spec_id + 1
      }
    }
  }

  grid_results <- bind_rows(grid_rows) %>%
    arrange(aggregate_metric, failed_fits)
  best_spec <- grid_results %>% slice(1)
  best_order <- as.integer(str_extract_all(best_spec$order[[1]], "-?\\d+")[[1]])
  best_seasonal_order <- as.integer(str_extract_all(best_spec$seasonal_order[[1]], "-?\\d+")[[1]][1:3])
  best_covariates <- if (best_spec$covariates[[1]] == "") {
    character()
  } else {
    str_split(best_spec$covariates[[1]], ";")[[1]]
  }

  cat("\nBest shared model:\n")
  print(best_spec %>% select(
    spec_id,
    aggregate_metric,
    mean_mae,
    median_mae,
    total_absolute_error,
    failed_fits,
    order,
    seasonal_order,
    n_covariates,
    covariates
  ))

  final <- evaluate_final_spec(
    state_data,
    args$splits,
    args,
    best_covariates,
    best_order,
    best_seasonal_order
  )
  metrics <- final$metrics
  predictions <- final$predictions

  metrics_path <- file.path(args$output_dir, "sarimax_metrics.csv")
  predictions_path <- file.path(args$output_dir, "sarimax_predictions.csv")
  grid_path <- file.path(args$output_dir, "sarimax_model_grid.csv")

  write_csv(metrics, metrics_path)
  write_csv(predictions, predictions_path)
  write_csv(grid_results %>% select(-detail), grid_path)

  cat("\nWrote ", metrics_path, "\n", sep = "")
  cat("Wrote ", predictions_path, "\n", sep = "")
  cat("Wrote ", grid_path, "\n", sep = "")

  if ("mae" %in% names(metrics)) {
    cat("\nMAE summary by split:\n")
    print(metrics %>% group_by(split) %>% summarise(mean_mae = mean(mae, na.rm = TRUE), .groups = "drop"))
  }
}

main()
