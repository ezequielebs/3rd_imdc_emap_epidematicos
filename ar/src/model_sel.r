source("ar/src/utils.r")

# AR-model counterpart of sarimax/src/model_sel.r: same per-state CV/WIS
# pipeline, but fitting a genuine AR(p) model -- no exogenous covariates, no
# differencing, no MA term, no seasonal component. run_ar_grid_search() grid
# searches only over the lag order p. Useful as a baseline against the
# SARIMAX-with-covariates results in sarimax/results/metrics/best_wis_all_states.csv.

run_model_selection <- function(
  state,
  concluded_states,
  metric = "wis",
  levels = c(50, 80, 90, 95),
  max_p = 10,                 # grid search p = 1..max_p
  n_cores = parallel::detectCores() - 1,
  method = "CSS-ML",
  lambda = NULL,
  bootstrap = TRUE,           # simulate forecast paths for better-calibrated intervals
  npaths = 1000,
  concluded_states_path = "ar/results/concluded_states.csv"  # overridable so test
                                                              # runs don't touch the
                                                              # real checkpoint file
) {
  if (state %in% concluded_states$state) {
    message("State ", state, " already processed. Skipping.")
    return(NULL)
  }
  message("Processing state: ", state)
  file_name <- paste0("processed_data/dengue/dengue_", state, "_agg.csv.gz")
  dengue_state <- read_csv(file_name, show_col_types = FALSE)

  order_screen <- run_ar_grid_search(
    data      = dengue_state,
    levels    = levels,
    max_p     = max_p,
    n_cores   = n_cores,
    method    = method,
    lambda    = lambda,
    bootstrap = bootstrap,
    npaths    = npaths
  )

  metric_name <- paste0("mean_", metric)
  best_par <- order_screen$metrics |>
    group_by(order) |>
    filter(all(!failed)) |>
    summarise(mean_metric = mean(get(metric)), .groups = "drop") |>
    arrange(mean_metric) |>
    slice(1)
  best_order <- best_par$order

  metrics <- order_screen$metrics |>
    group_by(order) |>
    filter(all(!failed)) |>
    summarise(mean_metric = mean(get(metric)), .groups = "drop") |>
    arrange(mean_metric)

  best_pred <- order_screen$predictions |> filter(order == best_order)

  train_ids  <- paste0("train_", 1:4)
  target_ids <- paste0("target_", 1:4)

  final_metrics <- lapply(seq_along(train_ids), function(i) {
    actual  <- dengue_state$cases[dengue_state[[target_ids[i]]] == 1]
    pred    <- best_pred |> filter(train_id == train_ids[i])
    compute_metrics(pred, actual)
  }) |> bind_rows() |> mutate(target_id = target_ids)

  # save files
  write_csv(final_metrics, file.path("ar/results/metrics/", paste0(metric_name, "_best_model_", state, ".csv")))
  write_csv(metrics, file.path("ar/results/metrics/", paste0("metrics_all_orders_", state, ".csv")))

  # update concluded states
  concluded_states <- rbind(concluded_states, data.frame(state = state, stringsAsFactors = FALSE))
  write_csv(concluded_states, concluded_states_path)

  return(list(
    best_order = best_order,
    best_pred = best_pred,
    final_metrics = final_metrics,
    all_metrics = metrics
  ))
}


states <- c("AC", "AL", "AM", "AP", "BA", "CE", "DF", "GO", "MA",
            "MG", "MS", "MT", "PA", "PB", "PE", "PI", "PR", "RJ",
            "RN", "RO", "RR", "RS", "SC", "SE", "SP", "TO")

results_state <- lapply(states, function(st) {
  concluded_states <- tryCatch({
    read_csv("ar/results/concluded_states.csv", show_col_types = FALSE)
  }, error = function(e) {
    data.frame(state = character(0), stringsAsFactors = FALSE)
  })
  run_model_selection(
    state = st,
    concluded_states = concluded_states,
    max_p = 30
  )
}
)

# get summary
best_wis_df <- lapply(states, function(st) {
  file_name <- paste0("ar/results/metrics/metrics_all_orders_", st, ".csv")
  data <- read_csv(file_name, show_col_types = FALSE)
  data[1, ] |> mutate(state = st, .before = 1)
}) |> bind_rows()
write_csv(best_wis_df, "ar/results/metrics/best_wis_all_states.csv")
