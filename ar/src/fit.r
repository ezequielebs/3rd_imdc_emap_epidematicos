source("ar/src/utils.r")

best_wis_df <- read_csv("ar/results/metrics/best_wis_all_states.csv", show_col_types = FALSE)

preds <- lapply(best_wis_df$state, function(st) {
  file_name <- paste0("processed_data/dengue/dengue_", state, "_agg.csv.gz")
  d <- read_csv(file_name, show_col_types = FALSE)
  best_order <- best_wis_df |> filter(state == st) |> pull(order)
  p <- as.integer(sub(".*\\((\\d+)\\).*", "\\1", best_order))
  train_ids  <- paste0("train_", 1:4)
  target_ids <- paste0("target_", 1:4)
  pred_target <- lapply(seq_along(train_ids), function(i) {
    train_id <- train_ids[i]
    target_id <- target_ids[i]
    fit <- fit_ar(d,
                  train_id,
                  method = "CSS-ML",
                  lambda = NULL,
                  optim.control = list(maxit = 500),   # more iterations
                  optim.method  = "BFGS",
                  levels = c(50, 80, 90, 95),
                  order      = c(p, 0, 0),
                  seasonal   = list(order = c(0, 0, 0), period = 52),
                  bootstrap  = TRUE,   # simulate forecast paths instead of
                  npaths     = 1000)
    write_csv(fit, file.path("ar/results/preds/", paste0("pred_dengue_", st, "_", target_id, ".csv")))
  })
  names(pred_target) <- target_ids
  bind_rows(pred_target, .id = "target_id") |>
    mutate(state = st)
})
    