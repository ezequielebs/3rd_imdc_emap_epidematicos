
# Configurable runner for the SARIMAX model-selection pipeline
# (sarimax/src/model_sel.r). Used by .github/workflows/sarimax-model-sel.yml,
# but also runnable locally:
#
#   Rscript sarimax/run_states.R
#   STATES=MG,SP Rscript sarimax/run_states.R
#   STATES=all RESET_CHECKPOINT=false NPATHS=300 Rscript sarimax/run_states.R
#
# Env vars (all optional):
#   STATES            comma-separated state codes, or "all" (default "all")
#   RESET_CHECKPOINT  "true"/"false" (default "true"). Only applies when
#                     STATES=all -- clears results/concluded_states.csv first
#                     so every state reruns instead of being skipped as done.
#   NPATHS            bootstrap forecast paths passed to run_model_selection
#                     (default 1000; lower = faster but coarser intervals)
#
# Always updates the real sarimax/results/concluded_states.csv and
# sarimax/results/metrics/*.csv files (unlike sarimax/run_mg_sp_test.R, which
# is a throwaway local test that avoids touching them).
 
all_states <- c("AC", "AL", "AM", "AP", "BA", "CE", "DF", "GO", "MA",
                "MG", "MS", "MT", "PA", "PB", "PE", "PI", "PR", "RJ",
                "RN", "RO", "RR", "RS", "SC", "SE", "SP", "TO")
 
states_env <- Sys.getenv("STATES", "all")
reset_env  <- tolower(Sys.getenv("RESET_CHECKPOINT", "true")) == "true"
npaths_env <- as.numeric(Sys.getenv("NPATHS", "1000"))
 
states_to_run <- if (identical(states_env, "all") || states_env == "") {
  all_states
} else {
  trimws(strsplit(states_env, ",")[[1]])
}
unknown <- setdiff(states_to_run, all_states)
if (length(unknown) > 0) {
  stop("Unknown state code(s): ", paste(unknown, collapse = ", "))
}
 
checkpoint_path <- "sarimax/results/concluded_states.csv"
 
# ---- Load run_model_selection() without executing model_sel.r's own bottom
# ---- loop (which would always process all 26 states and rebuild the summary
# ---- itself -- we replicate that part below so it also works for subsets).
model_sel_lines <- readLines("sarimax/src/model_sel.r")
cutoff <- grep("^states <- c\\(", model_sel_lines)[1]
if (is.na(cutoff)) {
  stop("Could not find the 'states <- c(' marker in sarimax/src/model_sel.r ",
       "-- the file may have changed shape; update this script's cutoff logic.")
}
source(textConnection(model_sel_lines[seq_len(cutoff - 1)]))
 
# Apply the NPATHS override to every run_model_selection() call below.
formals(run_model_selection)$npaths <- npaths_env
 
# ---- Checkpoint handling
if (identical(sort(states_to_run), sort(all_states)) && reset_env) {
  message("Resetting checkpoint: every state will be (re)run.")
  writeLines("state", checkpoint_path)
}
 
for (st in states_to_run) {
  message("==== ", st, " ====")
  concluded_states <- tryCatch(
    readr::read_csv(checkpoint_path, show_col_types = FALSE),
    error = function(e) data.frame(state = character(0), stringsAsFactors = FALSE)
  )
  run_model_selection(state = st, concluded_states = concluded_states)
}
 
# ---- Rebuild the all-states summary from whatever per-state files exist
# ---- (covers both a full run and a partial/subset run).
existing <- Filter(function(st) {
  file.exists(paste0("sarimax/results/metrics/metrics_all_formulas_", st, ".csv"))
}, all_states)
 
best_wis_df <- dplyr::bind_rows(lapply(existing, function(st) {
  d <- readr::read_csv(paste0("sarimax/results/metrics/metrics_all_formulas_", st, ".csv"),
                       show_col_types = FALSE)
  dplyr::mutate(d[1, ], state = st, .before = 1)
}))
readr::write_csv(best_wis_df, "sarimax/results/metrics/best_wis_all_states.csv")
 
message("Done. (Re)ran: ", paste(states_to_run, collapse = ", "))
 
