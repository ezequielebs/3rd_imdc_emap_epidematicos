#' Submit city-level forecasts to the Mosqlimate platform
#'
#' Reads the best forecasts produced by `fit.r` (one CSV per city x target
#' window under `sarimax/results/preds/`) and uploads them to the
#' Mosqlimate Predictions Registry through the `mosqlient` Python package
#' (accessed from R via `reticulate`).
#'
#' Credentials: the Mosqlimate API key is NEVER hardcoded here. It is read
#' from the `MOSQLIMATE_API_KEY` environment variable, which should be
#' defined in a local, git-ignored `.env` file as:
#'   MOSQLIMATE_API_KEY=your-key-here
#' Get your key from the "Auth" section of your Mosqlimate profile
#' (https://mosqlimate.org/). Never commit a real key to a public repository.
#'
#' NOTE: an earlier version of this script had a real API key hardcoded in
#' plain text. If that key was ever committed or shared, rotate/regenerate
#' it on mosqlimate.org as soon as possible.

library(reticulate)
library(data.table)
library(dotenv)
library(readr)

py_config()
py_require(c("epiweeks", "python-dotenv", "mosqlient"))

# Load MOSQLIMATE_API_KEY from a local .env file (ignored by git, see .gitignore).
# Falls back to whatever is already in the environment if no .env is present.
if (file.exists(".env")) dotenv::load_dot_env(".env")

#' Submit a batch of city-level forecasts for a single disease to Mosqlimate
#'
#' @param disease Character. ICD-10 code submitted to Mosqlimate
#'   (e.g. "A90" for dengue, "A92.0" for chikungunya).
#' @param disease_label Character. Lowercase label used in file paths
#'   (e.g. "dengue", "chikungunya"). Defaults to `disease` if not given.
#' @param commit Character. Commit hash of the model version used to
#'   generate these predictions.
#' @param repository Character. GitHub repository in "owner/repo" form.
#' @param cities Numeric/character vector of IBGE city geocodes to submit.
#' @param n_targets Integer. Number of target windows per city
#'   (target_1..target_n). Defaults to 4.
#' @param processed_data_dir Character. Base directory containing the
#'   aggregated case-count CSVs, expected at
#'   `<processed_data_dir>/<disease_label>/sel_cities/<disease_label>_<city>_agg.csv.gz`.
#' @param preds_dir Character. Directory containing prediction CSVs produced
#'   by `fit.r`, expected at
#'   `<preds_dir>/pred_<disease_label>_<city>_target_<i>.csv`.
#' @param case_definition Character. Case definition reported to Mosqlimate
#'   (e.g. "probable").
#' @param adm_level Integer. Administrative level of the prediction
#'   (2 = city/municipality).
#' @param published Logical. Whether the prediction should be published.
#' @param api_key Character. Mosqlimate API key. Defaults to the
#'   `MOSQLIMATE_API_KEY` environment variable.
#' @param on_duplicate Character. What to do when Mosqlimate reports that a
#'   prediction already exists for this model/commit/city/target:
#'   "skip" (default, log and move on), "error" (stop the whole run), or
#'   "overwrite" (delete the existing prediction via `mosq$remove_prediction()`
#'   if available, then re-upload). "overwrite" requires the mosqlient
#'   version installed to expose a removal function; if it doesn't, this
#'   falls back to "skip" with a warning.
#' @param verbose Logical. Print progress messages. Defaults to TRUE.
#'
#' @return Invisibly, a data.table logging each city/target pair and its
#'   status ("submitted", "duplicate_skipped", "duplicate_overwritten",
#'   "missing_case_file", "missing_pred_file", or "error").
submit_mosqlimate_city_forecasts <- function(disease,
                                              commit,
                                              repository,
                                              cities,
                                              disease_label = disease,
                                              n_targets = 4,
                                              processed_data_dir = "processed_data",
                                              preds_dir = "sarimax/results/preds",
                                              case_definition = "probable",
                                              adm_level = 2,
                                              published = TRUE,
                                              api_key = Sys.getenv("MOSQLIMATE_API_KEY"),
                                              on_duplicate = c("skip", "error", "overwrite"),
                                              verbose = TRUE) {

  on_duplicate <- match.arg(on_duplicate)

  if (identical(api_key, "")) {
    stop(
      "MOSQLIMATE_API_KEY is not set. Create a .env file in the repository ",
      "root with a line `MOSQLIMATE_API_KEY=your-key-here` (see sarimax/README.md), ",
      "or pass `api_key` explicitly."
    )
  }

  mosq <- import("mosqlient")
  target_ids <- paste0("target_", seq_len(n_targets))

  log_rows <- list()

  for (city in cities) {

    case_file <- file.path(
      processed_data_dir, disease_label, "sel_cities",
      paste0(disease_label, "_", city, "_agg.csv.gz")
    )
    if (!file.exists(case_file)) {
      warning("Skipping ", city, ": case-count file not found at ", case_file)
      log_rows[[length(log_rows) + 1]] <- data.table(
        disease = disease_label, city = city, target = NA_character_,
        commit = commit, status = "missing_case_file"
      )
      next
    }
    d <- read_csv(case_file, show_col_types = FALSE)
    adm_1 <- d$uf_code[1]
    adm_2 <- d$geocode[1]

    for (target_id in target_ids) {
      pred_file <- file.path(
        preds_dir,
        paste0("pred_", disease_label, "_", city, "_", target_id, ".csv")
      )
      if (!file.exists(pred_file)) {
        warning("Skipping ", city, " ", target_id, ": prediction file not found at ", pred_file)
        log_rows[[length(log_rows) + 1]] <- data.table(
          disease = disease_label, city = city, target = target_id,
          commit = commit, status = "missing_pred_file"
        )
        next
      }
      pred <- read_csv(pred_file, show_col_types = FALSE)
      description <- paste0(
        tools::toTitleCase(disease_label), " prediction for city ", city,
        " and target ", target_id
      )

      if (verbose) message("Submitting ", disease_label, " | ", city, " | ", target_id)

      do_upload <- function() {
        mosq$upload_prediction(
          api_key = api_key,
          repository = repository,
          description = description,
          commit = commit,
          disease = disease,
          case_definition = case_definition,
          adm_level = adm_level,
          adm_1 = adm_1,
          adm_2 = adm_2,
          published = published,
          prediction = pred
        )
      }

      status <- tryCatch({
        do_upload()
        "submitted"
      }, error = function(e) {
        msg <- conditionMessage(e)
        is_duplicate <- grepl("Duplication found", msg, fixed = TRUE)

        if (!is_duplicate) {
          if (verbose) message("  -> error: ", msg)
          return("error")
        }

        if (on_duplicate == "error") {
          stop(e)
        } else if (on_duplicate == "skip") {
          if (verbose) message("  -> duplicate, skipping")
          return("duplicate_skipped")
        } else { # overwrite
          if (verbose) message("  -> duplicate, attempting overwrite")
          removed <- tryCatch({
            if ("remove_prediction" %in% names(mosq)) {
              mosq$remove_prediction(
                api_key = api_key,
                repository = repository,
                commit = commit,
                disease = disease,
                adm_level = adm_level,
                adm_1 = adm_1,
                adm_2 = adm_2
              )
              TRUE
            } else {
              FALSE
            }
          }, error = function(e2) {
            if (verbose) message("  -> removal failed: ", conditionMessage(e2))
            FALSE
          })

          if (!removed) {
            warning(
              "Could not remove existing prediction for ", city, " ", target_id,
              " (mosqlient has no removal function, or removal failed). ",
              "Delete it manually on mosqlimate.org if you need to overwrite. Skipping."
            )
            return("duplicate_skipped")
          }

          re_status <- tryCatch({
            do_upload()
            "duplicate_overwritten"
          }, error = function(e3) {
            if (verbose) message("  -> re-upload failed: ", conditionMessage(e3))
            "error"
          })
          return(re_status)
        }
      })

      log_rows[[length(log_rows) + 1]] <- data.table(
        disease = disease_label, city = city, target = target_id,
        commit = commit, status = status
      )
    }
  }

  result <- rbindlist(log_rows, fill = TRUE)
  if (verbose && nrow(result) > 0) {
    message("\nSummary:")
    print(result[, .N, by = status])
  }

  invisible(result)
}

# ── Example usage ─────────────────────────────────────────────────────────
# Fill in the commit hashes, repository name, and city lists for the run
# being submitted, then call the function once per disease.

repository <- "EzequielEBS/3rd_imdc_emap_epidematicos_sarimax_muni"

cities_dengue <- c(
  2931350, 2933307, 2302503, 3119401, 3549805,
  3541406, 1200401, 1200203, 1716109, 4113700,
  4103701, 4104808, 5201405, 5102637, 5215231
)

# Dengue
submit_mosqlimate_city_forecasts(
  disease       = "A90",
  disease_label = "dengue",
  commit        = "8c048fe65ff3db5ab02e79cb894f27fcded3e64c",
  repository    = repository,
  cities        = cities_dengue,
  on_duplicate  = "skip"
)

cities_chikungunya <- c(
  2211001, 2931350, 3143302, 3119401, 1721000,
  1716109, 4104808, 4219507, 5103403, 5102637
)

# Chikungunya
submit_mosqlimate_city_forecasts(
  disease       = "A92.0",
  disease_label = "chikungunya",
  commit        = "52f253dbd7380af22ba4cb63d839ea9dbad73452",
  repository    = repository,
  cities        = cities_chikungunya,
  on_duplicate  = "skip"
)
