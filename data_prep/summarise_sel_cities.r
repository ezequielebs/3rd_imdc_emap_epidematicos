library(tidyverse)
library(runner)

# IBGE geocodes of state-capital/large cities tracked for dengue diagnostics.
cities_dengue <- c(
  2931350,
  2933307,
  2302503,
  3119401,
  3549805,
  3541406,
  1200401,
  1200203,
  1716109,
  4113700,
  4103701,
  4104808,
  5201405,
  5102637,
  5215231
)

# IBGE geocodes of state-capital/large cities tracked for chikungunya diagnostics.
cities_chikungunya <- c(
  2211001,
  2931350,
  3143302,
  3119401,
  1721000,
  1716109,
  4104808,
  4219507,
  5103403,
  5102637
)

for (city in cities_dengue) {
  data <- read_csv(paste0("processed_data/dengue/sel_cities/dengue_", city, ".csv.gz"))

  data <- data %>%
    mutate(
      cases = casos,
      pop = population
    )
  # create lagged covariates for dengue
  data <- data %>%
    arrange(epiweek) %>%
    mutate(
      across(
        c(temp_min, temp_med, temp_max, precip_min, precip_med, precip_max,
          pressure_min, pressure_med, pressure_max, rel_humid_min,
          rel_humid_med, rel_humid_max, thermal_range, rainy_days),
        list(
          lag4 = ~ lag(., 4, default = first(.)),
          lag8 = ~ lag(., 8, default = first(.)),
          lag12 = ~ lag(., 12, default = first(.)),
          lag16 = ~ lag(., 16, default = first(.))
        ),
        .names = "{col}_{fn}"
      )
    )
  
  # create summary covariates for dengue
  data <- data %>%
    arrange(epiweek) %>%
    mutate(
      across(
        c(temp_min, temp_med, temp_max, precip_min, precip_med, precip_max,
          pressure_min, pressure_med, pressure_max, rel_humid_min,
          rel_humid_med, rel_humid_max, thermal_range, rainy_days),
        list(
          mean_3mo = ~ dplyr::lag(mean_run(., k = 12, na_rm = TRUE), default = first(.)),
          mean_6mo = ~ dplyr::lag(mean_run(., k = 24, na_rm = TRUE), default = first(.)),
          mean_9mo = ~ dplyr::lag(mean_run(., k = 36, na_rm = TRUE), default = first(.)),
          mean_12mo = ~ dplyr::lag(mean_run(., k = 48, na_rm = TRUE), default = first(.))
        ),
        .names = "{col}_{fn}"
      )
    )
  write_csv(data, paste0("processed_data/dengue/sel_cities/dengue_", city, "_agg.csv.gz"))
}

for (city in cities_chikungunya) {
  data <- read_csv(paste0("processed_data/chikungunya/sel_cities/chikungunya_", city, ".csv.gz"))

  data <- data %>%
    mutate(
      cases = casos,
      pop = population
    )
  # create lagged covariates for chikungunya
  data <- data %>%
    arrange(epiweek) %>%
    mutate(
      across(
        c(temp_min, temp_med, temp_max, precip_min, precip_med, precip_max,
          pressure_min, pressure_med, pressure_max, rel_humid_min,
          rel_humid_med, rel_humid_max, thermal_range, rainy_days),
        list(
          lag4 = ~ lag(., 4, default = first(.)),
          lag8 = ~ lag(., 8, default = first(.)),
          lag12 = ~ lag(., 12, default = first(.)),
          lag16 = ~ lag(., 16, default = first(.))
        ),
        .names = "{col}_{fn}"
      )
    )
  
  # create summary covariates for chikungunya
  data <- data %>%
    arrange(epiweek) %>%
    mutate(
      across(
        c(temp_min, temp_med, temp_max, precip_min, precip_med, precip_max,
          pressure_min, pressure_med, pressure_max, rel_humid_min,
          rel_humid_med, rel_humid_max, thermal_range, rainy_days),
        list(
          mean_3mo = ~ dplyr::lag(mean_run(., k = 12, na_rm = TRUE), default = first(.)),
          mean_6mo = ~ dplyr::lag(mean_run(., k = 24, na_rm = TRUE), default = first(.)),
          mean_9mo = ~ dplyr::lag(mean_run(., k = 36, na_rm = TRUE), default = first(.)),
          mean_12mo = ~ dplyr::lag(mean_run(., k = 48, na_rm = TRUE), default = first(.))
        ),
        .names = "{col}_{fn}"
      )
    )
  write_csv(data, paste0("processed_data/chikungunya/sel_cities/chikungunya_", city, "_agg.csv.gz"))
}


