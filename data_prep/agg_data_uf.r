library(tidyverse)

dengue_merged <- read_csv("processed_data/dengue/dengue_merged.csv.gz")
chikungunya_merged <- read_csv("processed_data/chikungunya/chikungunya_merged.csv.gz")

mode_stat <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

dengue_uf <- dengue_merged %>%
  group_by(uf, uf_code, epiweek) %>%
  summarise(
    cases = sum(casos, na.rm = TRUE),
    temp_min_mean = mean(temp_min, na.rm = TRUE),
    temp_med_mean = mean(temp_med, na.rm = TRUE),
    temp_max_mean = mean(temp_max, na.rm = TRUE),
    precip_min_mean = mean(precip_min, na.rm = TRUE),
    precip_med_mean = mean(precip_med, na.rm = TRUE),
    precip_max_mean = mean(precip_max, na.rm = TRUE),
    pressure_min_mean = mean(pressure_min, na.rm = TRUE),
    pressure_med_mean = mean(pressure_med, na.rm = TRUE),
    pressure_max_mean = mean(pressure_max, na.rm = TRUE),
    rel_humid_min_mean = mean(rel_humid_min, na.rm = TRUE),
    rel_humid_med_mean = mean(rel_humid_med, na.rm = TRUE),
    rel_humid_max_mean = mean(rel_humid_max, na.rm = TRUE),
    thermal_range_mean = mean(thermal_range, na.rm = TRUE),
    rainy_days_mean = mean(rainy_days, na.rm = TRUE),
    koppen = mode_stat(koppen),
    biome = mode_stat(biome),
    enso = first(enso),
    iod = first(iod),
    pdo = first(pdo),
    pop = sum(population, na.rm = TRUE),
    year = first(year),
    train_1 = first(train_1),
    train_2 = first(train_2),
    train_3 = first(train_3),
    train_4 = first(train_4),
    target_1 = first(target_1),
    target_2 = first(target_2),
    target_3 = first(target_3),
    target_4 = first(target_4)
  ) %>%
  ungroup()

chikungunya_uf <- chikungunya_merged %>%
  group_by(uf, uf_code, epiweek) %>%
  summarise(
    cases = sum(casos, na.rm = TRUE),
    temp_min_mean = mean(temp_min, na.rm = TRUE),
    temp_med_mean = mean(temp_med, na.rm = TRUE),
    temp_max_mean = mean(temp_max, na.rm = TRUE),
    precip_min_mean = mean(precip_min, na.rm = TRUE),
    precip_med_mean = mean(precip_med, na.rm = TRUE),
    precip_max_mean = mean(precip_max, na.rm = TRUE),
    pressure_min_mean = mean(pressure_min, na.rm = TRUE),
    pressure_med_mean = mean(pressure_med, na.rm = TRUE),
    pressure_max_mean = mean(pressure_max, na.rm = TRUE),
    rel_humid_min_mean = mean(rel_humid_min, na.rm = TRUE),
    rel_humid_med_mean = mean(rel_humid_med, na.rm = TRUE),
    rel_humid_max_mean = mean(rel_humid_max, na.rm = TRUE),
    thermal_range_mean = mean(thermal_range, na.rm = TRUE),
    rainy_days_mean = mean(rainy_days, na.rm = TRUE),
    koppen = mode_stat(koppen),
    biome = mode_stat(biome),
    enso = first(enso),
    iod = first(iod),
    pdo = first(pdo),
    pop = sum(population, na.rm = TRUE),
    year = first(year),
    train_1 = first(train_1),
    train_2 = first(train_2),
    train_3 = first(train_3),
    train_4 = first(train_4),
    target_1 = first(target_1),
    target_2 = first(target_2),
    target_3 = first(target_3),
    target_4 = first(target_4)
  ) %>%
  ungroup()

# count NAs by column
na_counts_dengue <- sapply(dengue_uf, function(x) sum(is.na(x)))
na_counts_chikungunya <- sapply(chikungunya_uf, function(x) sum(is.na(x)))

lapply(unique(dengue_uf$uf), function(uf) {
  if (uf == "ES") {
    return()  # Skip ES
  }
  dengue_uf %>%
    filter(uf == !!uf) %>%
    write_csv(paste0("processed_data/dengue/dengue_", uf, "_agg.csv.gz"))
  chikungunya_uf %>%
    filter(uf == !!uf) %>%
    write_csv(paste0("processed_data/chikungunya/chikungunya_", uf, "_agg.csv.gz"))
})

