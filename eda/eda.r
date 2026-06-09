library(tidyverse)
library(ggplot2)
library(geobr)

states <- read_state(year = 2010) %>%
  select(abbrev_state) %>%
  filter(abbrev_state != "ES") %>%
  pull(abbrev_state)

dengue_merged <- read_csv("processed_data/dengue/dengue_merged.csv.gz")

time_series_uf <- 
  ggplot(
    dengue_merged %>%
      group_by(uf, epiweek, date) %>%
      summarise(cases = sum(casos, na.rm = TRUE)) %>%
      ungroup(),
    aes(x = as.Date(date), y = cases)
  ) +
  geom_line() +
  facet_wrap(~ uf, scales = "free_y") +
  labs(title = "",
       x = "Date",
       y = "Count") +
  theme_bw()
time_series_uf

dengue_uf <- lapply(states, function(state){
  file_name <- paste0("processed_data/dengue/dengue_", state, "_agg.csv.gz")
  read_csv(file_name, show_col_types = FALSE)
})

names(dengue_uf) <- states
names(dengue_uf$RO)

corr_plots_starting_cov <- lapply(dengue_uf, function(df){
  df <- df %>%
    select(cases, temp_min_mean, temp_med_mean, temp_max_mean,
           precip_min_mean, precip_med_mean, precip_max_mean,
           pressure_min_mean, pressure_med_mean, pressure_max_mean,
           rel_humid_min_mean, rel_humid_med_mean, rel_humid_max_mean,
           thermal_range_mean, rainy_days_mean, koppen, biome,
           enso, iod, pdo)
  
  df$koppen <- as.numeric(as.factor(df$koppen))
  df$biome <- as.numeric(as.factor(df$biome))

  df %>%
    cor(use = "complete.obs") %>%
    as.data.frame() %>%
    rownames_to_column(var = "variable") %>%
    pivot_longer(-variable, names_to = "variable2", values_to = "correlation") %>%
    ggplot(aes(x = variable, y = variable2, fill = correlation)) +
    geom_tile() +
    scale_fill_gradient2(low = "blue", mid = "white", high = "red", midpoint = 0) +
    labs(title = paste("Correlation Matrix -", unique(df$uf)),
          x = "Variable",
          y = "Variable") +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
})
names(corr_plots_starting_cov) <- states

corr_plots_lagged_cov <- lapply(dengue_uf, function(df){
  df <- df %>%
    select(cases, temp_min_mean_lag4, temp_min_mean_lag8, temp_min_mean_lag12,
           temp_med_mean_lag4, temp_med_mean_lag8, temp_med_mean_lag12,
           temp_max_mean_lag4, temp_max_mean_lag8, temp_max_mean_lag12,
           precip_min_mean_lag4, precip_min_mean_lag8, precip_min_mean_lag12,
           precip_med_mean_lag4, precip_med_mean_lag8, precip_med_mean_lag12,
           precip_max_mean_lag4, precip_max_mean_lag8, precip_max_mean_lag12,
           pressure_min_mean_lag4, pressure_min_mean_lag8, pressure_min_mean_lag12,
           pressure_med_mean_lag4, pressure_med_mean_lag8, pressure_med_mean_lag12,
           pressure_max_mean_lag4, pressure_max_mean_lag8, pressure_max_mean_lag12,
           rel_humid_min_mean_lag4, rel_humid_min_mean_lag8, rel_humid_min_mean_lag12,
           rel_humid_med_mean_lag4, rel_humid_med_mean_lag8, rel_humid_med_mean_lag12,
           rel_humid_max_mean_lag4, rel_humid_max_mean_lag8, rel_humid_max_mean_lag12,
           thermal_range_mean_lag4, thermal_range_mean_lag8, thermal_range_mean_lag12,
           rainy_days_mean_lag4, rainy_days_mean_lag8, rainy_days_mean_lag12,
           koppen, biome, enso, iod, pdo)
  
  df$koppen <- as.numeric(as.factor(df$koppen))
  df$biome <- as.numeric(as.factor(df$biome))

  df %>%
    cor(use = "complete.obs") %>%
    as.data.frame() %>%
    rownames_to_column(var = "variable") %>%
    pivot_longer(-variable, names_to = "variable2", values_to = "correlation") %>%
    ggplot(aes(x = variable, y = variable2, fill = correlation)) +
    geom_tile() +
    scale_fill_gradient2(low = "blue", mid = "white", high = "red", midpoint = 0) +
    labs(title = paste("Correlation Matrix -", unique(df$uf)),
         x = "Variable",
         y = "Variable") +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
})
names(corr_plots_lagged_cov) <- states

corr_plots_summ_cov <- lapply(dengue_uf, function(df){
  df <- df %>%
    select(cases, temp_min_mean_mean_3mo, temp_min_mean_mean_6mo, temp_min_mean_mean_9mo,
           temp_med_mean_mean_3mo, temp_med_mean_mean_6mo, temp_med_mean_mean_9mo,
           temp_max_mean_mean_3mo, temp_max_mean_mean_6mo, temp_max_mean_mean_9mo,
           precip_min_mean_mean_3mo, precip_min_mean_mean_6mo, precip_min_mean_mean_9mo,
           precip_med_mean_mean_3mo, precip_med_mean_mean_6mo, precip_med_mean_mean_9mo,
           precip_max_mean_mean_3mo, precip_max_mean_mean_6mo, precip_max_mean_mean_9mo,
           pressure_min_mean_mean_3mo, pressure_min_mean_mean_6mo, pressure_min_mean_mean_9mo,
           pressure_med_mean_mean_3mo, pressure_med_mean_mean_6mo, pressure_med_mean_mean_9mo,
           pressure_max_mean_mean_3mo, pressure_max_mean_mean_6mo, pressure_max_mean_mean_9mo,
           rel_humid_min_mean_mean_3mo, rel_humid_min_mean_mean_6mo, rel_humid_min_mean_mean_9mo,
           rel_humid_med_mean_mean_3mo, rel_humid_med_mean_mean_6mo, rel_humid_med_mean_mean_9mo,
           rel_humid_max_mean_mean_3mo, rel_humid_max_mean_mean_6mo, rel_humid_max_mean_mean_9mo,
           thermal_range_mean_mean_3mo, thermal_range_mean_mean_6mo, thermal_range_mean_mean_9mo,
           rainy_days_mean_mean_3mo, rainy_days_mean_mean_6mo, rainy_days_mean_mean_9mo,
           koppen, biome, enso, iod, pdo)
  
  df$koppen <- as.numeric(as.factor(df$koppen))
  df$biome <- as.numeric(as.factor(df$biome))

  df %>%
    cor(use = "complete.obs") %>%
    as.data.frame() %>%
    rownames_to_column(var = "variable") %>%
    pivot_longer(-variable, names_to = "variable2", values_to = "correlation") %>%
    ggplot(aes(x = variable, y = variable2, fill = correlation)) +
    geom_tile() +
    scale_fill_gradient2(low = "blue", mid = "white", high = "red", midpoint = 0) +
    labs(title = paste("Correlation Matrix -", unique(df$uf)),
         x = "Variable",
         y = "Variable") +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
})
names(corr_plots_summ_cov) <- states

