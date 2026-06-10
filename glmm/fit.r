library(INLA)
library(tidyverse)
library(geobr)

states <- read_state(year = 2010) %>%
  select(abbrev_state) %>%
  filter(abbrev_state != "ES") %>%
  pull(abbrev_state)

dengue_merged <- read_csv("processed_data/dengue/dengue_merged.csv.gz")
dengue_uf <- lapply(states, function(state){
  file_name <- paste0("processed_data/dengue/dengue_", state, "_agg.csv.gz")
  read_csv(file_name, show_col_types = FALSE)
})


data <- dengue_uf$SP %>%
  filter(train_3 | target_3)
data[data$target_3, "cases"] <- NA
data <- data %>%
  mutate(
    time_id = row_number(),
    week_id = as.integer(substr(epiweek, 5, 6)),
    year_id = as.numeric(as.factor(year)),
    year_id2 = as.numeric(as.factor(year))
  )

formula <- 
  cases ~ enso + 
          temp_max_mean_mean_9mo +
          f(week_id, model = "rw2", cyclic = TRUE,
            constr = TRUE, group = year_id, scale.model = TRUE) +
          f(year_id, model = "iid") +
          offset(log(pop))
family <- "nbinomial"
quantiles <- c(0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975)
model <- inla(
  formula,
  data = data,
  family = family,
  control.predictor = list(compute = TRUE,
                           link = 1,
                           quantiles = quantiles
                           ),
  control.compute = list(dic = TRUE, waic = TRUE)
)

summary(model)

sum((model$summary.fitted.values[data$target_3,]$mean - 
  dengue_uf$SP$cases[data$target_3])^2)
