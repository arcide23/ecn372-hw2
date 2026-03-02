# ------------------------------------------------------------------------------
# Evaluate final model on held-out test set.
# - Loads trained final model from output/fits/fit_final.rds
# - Reads test data from data/raw/test.csv (PATH_TEST)
# - Prints only the test MSE (on log_shares) to stdout, nothing else.
# Run from project root: Rscript scripts/model-evaluate.R
# ------------------------------------------------------------------------------

# Suppress package/runtime warnings so only the MSE line is printed
options(warn = -1)

suppressPackageStartupMessages({
  source(file.path("src", "config.R"))
  library(tidyverse)
  library(tidymodels)
  library(glmnet)
  source(file.path("src", "data.R"))
})

set.seed(SEED)

final_model_path <- file.path(DIR_FITS, "fit_final.rds")
if (!file.exists(final_model_path)) {
  stop(
    "Trained final model not found at ", final_model_path,
    ". Run 'make model' first to train it.",
    call. = FALSE
  )
}

final_model <- readRDS(final_model_path)

if (!file.exists(PATH_TEST)) {
  stop(
    "Test data file not found at ", PATH_TEST,
    ". Please provide data/raw/test.csv.",
    call. = FALSE
  )
}

# Prepare test data in the same way as training data (log_shares, drop url)
test_raw <- read_csv(PATH_TEST, show_col_types = FALSE)
test_data <- test_raw %>%
  select(-url) %>%
  mutate(log_shares = log(shares))

pred <- predict(final_model, new_data = test_data)
mse <- mean((test_data$log_shares - pred$.pred)^2)

cat("MSE:", mse, "\n")

