# ------------------------------------------------------------------------------
# Model assessment: Lasso (nested 5x5 CV), full OLS (5-fold CV),
# and both with squared predictor transformations.
# Target: log(shares). Outputs assessment MSE (in log space) and SD to output/assessment/.
# Run from project root: Rscript scripts/model-assessment.R
# ------------------------------------------------------------------------------

source(file.path("src", "setup.R"))
source(file.path("src", "nested_cv.R"))
source(file.path("src", "data.R"))

set.seed(SEED)
dir.create(DIR_ASSESSMENT, showWarnings = FALSE, recursive = TRUE)

train_data <- load_train_data()

# ------------------------------------------------------------------------------
# Recipes: baseline and with squared predictors
# ------------------------------------------------------------------------------

# Identify numeric predictor names (excluding outcome)
num_predictors <- train_data %>%
  select(-log_shares) %>%
  select(where(is.numeric)) %>%
  names()

# Add squared terms only for count-like predictors (numerically stable).
# Exclude: binary, proportions [0,1], LDA, kw_* (can be huge), self_reference_* (scale issues)
safe_for_sq <- c(
  "timedelta", "n_tokens_title", "n_tokens_content",
  "num_hrefs", "num_self_hrefs", "num_imgs", "num_videos",
  "average_token_length", "num_keywords"
)
predictors_to_sq <- intersect(num_predictors, safe_for_sq)

# Baseline: outcome = log_shares; standardize predictors
rec <- recipe(log_shares ~ ., data = train_data) %>%
  step_normalize(all_numeric_predictors())

# Same predictors plus squared terms for non-binary numeric predictors only
rec_sq <- recipe(log_shares ~ ., data = train_data)
for (nm in predictors_to_sq) {
  rec_sq <- rec_sq %>%
    step_mutate(!!paste0(nm, "_sq") := (!!rlang::sym(nm))^2)
}
rec_sq <- rec_sq %>%
  step_normalize(all_numeric_predictors())

# ------------------------------------------------------------------------------
# Same outer folds for all models; nested CV structure reused
# ------------------------------------------------------------------------------

outer_folds <- vfold_cv(train_data, v = N_FOLDS_OUTER)
nested_splits <- nested_cv(
  train_data,
  outside = outer_folds,
  inside  = vfold_cv(v = N_FOLDS_INNER)
)

# ------------------------------------------------------------------------------
# Lasso (baseline): nested CV (inner for tuning), outer evaluation on outer_folds
# ------------------------------------------------------------------------------

lasso_spec <- linear_reg(penalty = tune(), mixture = 1) %>%
  set_engine("glmnet")
penalty_grid <- grid_regular(penalty(range = PENALTY_RANGE), levels = PENALTY_LEVELS)

message("Running nested CV for Lasso (5 outer x 5 inner)...")
best_penalty_lasso <- map_dbl(
  seq_len(nrow(nested_splits)),
  function(i) {
    tune_inner(
      nested_splits$splits[[i]],
      nested_splits$inner_resamples[[i]],
      lasso_spec, rec, penalty_grid
    )
  }
)
nested_splits_lasso <- nested_splits %>%
  mutate(best_penalty = best_penalty_lasso)
outer_metrics_lasso <- map2_dfr(
  nested_splits_lasso$splits,
  nested_splits_lasso$best_penalty,
  ~ eval_outer(.x, .y, lasso_spec, rec, outcome = "log_shares")
)

assessment_mse_lasso    <- mean(outer_metrics_lasso$mse)
assessment_mse_sd_lasso <- sd(outer_metrics_lasso$mse)
assessment_mae_lasso     <- mean(outer_metrics_lasso$mae)
assessment_mae_sd_lasso  <- sd(outer_metrics_lasso$mae)

# ------------------------------------------------------------------------------
# Lasso with squared predictors: nested CV, same outer folds
# ------------------------------------------------------------------------------

message("Running nested CV for Lasso with squared predictors (5 outer x 5 inner)...")
best_penalty_lasso_sq <- map_dbl(
  seq_len(nrow(nested_splits)),
  function(i) {
    tune_inner(
      nested_splits$splits[[i]],
      nested_splits$inner_resamples[[i]],
      lasso_spec, rec_sq, penalty_grid
    )
  }
)
nested_splits_lasso_sq <- nested_splits %>%
  mutate(best_penalty = best_penalty_lasso_sq)
outer_metrics_lasso_sq <- map2_dfr(
  nested_splits_lasso_sq$splits,
  nested_splits_lasso_sq$best_penalty,
  ~ eval_outer(.x, .y, lasso_spec, rec_sq, outcome = "log_shares")
)

assessment_mse_lasso_sq    <- mean(outer_metrics_lasso_sq$mse)
assessment_mse_sd_lasso_sq <- sd(outer_metrics_lasso_sq$mse)
assessment_mae_lasso_sq     <- mean(outer_metrics_lasso_sq$mae)
assessment_mae_sd_lasso_sq  <- sd(outer_metrics_lasso_sq$mae)

# ------------------------------------------------------------------------------
# Full OLS (all regressors): 5-fold CV on outer_folds, no tuning
# ------------------------------------------------------------------------------

message("Running 5-fold CV for full OLS model (same folds)...")
wf_full <- workflow() %>%
  add_recipe(rec) %>%
  add_model(linear_reg() %>% set_engine("lm"))
cv_full <- fit_resamples(
  wf_full,
  resamples = outer_folds,
  metrics   = metric_set(rmse, mae)
)
metrics_full <- collect_metrics(cv_full, summarize = FALSE)
metrics_full_rmse <- metrics_full %>% filter(.metric == "rmse") %>% pull(.estimate)
assessment_mse_full    <- mean(metrics_full_rmse^2)
assessment_mse_sd_full <- sd(metrics_full_rmse^2)
assessment_mae_full     <- mean(metrics_full %>% filter(.metric == "mae") %>% pull(.estimate))
assessment_mae_sd_full  <- sd(metrics_full %>% filter(.metric == "mae") %>% pull(.estimate))

# Full OLS with squared predictors (Ridge penalty for stability; whitelist of count vars)
message("Running 5-fold CV for full OLS + squared model (Ridge, same folds)...")
ridge_sq_spec <- linear_reg(penalty = 0.01, mixture = 0) %>% set_engine("glmnet")
wf_full_sq <- workflow() %>%
  add_recipe(rec_sq) %>%
  add_model(ridge_sq_spec)
cv_full_sq <- fit_resamples(
  wf_full_sq,
  resamples = outer_folds,
  metrics   = metric_set(rmse, mae)
)
metrics_full_sq <- collect_metrics(cv_full_sq, summarize = FALSE)
metrics_full_sq_rmse <- metrics_full_sq %>% filter(.metric == "rmse") %>% pull(.estimate)
assessment_mse_full_sq    <- mean(metrics_full_sq_rmse^2)
assessment_mse_sd_full_sq <- sd(metrics_full_sq_rmse^2)
assessment_mae_full_sq     <- mean(metrics_full_sq %>% filter(.metric == "mae") %>% pull(.estimate))
assessment_mae_sd_full_sq  <- sd(metrics_full_sq %>% filter(.metric == "mae") %>% pull(.estimate))

# ------------------------------------------------------------------------------
# Report and write to output/assessment/
# ------------------------------------------------------------------------------

message("\n========== Assessment MSE (5-fold or nested CV) ==========")
message("Lasso (nested 5x5): ", round(assessment_mse_lasso, 4), " (SD = ", round(assessment_mse_sd_lasso, 4), ")")
message("Lasso + squared predictors (nested 5x5): ", round(assessment_mse_lasso_sq, 4), " (SD = ", round(assessment_mse_sd_lasso_sq, 4), ")")
message("Full OLS (5-fold): ", round(assessment_mse_full, 4), " (SD = ", round(assessment_mse_sd_full, 4), ")")
message("Full OLS + squared predictors (5-fold): ", round(assessment_mse_full_sq, 4), " (SD = ", round(assessment_mse_sd_full_sq, 4), ")")
message("\n========== Assessment MAE (5-fold or nested CV) ==========")
message("Lasso (nested 5x5): ", round(assessment_mae_lasso, 4), " (SD = ", round(assessment_mae_sd_lasso, 4), ")")
message("Lasso + squared predictors (nested 5x5): ", round(assessment_mae_lasso_sq, 4), " (SD = ", round(assessment_mae_sd_lasso_sq, 4), ")")
message("Full OLS (5-fold): ", round(assessment_mae_full, 4), " (SD = ", round(assessment_mae_sd_full, 4), ")")
message("Full OLS + squared predictors (5-fold): ", round(assessment_mae_full_sq, 4), " (SD = ", round(assessment_mae_sd_full_sq, 4), ")")

out_assessment <- tibble(
  model             = c("lasso", "lasso_sq", "full", "full_sq"),
  assessment_mse    = c(assessment_mse_lasso, assessment_mse_lasso_sq, assessment_mse_full, assessment_mse_full_sq),
  assessment_mse_sd = c(assessment_mse_sd_lasso, assessment_mse_sd_lasso_sq, assessment_mse_sd_full, assessment_mse_sd_full_sq),
  assessment_mae    = c(assessment_mae_lasso, assessment_mae_lasso_sq, assessment_mae_full, assessment_mae_full_sq),
  assessment_mae_sd = c(assessment_mae_sd_lasso, assessment_mae_sd_lasso_sq, assessment_mae_sd_full, assessment_mae_sd_full_sq)
)
write_csv(out_assessment, file.path(DIR_ASSESSMENT, "assessment_mse.csv"))
message("\nOutput written to ", file.path(DIR_ASSESSMENT, "assessment_mse.csv"))

# Identify best model by lowest Assessment MSE and record it for downstream use
best_idx <- which.min(out_assessment$assessment_mse)
best_model <- out_assessment$model[best_idx]
best_mse  <- out_assessment$assessment_mse[best_idx]
message("Best model by Assessment MSE: ", best_model,
        " (MSE = ", round(best_mse, 4), ")")
dir.create(DIR_FITS, showWarnings = FALSE, recursive = TRUE)
writeLines(best_model, file.path(DIR_FITS, "best_model.txt"))

message("Done.")
