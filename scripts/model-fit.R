# ------------------------------------------------------------------------------
# Fit Lasso, full OLS, and their versions with squared predictors to train data.
# Saves workflows and coefficient estimates (output/fits/coefficients.csv) to output/fits/.
# Run from project root: Rscript scripts/model-fit.R
# ------------------------------------------------------------------------------

source(file.path("src", "setup.R"))
source(file.path("src", "data.R"))

set.seed(SEED)
dir.create(DIR_FITS, showWarnings = FALSE, recursive = TRUE)

train_data <- load_train_data()

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

# Recipes: outcome = log_shares; standardize predictors for lasso/full
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
# Lasso (baseline): 5-fold CV for hyperparameter tuning
# ------------------------------------------------------------------------------
lasso_spec <- linear_reg(penalty = tune(), mixture = 1) %>%
  set_engine("glmnet")
penalty_grid <- grid_regular(penalty(range = PENALTY_RANGE), levels = PENALTY_LEVELS)
train_cv <- vfold_cv(train_data, v = N_FOLDS_OUTER)

message("Tuning and fitting Lasso (5-fold CV)...")
wf_lasso <- workflow() %>%
  add_recipe(rec) %>%
  add_model(lasso_spec)
tune_lasso <- tune_grid(
  wf_lasso,
  resamples = train_cv,
  grid = penalty_grid,
  metrics = metric_set(rmse)
)
best_lasso <- select_best(tune_lasso, metric = "rmse")
final_lasso <- finalize_workflow(wf_lasso, best_lasso) %>%
  fit(train_data)

# ------------------------------------------------------------------------------
# Lasso with squared predictors: 5-fold CV for hyperparameter tuning
# ------------------------------------------------------------------------------

message("Tuning and fitting Lasso with squared predictors (5-fold CV)...")
wf_lasso_sq <- workflow() %>%
  add_recipe(rec_sq) %>%
  add_model(lasso_spec)
tune_lasso_sq <- tune_grid(
  wf_lasso_sq,
  resamples = train_cv,
  grid = penalty_grid,
  metrics = metric_set(rmse)
)
best_lasso_sq <- select_best(tune_lasso_sq, metric = "rmse")
final_lasso_sq <- finalize_workflow(wf_lasso_sq, best_lasso_sq) %>%
  fit(train_data)

# ------------------------------------------------------------------------------
# Full OLS: fit on full train (no tuning)
# ------------------------------------------------------------------------------
message("Fitting full OLS model...")
wf_full <- workflow() %>%
  add_recipe(rec) %>%
  add_model(linear_reg() %>% set_engine("lm"))
final_full <- fit(wf_full, train_data)

message("Fitting full OLS + squared model (Ridge for stability)...")
ridge_sq_spec <- linear_reg(penalty = 0.01, mixture = 0) %>% set_engine("glmnet")
wf_full_sq <- workflow() %>%
  add_recipe(rec_sq) %>%
  add_model(ridge_sq_spec)
final_full_sq <- fit(wf_full_sq, train_data)

# ------------------------------------------------------------------------------
# Report and save
# ------------------------------------------------------------------------------
metrics_lasso <- collect_metrics(tune_lasso) %>%
  filter(.config == best_lasso$.config)
message("\n========== Lasso (5-fold CV on train) ==========")
message("Best penalty = ", format(best_lasso$penalty, digits = 4),
        ", mean MSE = ", round(metrics_lasso$mean, 4),
        " (sd = ", round(metrics_lasso$std_err * sqrt(N_FOLDS_OUTER), 4), ")")

metrics_lasso_sq <- collect_metrics(tune_lasso_sq) %>%
  filter(.config == best_lasso_sq$.config)
message("\n========== Lasso with squared predictors (5-fold CV on train) ==========")
message("Best penalty = ", format(best_lasso_sq$penalty, digits = 4),
        ", mean MSE = ", round(metrics_lasso_sq$mean, 4),
        " (sd = ", round(metrics_lasso_sq$std_err * sqrt(N_FOLDS_OUTER), 4), ")")

saveRDS(final_lasso,    file.path(DIR_FITS, "fit_lasso.rds"))
saveRDS(final_lasso_sq, file.path(DIR_FITS, "fit_lasso_sq.rds"))
saveRDS(final_full,     file.path(DIR_FITS, "fit_full.rds"))
saveRDS(final_full_sq,  file.path(DIR_FITS, "fit_full_sq.rds"))
message("\nFitted workflows saved to output/fits/")

# ------------------------------------------------------------------------------
# Write coefficient estimates: one row per term, columns for each model
# Top row: tuned lambda values for lasso models (full_* columns = NA)
# ------------------------------------------------------------------------------
lambda_row <- tibble(
  term         = "lambda",
  lasso_est    = best_lasso$penalty,
  lasso_sq_est = best_lasso_sq$penalty,
  full_est     = NA_real_,
  full_sq_est  = NA_real_
)
coef_lasso <- tidy(extract_fit_parsnip(final_lasso)) %>%
  select(term, estimate) %>%
  rename(lasso_est = estimate)
coef_lasso_sq <- tidy(extract_fit_parsnip(final_lasso_sq)) %>%
  select(term, estimate) %>%
  rename(lasso_sq_est = estimate)
coef_full  <- tidy(extract_fit_parsnip(final_full)) %>%
  select(term, estimate) %>%
  rename(full_est = estimate)
coef_full_sq  <- tidy(extract_fit_parsnip(final_full_sq)) %>%
  select(term, estimate) %>%
  rename(full_sq_est = estimate)

out_coef <- bind_rows(
  lambda_row,
  coef_lasso %>%
    full_join(coef_lasso_sq, by = "term") %>%
    full_join(coef_full,     by = "term") %>%
    full_join(coef_full_sq,  by = "term")
)
write_csv(out_coef, file.path(DIR_FITS, "coefficients.csv"))
message("Coefficients written to ", file.path(DIR_FITS, "coefficients.csv"))

# ------------------------------------------------------------------------------
# Save final model corresponding to best Assessment RMSE
# ------------------------------------------------------------------------------
best_model_path <- file.path(DIR_FITS, "best_model.txt")
best_model <- if (file.exists(best_model_path)) {
  readLines(best_model_path, n = 1)
} else {
  "lasso"
}

final_model <- switch(
  best_model,
  lasso    = final_lasso,
  lasso_sq = final_lasso_sq,
  full     = final_full,
  full_sq  = final_full_sq,
  final_lasso
)

saveRDS(final_model, file.path(DIR_FITS, "fit_final.rds"))
message("Final model type: ", best_model,
        " (saved to ", file.path(DIR_FITS, "fit_final.rds"), ")")
message("Done.")
