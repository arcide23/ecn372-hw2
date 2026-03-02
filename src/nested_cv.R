# ------------------------------------------------------------------------------
# Nested cross-validation helpers for tidymodels workflows.
# Source after setup.R: source(file.path("src", "nested_cv.R"))
# ------------------------------------------------------------------------------

#' Run inner tuning on one outer split and return best penalty.
#'
#' @param outer_split An rsplit (one fold from the outer resample).
#' @param inner_resamples An rset of inner resamples (e.g. from nested_cv()).
#' @param model_spec A parsnip model spec with penalty = tune().
#' @param rec A recipe.
#' @param grid A tuning grid (e.g. from grid_regular(penalty(...))).
#' @param metric Metric name for select_best (default "rmse").
#' @return The best penalty value (scalar).
tune_inner <- function(outer_split, inner_resamples, model_spec, rec, grid, metric = "rmse") {
  wf <- workflow() %>%
    add_recipe(rec) %>%
    add_model(model_spec)
  tune_res <- tune_grid(
    wf,
    resamples = inner_resamples,
    grid = grid,
    control = control_grid(save_workflow = TRUE),
    metrics = metric_set(rmse)
  )
  best <- select_best(tune_res, metric = metric)
  best$penalty
}

#' Fit workflow with given penalty on outer analysis set; return MSE and MAE on assessment set.
#'
#' @param outer_split An rsplit (one fold from the outer resample).
#' @param penalty_val The penalty value to use (e.g. from tune_inner).
#' @param model_spec A parsnip model spec with penalty = tune().
#' @param rec A recipe.
#' @param outcome Name of the outcome column in the assessment data (default "shares").
#' @return One-row tibble with columns mse, mae.
eval_outer <- function(outer_split, penalty_val, model_spec, rec, outcome = "shares") {
  wf <- workflow() %>%
    add_recipe(rec) %>%
    add_model(model_spec) %>%
    finalize_workflow(tibble(penalty = penalty_val))
  fit <- fit(wf, data = analysis(outer_split))
  pred <- predict(fit, new_data = assessment(outer_split))
  truth <- assessment(outer_split)[[outcome]]
  tibble(
    mse = mean((truth - pred$.pred)^2),
    mae = mean(abs(truth - pred$.pred))
  )
}
