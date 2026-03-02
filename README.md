## ECN 372 – Homework 2

**Goal**: predict article popularity (`shares`) on a held-out test set using the Online News Popularity data.

`make evaluate` is the grading entry point; it must print only the test MSE.

---

## Data and target

- **Train data**: `data/raw/train.csv`
- **Test data**: `data/raw/test.csv` (added at grading time)
- **Documentation**: `OnlineNewsPopularity.names`

Preprocessing (implemented in `src/data.R`):

- **Drop** `url` (identifier).
- **Create** `log_shares = log(shares)` and use **`log_shares` as the modeling target**.
- Keep all other features as candidate predictors.

**Why log(shares)?**  
`shares` is highly skewed with heteroskedastic errors. Modeling `log(shares)` stabilizes variance and makes linear methods more appropriate. All MSE/MAE metrics are computed in log space.

---

## Features and transformations

I fit models on:

- **Original standardized predictors** (`rec`):  
  `recipe(log_shares ~ ., data = train_data) %>% step_normalize(all_numeric_predictors())`

- **Original + squared terms** for a **whitelist of count-like variables** (`rec_sq`):  
  `timedelta`, `n_tokens_title`, `n_tokens_content`, `num_hrefs`, `num_self_hrefs`,
  `num_imgs`, `num_videos`, `average_token_length`, `num_keywords`.

For each chosen variable `x`, I add `x_sq = x^2` and then standardize **all** numeric predictors.

**Why only these squared terms?**

- They are natural candidates for nonlinear patterns (e.g. diminishing returns).
- I **avoid squaring binaries and proportions** (e.g. `weekday_is_*`, LDA topics) to reduce collinearity and numerical instability.

---

## Models

I compare four models:

1. **Lasso** (original predictors)  
   - `linear_reg(penalty = tune(), mixture = 1) %>% set_engine("glmnet")` on `rec`.

2. **Lasso + squared**  
   - Same Lasso spec on `rec_sq` (original + squared terms).

3. **Full OLS** (original predictors)  
   - `linear_reg() %>% set_engine("lm")` on `rec`.

4. **Full OLS + squared (Ridge)**  
   - `linear_reg(penalty = 0.01, mixture = 0) %>% set_engine("glmnet")` on `rec_sq`.

**Why Ridge for the squared model?**  
Adding squared terms introduces multicollinearity even with a whitelist. OLS became numerically unstable (huge coefficients, huge fold errors), so I use a small Ridge penalty for stability.

---

## Cross-validation, tuning, and metrics

### Nested 5×5 CV for Lasso (model-assessment)

Located in `src/nested_cv.R` and `scripts/model-assessment.R`:

- **Outer CV**: `vfold_cv(train_data, v = 5)` for assessment.
- **Inner CV**: 5-fold inner resamples per outer split via `nested_cv()`.
- **Tuning metric**: **RMSE** using `metric_set(rmse)` (yardstick’s built-in).
- **Inner tuning** (`tune_inner`): choose `penalty` that minimizes inner-CV RMSE.
- **Outer evaluation** (`eval_outer`): for each outer split:
  - Fit with tuned `penalty`.
  - Compute **MSE** and **MAE** manually:
    - `mse = mean((truth - pred)^2)`
    - `mae = mean(abs(truth - pred))`

Assessment MSE for each Lasso model is `mean(mse)` across outer folds; SD is `sd(mse)`.

### 5-fold CV for OLS and Ridge models

In `model-assessment.R`:

- Use the same 5 outer folds for:
  - Full OLS (`rec` + `lm`)
  - Full OLS + squared (`rec_sq` + Ridge)
- Use `fit_resamples(..., metrics = metric_set(rmse, mae))`.
- Convert RMSE to **MSE** for reporting:
  - `MSE = mean(rmse^2)` across folds; SD from `rmse^2`.

### Reported metrics

`scripts/model-assessment.R` prints and writes (to `output/assessment/assessment_mse.csv`):

- **Assessment MSE** and its SD (primary metric)
- **Assessment MAE** and its SD

All metrics are in **log(shares)** space.

---

## Model selection and final training

### Selecting the best model type

From `assessment_mse.csv`, I choose the model type with the **lowest Assessment MSE** among:

- `lasso`, `lasso_sq`, `full`, `full_sq`

This selection is recorded in `output/fits/best_model.txt`.

### Refitting on full training data

In `scripts/model-fit.R`, I:

1. Fit all four models on the **entire training set** using the tuned penalties for Lasso:
   - Save as `fit_lasso.rds`, `fit_lasso_sq.rds`, `fit_full.rds`, `fit_full_sq.rds`.
2. Read `best_model.txt` and select the corresponding fitted workflow.
3. Save the chosen **final model** to:
   - `output/fits/fit_final.rds`

**Reasoning**:  
Nested/outer CV is used only to select the model type; once chosen, I refit on all available training data for maximum efficiency.

---

## Evaluation on test set (`make evaluate`)

The script `scripts/model-evaluate.R`:

1. Loads `output/fits/fit_final.rds`.
2. Reads `data/raw/test.csv`, drops `url`, and creates `log_shares = log(shares)`.
3. Predicts `log_shares` and computes:

   ```r
   mse <- mean((test_data$log_shares - pred$.pred)^2)
   cat("MSE:", mse, "\n")
   ```

The `Makefile` defines:

```make
evaluate:
	Rscript scripts/model-evaluate.R
```

So **`make evaluate`** prints exactly one line:

```text
MSE: <value>
```

with no extra output.

---

## Make targets

- **`make all`**:
  - Checks that train/test files exist (`scripts/split_data.R`).
  - Runs model assessment (`scripts/model-assessment.R`).
  - Fits all models and saves the final one (`scripts/model-fit.R`).
  - Produces an Assessment MSE/MAE comparison plot (`scripts/plot_mse_comparison.R` → `output/figures/assessment_mse_mae.pdf`).

- **`make model`**:
  - Same as `all` but without the initial data check.

- **`make evaluate`**:
  - Evaluates `fit_final.rds` on `data/raw/test.csv` and prints test MSE.

---

## AI usage

I used AI tools (Cursor/ChatGPT) to:

- Help refactor the original assignment scaffold into a coherent pipeline (`config.R`, `setup.R`, `data.R`, `model-assessment.R`, `model-fit.R`, `model-evaluate.R`).
- Design and implement:
  - Nested CV for Lasso (`src/nested_cv.R`).
  - The squared-feature whitelist and Ridge formulation for the squared OLS model.
- Convert the code to **report MSE** consistently while tuning on RMSE where required by the library.
- Ensure that `make all` and `make evaluate` satisfy the assignment’s exact behavior.

All code and modeling decisions were reviewed and adjusted by me; I understand and can justify each design choice described above.

## ECN 372 – Homework 2: Modeling Choices and Rationale

This repository implements a predictive model for **online article popularity** (`shares`) using the **Online News Popularity** data.  
This README explains, in detail, the **choices I made** and **why** I made them:

---

Everything is wired so that:

- `make all` runs the assessment, fits models, and produces comparison plots.
- `make evaluate` reads `data/raw/test.csv`, evaluates the final model, and prints **only** the test MSE.

---

## Preprocessing and target transformation

Before I worked on the models, I created a histogram of `shares` (output/figures/shares_histogram.pdf), which made it clear that it is heavily right-skewed. To fix this, I used a log-transformation, which made the data much more normally distributed (output/figures/log_shares_histogram.pdf).

Because of this, I took the log of shares in both the training and test data.

---

## Feature engineering: 

I added square terms the models to potentially capture nonlinear relationships between the predictors and `shares`. After adding the squared terms, I standardize all numeric predictors (`step_normalize(all_numeric_predictors())`) so that Lasso penalties are comparable across features.

I considered adding global squared terms for all numeric predictors, but this created severe multicollinearity and unstable OLS fits, especially for binary indicators.

To avoid this, I restricted squared terms to a whitelist of count-like predictors would be less problematic for collinearity:

- `timedelta`
- `n_tokens_title`
- `n_tokens_content`
- `num_hrefs`
- `num_self_hrefs`
- `num_imgs`
- `num_videos`
- `average_token_length`
- `num_keywords`

---

## Models considered

I fit **four candidate models** on the same standardized feature sets:

1. **Lasso (original predictors)**  

2. **Lasso + squared predictors**  

3. **Full OLS (original predictors)**  

4. **“Full OLS + squared predictors” (Ridge with squared predictors)**  

**Why Ridge instead of plain OLS for the squared model?**

- With squared terms, even after whitelisting, there is still substantial multicollinearity.
- Plain OLS produced very large and unstable coefficients and extremely large fold‑wise errors.
- A small Ridge penalty regularized these coefficients and yielded much more stable out‑of‑sample behavior.

---

## Cross-validation, tuning, and metrics

### Nested 5×5 CV for Lasso models

- **Outer CV**: `outer_folds <- vfold_cv(train_data, v = N_FOLDS_OUTER)` (5 folds).
  - Used to estimate **Assessment MSE**.
- **Inner CV** (per outer split): `nested_cv()` builds 5 inner folds on the outer **analysis** set.
- **Inner tuning** (`tune_inner()`):

- **Why RMSE for tuning, if I ultimately report MSE?**  
  My installed yardstick version does not expose a built‑in `mse` metric function, so `metric_set(mse)` fails.  
  I therefore:
  - Tune on **RMSE** (which exists and is smooth/monotone in MSE),
  - Then convert RMSE to **MSE for reporting** by squaring when aggregating.

- **Outer evaluation** (`eval_outer()`):

  ```r
  eval_outer <- function(outer_split, penalty_val, model_spec, rec, outcome = "log_shares") {
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
  ```

  - This is completely manual, so I avoid reliance on `mse_vec()` (which may not exist).
  - Assessment MSE is then `mean(mse)` across outer folds, with SD `sd(mse)`.

### 5‑fold CV for OLS and Ridge models

For the OLS and Ridge models in `model-assessment.R`:

- I reuse the same **outer 5‑fold splits** (`outer_folds`) to make comparisons fair.
- I call `fit_resamples()` with:

  ```r
  metrics = metric_set(rmse, mae)
  ```

- Then I **convert RMSE to MSE** for reporting:

  ```r
  metrics_full_rmse <- metrics_full %>% filter(.metric == "rmse") %>% pull(.estimate)
  assessment_mse_full    <- mean(metrics_full_rmse^2)
  assessment_mse_sd_full <- sd(metrics_full_rmse^2)
  ```

### Summary of metrics

- **Tuning metric**: RMSE (due to yardstick support).
- **Reported metrics**:
  - **Assessment MSE** and its SD (for all four models).
  - **Assessment MAE** and its SD.
  - **Test MSE** (for the final model) in `model-test.R` and `model-evaluate.R`, all in **log(shares)** space.

---

## Model selection

After running `model-assessment.R`, I have:

- A tibble with rows `lasso`, `lasso_sq`, `full`, `full_sq` and columns:
  - `assessment_mse`, `assessment_mse_sd`
  - `assessment_mae`, `assessment_mae_sd`

I then:

- **Choose the best model type** as the one with the **lowest Assessment MSE**:

  ```r
  best_idx   <- which.min(out_assessment$assessment_mse)
  best_model <- out_assessment$model[best_idx]
  best_mse   <- out_assessment$assessment_mse[best_idx]
  ```

- Save the best model name to `output/fits/best_model.txt`.

**Rationale**:

- Assessment MSE (nested or outer CV) is my estimate of **generalization error** on log(shares).
- Using the same metric for **selection** and **reporting** keeps the process coherent.
- I keep MAE as a secondary diagnostic for robustness and interpretability but not as the primary selection criterion.

---

## Final model training

In `scripts/model-fit.R`:

1. I **refit all four models** on the **full training data**:
   - `fit_lasso.rds`
   - `fit_lasso_sq.rds`
   - `fit_full.rds`
   - `fit_full_sq.rds`

2. I then read `best_model.txt` and select the corresponding fitted object:

   ```r
   final_model <- switch(
     best_model,
     lasso    = final_lasso,
     lasso_sq = final_lasso_sq,
     full     = final_full,
     full_sq  = final_full_sq,
     final_lasso  # fallback
   )
   ```

3. I save this as my **final model**:

   ```r
   saveRDS(final_model, file.path(DIR_FITS, "fit_final.rds"))
   ```

**Rationale**:

- Always refitting on the full training set ensures I use **all available data** for the final model once the type (Lasso vs OLS vs squared) has been chosen.
- Keeping all four fitted models in `output/fits/` helps with diagnostics and potential post‑hoc inspection of coefficients.

---

## Evaluation on the held‑out test set (`make evaluate`)

The script `scripts/model-evaluate.R` implements the grading entry point:

1. Loads configuration and data helpers.
2. Loads the final model from `output/fits/fit_final.rds`.
3. Reads `data/raw/test.csv`, drops `url`, and computes `log_shares = log(shares)`.
4. Computes predictions \(\hat{y}\) for `log_shares`.
5. Computes **test MSE in log space**:

   ```r
   mse <- mean((test_data$log_shares - pred$.pred)^2)
   cat("MSE:", mse, "\n")
   ```

This is exactly what `make evaluate` does, via the `Makefile`:

```make
evaluate:
	Rscript scripts/model-evaluate.R
```

**Rationale**:

- Keeping the evaluation logic in a **single, simple script** minimizes the chance of mistakes when grading.
- Printing only the numeric MSE line (`MSE: <value>`) satisfies the assignment requirement and makes automated checking easy.

---

## Makefile targets

The `Makefile` defines:

- **`make all`**:

  ```make
  all:
  	Rscript scripts/split_data.R
  	Rscript scripts/model-assessment.R
  	Rscript scripts/model-fit.R
  	Rscript scripts/plot_mse_comparison.R
  ```

  - Verifies train/test files.
  - Runs assessment (nested/outer CV, writes `assessment_mse.csv`).
  - Fits all four models and saves `fit_final.rds`.
  - Produces `output/figures/assessment_mse_mae.pdf`.

- **`make model`**:

  ```make
  model:
  	Rscript scripts/model-assessment.R
  	Rscript scripts/model-fit.R
  	Rscript scripts/plot_mse_comparison.R
  ```

  - Same as `all` but without the initial data check.

- **`make evaluate`**:

  ```make
  evaluate:
  	Rscript scripts/model-evaluate.R
  ```

**Rationale**:

- Keeps the full pipeline (`all`), modeling-only (`model`), and evaluation (`evaluate`) clearly separated.
- Mirrors the grading workflow: I expect the grader will primarily run `make evaluate`, with `make all` available for full reproduction.

---

## AI usage

I used **AI tooling (Cursor / ChatGPT)** to assist with:

- Refactoring the original assignment scripts into a cleaner pipeline:
  - Separating configuration (`config.R`), setup (`setup.R`), data helpers (`data.R`), and modeling scripts.
- Designing and implementing:
  - Nested cross-validation for the Lasso models (`src/nested_cv.R`).
  - The squared‑feature recipes and the whitelist for stable squared terms.
  - The Ridge version of the “Full OLS + squared” model for numerical stability.
- Converting all reported metrics from RMSE to **MSE** while remaining compatible with the installed `yardstick` version (tuning on RMSE, reporting MSE via manual computations).
- Ensuring that:
  - `make all` and `make evaluate` behave exactly as required.
  - The final evaluation prints only `MSE: <value>` to stdout.

All code and modeling decisions have been **reviewed and adapted by me** to ensure they are appropriate for this assignment and that I understand the reasoning behind each step.

