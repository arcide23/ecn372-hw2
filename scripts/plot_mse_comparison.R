# ------------------------------------------------------------------------------
# Plot Assessment MSE and MAE by model (Lasso, Full OLS, with/without squared predictors).
# Target is log(shares). Writes output/figures/assessment_mse_mae.pdf.
# Run from project root: Rscript scripts/plot_mse_comparison.R
# ------------------------------------------------------------------------------

source(file.path("src", "setup.R"))

dir.create(DIR_FIGURES, showWarnings = FALSE, recursive = TRUE)
path_fig    <- file.path(DIR_FIGURES, "assessment_mse_mae.pdf")

assessment  <- read_csv(file.path(DIR_ASSESSMENT, "assessment_mse.csv"), show_col_types = FALSE) %>%
  mutate(model = case_when(
    model == "lasso"     ~ "Lasso",
    model == "lasso_sq"  ~ "Lasso + squared",
    model == "full"      ~ "Full OLS",
    model == "full_sq"   ~ "Full OLS + squared",
    TRUE ~ model
  ))

# Reshape for plotting: one figure with two facets (Assessment MSE and Assessment MAE)
plot_dat <- assessment %>%
  select(model, assessment_mse, assessment_mse_sd, assessment_mae, assessment_mae_sd) %>%
  pivot_longer(
    cols = c(assessment_mse, assessment_mae),
    names_to = "metric",
    values_to = "value"
  ) %>%
  mutate(
    metric = dplyr::recode(
      metric,
      assessment_mse = "Assessment MSE",
      assessment_mae  = "Assessment MAE"
    ),
    sd = if_else(metric == "Assessment MSE", assessment_mse_sd, assessment_mae_sd)
  ) %>%
  mutate(model = factor(model, levels = c("Full OLS", "Full OLS + squared", "Lasso", "Lasso + squared")))

val_range <- range(plot_dat$value)
delta <- max(diff(val_range), 1)
y_min <- val_range[1] - 0.02 * delta
y_max <- val_range[2] + 0.22 * delta

p <- ggplot(plot_dat, aes(x = model, y = value)) +
  geom_col(fill = "steelblue", width = 0.7) +
  geom_errorbar(
    aes(ymin = value - sd, ymax = value + sd),
    width = 0.2,
    linewidth = 0.5
  ) +
  geom_text(
    aes(label = sprintf("%.2f", value)),
    vjust = -0.4,
    size = 2.9
  ) +
  coord_cartesian(ylim = c(y_min, y_max)) +
  labs(
    x = "Model",
    y = "Value (log shares)"
  ) +
  facet_wrap(~ metric, ncol = 1, scales = "free_y") +
  theme_minimal() +
  theme(legend.position = "none")

ggsave(path_fig, plot = p, width = FIG_WIDTH, height = FIG_HEIGHT, device = "pdf")
message("Figure saved to ", path_fig)
