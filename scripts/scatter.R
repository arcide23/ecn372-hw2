source(file.path("src", "setup.R"))
source(file.path("src", "data.R"))

dir.create(DIR_FIGURES, showWarnings = FALSE, recursive = TRUE)

train_data <- read_csv(PATH_TRAIN, show_col_types = FALSE) %>%
  select(-url) %>%
  mutate(log_shares = log(shares))

p_shares <- ggplot(train_data, aes(x = shares)) +
  geom_histogram(bins = 50, fill = "steelblue", colour = "white", linewidth = 0.2) +
  labs(x = "shares", y = "Count", title = "Histogram of shares") +
  theme_minimal()

p_log <- ggplot(train_data, aes(x = log_shares)) +
  geom_histogram(bins = 50, fill = "darkorange", colour = "white", linewidth = 0.2) +
  labs(x = "log(shares)", y = "Count", title = "Histogram of log(shares)") +
  theme_minimal()

ggsave(file.path(DIR_FIGURES, "shares_histogram.pdf"),
       plot = p_shares, width = FIG_WIDTH, height = FIG_HEIGHT, device = "pdf")

ggsave(file.path(DIR_FIGURES, "log_shares_histogram.pdf"),
       plot = p_log, width = FIG_WIDTH, height = FIG_HEIGHT, device = "pdf")
