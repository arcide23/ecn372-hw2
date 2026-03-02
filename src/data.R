# ------------------------------------------------------------------------------
# Data helpers: load train/test with modeling prep (drop url, log(shares), drop shares)
# Source after setup.R (and thus config). Uses PATH_TRAIN, PATH_TEST from config.
# ------------------------------------------------------------------------------

load_train_data <- function() {
  read_csv(PATH_TRAIN, show_col_types = FALSE) %>%
    select(-url) %>%
    mutate(log_shares = log(shares)) %>%
    select(-shares)
}

load_test_data <- function() {
  read_csv(PATH_TEST, show_col_types = FALSE) %>%
    select(-url) %>%
    mutate(log_shares = log(shares)) %>%
    select(-shares)
}
