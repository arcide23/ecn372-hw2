# ------------------------------------------------------------------------------
# Pipeline configuration: paths, seeds, and tuning parameters
# Sourced by setup.R so all scripts get these when they source setup.
# ------------------------------------------------------------------------------

# Reproducibility
SEED <- 42L

# Data paths (relative to project root)
# Use user-provided raw train/test directly (no random split).
# Training data: data/raw/train.csv
# Test data:     data/raw/test.csv (can be provided later).
PATH_TRAIN <- file.path("data", "raw", "train.csv")
PATH_TEST  <- file.path("data", "raw", "test.csv")

# Output paths
DIR_ASSESSMENT <- file.path("output", "assessment")
DIR_FITS       <- file.path("output", "fits")
DIR_TEST       <- file.path("output", "test")
DIR_FIGURES    <- file.path("output", "figures")

# Cross-validation
N_FOLDS_OUTER <- 5L
N_FOLDS_INNER <- 5L

# Lasso tuning grid
PENALTY_RANGE  <- c(-5, 2)
PENALTY_LEVELS <- 25L

# Figure dimensions (inches)
FIG_WIDTH  <- 6
FIG_HEIGHT <- 4
