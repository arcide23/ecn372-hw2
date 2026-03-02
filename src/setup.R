# ------------------------------------------------------------------------------
# Project setup: ensure project root and load packages
# Source this from scripts with: source(file.path("src", "setup.R"))
# ------------------------------------------------------------------------------

# Set project root so paths work when run from terminal (any working directory)
if (basename(getwd()) == "scripts") {
  setwd("..")
}
if (!dir.exists("data")) {
  stop("Run this script from the project root (directory that contains 'data/' and 'scripts/').")
}

# Load pipeline config (paths, seeds, CV/tuning parameters)
source(file.path("src", "config.R"))

# Install packages if missing, then load them
# Add package names to this vector as the project grows
.pkgs <- c("tidyverse", "tidymodels", "glmnet")

install_and_load <- function(pkgs = .pkgs) {
  if (length(pkgs) == 0L) return(invisible())
  missing <- pkgs[!sapply(pkgs, requireNamespace, quietly = TRUE)]
  if (length(missing) > 0L) {
    install.packages(missing, repos = "https://cloud.r-project.org/")
  }
  for (p in pkgs) {
    library(p, character.only = TRUE)
  }
  invisible()
}

install_and_load()
