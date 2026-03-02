# Full pipeline (data check + assess + fit + assessment-only plots): make
# Modeling only (no data check): make model
# Evaluate final model on held-out test set: make evaluate

.PHONY: all model data evaluate 

all:
	Rscript scripts/model-assessment.R
	Rscript scripts/model-fit.R
	Rscript scripts/plot_mse_comparison.R

evaluate:
	Rscript scripts/model-evaluate.R
