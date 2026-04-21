library(caret)
library(CAST)
library(ranger)
library(jsonlite)

message("--- Starting Optimized Prediction & AOA Script ---")
args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 2) {
  stop("CRITICAL ERROR: Missing arguments. Expected [session_dir] [run_aoa_flag]")
}

session_dir <- args[1]
run_aoa     <- as.logical(args[2])

project_root <- "/Users/fred/Documents/uni/thesis/prediction-app"
model_path   <- file.path(project_root, "data/artifacts/rf/model_final.rds")
train_path   <- file.path(project_root, "data/training/training.csv")

message(paste("Session Directory:", session_dir))
setwd(session_dir)

if (!file.exists(model_path)) stop(paste("Model file missing:", model_path))
model_caret <- readRDS(model_path)

if (!file.exists("prediction_features.csv")) stop("prediction_features.csv not found in session dir.")
features <- read.csv("prediction_features.csv")

predictor_names <- c("B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12")

message("Executing Inference...")

probs <- predict(model_caret, newdata = features, type = "prob")

output_preds <- data.frame(X1 = probs$X1)
write.csv(output_preds, "prediction_results.csv", row.names = FALSE)

if (run_aoa) {
  if (!file.exists(train_path)) {
    message("WARNING: Training CSV missing. Skipping AOA.")
  } else {
    message("Computing AOA and LPD from scratch using kNNDM folds...")
    
    train_dat <- read.csv(train_path)
    
    missing_cols <- setdiff(predictor_names, names(train_dat))
    if(length(missing_cols) > 0) {
      stop(paste("Training CSV is missing required bands:", paste(missing_cols, collapse=", ")))
    }

    aoa_results <- aoa(
      newdata = features[, predictor_names],
      model = model_caret,
      train = train_dat[, predictor_names],
      LPD = TRUE,
      verbose = TRUE
    )
    
    metrics_df <- data.frame(
      DI = aoa_results$DI,
      LPD = aoa_results$LPD,
      AOA = aoa_results$AOA
    )
    
    write.csv(metrics_df, "spatial_metrics.csv", row.names = FALSE)
    
    message(paste("AOA Threshold Applied (kNNDM):", round(aoa_results$parameters$threshold, 4)))
  }
}

message("--- R Script Finished Successfully ---")
