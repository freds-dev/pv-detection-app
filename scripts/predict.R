# scripts/predict.R
library(ranger)

# --- 1. SETUP & LOGGING ---
message("--- Starting R Prediction Script ---")

# Get the session directory path passed from Python
args <- commandArgs(trailingOnly = TRUE)

if (length(args) == 0) {
  stop("CRITICAL ERROR: No session directory provided by Python.")
}

session_dir <- args[1]
message(paste("Working Directory set to:", session_dir))
setwd(session_dir)

# --- 2. LOAD THE MODEL ---
# Python runs this from static/results/[UUID]
# We need to go up 3 levels to reach the project root, then into data/
model_path <- "../../../data/artifacts/rf/model_final.rds"

if (!file.exists(model_path)) {
  # If it fails, let's see exactly where R is looking
  message(paste("Current Folder:", getwd()))
  stop(paste("CRITICAL ERROR: Model file not found at:", model_path))
}

message("Loading .rds model...")
model_caret <- readRDS(model_path)

# Extract the raw ranger model from the caret wrapper for faster inference
# This assumes you used caret::train(). If you saved ranger directly, 
# you might need to use model_caret directly.
final_model <- if(inherits(model_caret, "train")) model_caret$finalModel else model_caret
message("Model extracted successfully.")

# --- 3. LOAD INPUT DATA ---
if (!file.exists("prediction_features.csv")) {
  stop("CRITICAL ERROR: prediction_features.csv was not found in the session folder.")
}

features <- read.csv("prediction_features.csv")
message(paste("Predicting for", nrow(features), "pixels..."))

# --- 4. RUN INFERENCE ---
# We use the raw ranger model to get probabilities
# This requires that the model was trained with 'probability = TRUE'
pred_obj <- predict(final_model, data = features)

# --- 5. SAVE RESULTS ---
# The output will contain columns for each class (e.g., PV, NoPV)
write.csv(pred_obj$predictions, "prediction_results.csv", row.names = FALSE)

message("Results written to prediction_results.csv")
message("--- R Script Finished Successfully ---")