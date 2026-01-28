library(ranger)
library(CAST)
library(jsonlite)

# --- 1. SETUP ---
message("--- Starting R Prediction & AOA Script ---")
args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 2) {
  stop("CRITICAL ERROR: Missing arguments. Expected [session_dir] [run_aoa_flag]")
}

session_dir <- args[1]
run_aoa     <- as.logical(args[2])

message(paste("Session Directory:", session_dir))
message(paste("AOA Requested:", run_aoa))
setwd(session_dir)

# --- 2. LOAD CORE ARTIFACTS ---
model_path  <- "../../../data/artifacts/rf/model_final.rds"
di_ref_path <- "../../../data/artifacts/rf/train_di_ref.rds"

if (!file.exists(model_path)) stop(paste("Model not found at:", model_path))

model_caret <- readRDS(model_path)
features    <- read.csv("prediction_features.csv")

# --- 3. RUN PREDICTION (Always) ---
message("Executing Random Forest inference...")
final_model <- if(inherits(model_caret, "train")) model_caret$finalModel else model_caret
pred_obj    <- predict(final_model, data = features)
write.csv(pred_obj$predictions, "prediction_results.csv", row.names = FALSE)

if (run_aoa) {
  # Load the precomputed reference instead of the full training CSV
  train_di_ref <- readRDS("../../../data/artifacts/rf/train_di_ref.rds")
  
  # Calculate DI, LPD, and AOA simultaneously
  aoa_results <- aoa(
    newdata = features, 
    model = model_caret, 
    trainDI = train_di_ref, 
    LPD = TRUE
  )
  
  # Save all metrics for Python to process
  metrics_df <- data.frame(
    DI = aoa_results$DI,
    LPD = aoa_results$LPD,
    AOA = aoa_results$AOA
  )
  write.csv(metrics_df, "spatial_metrics.csv", row.names = FALSE)
}

message("--- R Script Finished Successfully ---")