# -----------------------------------------------------------------------------
# Calculate Dissimilarity Index (DI) and Local Data Point Density (LPD)
# -----------------------------------------------------------------------------

# --- 1. Load Libraries ---
library(here)
library(CAST)      # For AOA (DI/LPD)
library(ranger)    # For the random forest model
library(readr)
library(terra)
library(caret)     # For the train() function
library(jsonlite)

set.seed(42)

# --- 2. Configuration ---
# All paths are now simple and point to the root working directory
train_data_path   <- "training_data.csv"
predict_data_path <- "prediction_features.csv"
mask_path         <- "building_mask.tif"
meta_path         <- "spatial_meta.json"
output_dir        <- "."  # "." means the current working directory

# Parameters
num_trees <- 100 # Reduced from 400 for speed

# --- 3. Load Data ---
message("--- START: Data Loading ---")
# This is your ORIGINAL training data
training_data_raw <- read_csv(train_data_path, show_col_types = FALSE)
# This is your NEW on-the-fly prediction data
prediction_features_raw <- read_csv(predict_data_path, show_col_types = FALSE)
# This is the on-the-fly building mask
building_mask <- rast(mask_path)

# Clean NA values from both datasets
training_data_na_free <- na.omit(training_data_raw)
prediction_features <- na.omit(prediction_features_raw)

message(paste("Loaded", nrow(training_data_raw), "training samples, cleaned to", nrow(training_data_na_free), "rows."))
message(paste("Loaded", nrow(prediction_features_raw), "prediction pixels, cleaned to", nrow(prediction_features), "rows."))
message("--- END: Data Loading ---\n")


# -----------------------------------------------------------------------------
# --- 4. Train Model for Feature Space Context ---
# -----------------------------------------------------------------------------

outcome_col <- "label"


# --- Filter training data to match prediction data ---
# This is the key fix. It ensures the model is trained on *only*
# the features that exist in the prediction data.

# 1. Get the list of features from the prediction data
feature_cols <- names(prediction_features)
message(paste("Identified", length(feature_cols), "features from prediction data."))

# 2. Create the final, filtered training dataset:
# It will contain ONLY the outcome column and the feature columns.
# This prevents crashes from extra columns (like 'x', 'y', 'id')
training_data_filtered <- training_data_na_free[, c(outcome_col, feature_cols)]


# --- Subsample the training data ---
max_train_samples <- 5000 # Use 5000 samples

if (nrow(training_data_filtered) > max_train_samples) {
  message(paste("Subsampling full training data from", nrow(training_data_filtered), "to", max_train_samples, "rows."))
  sample_indices <- sample(1:nrow(training_data_filtered), max_train_samples)
  training_data_sampled <- training_data_filtered[sample_indices, ]
} else {
  training_data_sampled <- training_data_filtered
}


message("--- START: Context Model Training ---")
message(paste("Training ranger model on", nrow(training_data_sampled), "samples with", num_trees, "trees."))

# Make sure the outcome column is a factor
training_data_sampled[[outcome_col]] <- as.factor(training_data_sampled[[outcome_col]])

# Create the model formula dynamically (e.g., "label ~ .")
model_formula <- as.formula(paste(outcome_col, "~ ."))

model <- train(model_formula,
               data = training_data_sampled, # Use the SAMPLED & FILTERED data
               method = "ranger",
               na.action = na.omit, 
               trControl = trainControl(method = "none"),
               num.trees = num_trees,
               importance = "permutation",
               num.threads = 0) 
message("--- END: Context Model Training ---\n")


# --- 5. Calculate DI and LPD ---
# This step will now work, as the model's features and the
# prediction_features' columns are guaranteed to be identical.
message("--- START: DI and LPD Calculation ---")
dissimilarity_result <- aoa(newdata = prediction_features, model = model, LPD = TRUE)
message("DI and LPD calculation complete.")
message("--- END: DI and LPD Calculation ---\n")


# --- 6. Create, Mask, and Save GeoTIFFs ---
message("--- START: GeoTIFF Saving ---")
meta <- read_json(meta_path, simplifyVector = TRUE)
transform_vals <- meta$transform
xmin <- transform_vals[1]
xmax <- transform_vals[1] + (meta$width * transform_vals[2])
ymax <- transform_vals[4]
ymin <- transform_vals[4] + (meta$height * transform_vals[6])

template_raster <- rast(
  nrows = meta$height, ncols = meta$width, crs = meta$crs_wkt,
  extent = c(xmin, xmax, ymin, ymax)
)

save_geotiff <- function(values, file_path) {
  raster <- template_raster
  
  # We must re-index the values, as `na.omit` may have removed some pixels.
  non_na_indices <- which(complete.cases(prediction_features_raw))
  
  values(raster) <- NA
  raster[non_na_indices] <- values
  
  # Apply the building mask
  raster[building_mask == 0] <- NA
  writeRaster(raster, file_path, overwrite = TRUE)
  message(paste0("✅ Masked map saved to: ", file_path))
}

# Save to the root output directory
save_geotiff(dissimilarity_result$DI, file.path(output_dir, "aoi_di.tif"))
save_geotiff(dissimilarity_result$LPD, file.path(output_dir, "aoi_lpd.tif"))
message("--- END: GeoTIFF Saving ---")