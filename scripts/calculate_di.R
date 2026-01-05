library(CAST)
library(ranger)
library(readr)
library(terra)
library(caret)
library(jsonlite)

set.seed(42)

# Path handling for Flask environment
project_root <- Sys.getenv("PROJECT_ROOT")
if (project_root == "") { project_root <- getwd() }

train_data_path   <- "training_data.csv"
predict_data_path <- "prediction_features.csv"
mask_path         <- "building_mask.tif"
meta_path         <- "spatial_meta.json"

message("--- Loading Data ---")
training_data_raw <- read_csv(train_data_path, show_col_types = FALSE)
prediction_features_raw <- read_csv(predict_data_path, show_col_types = FALSE)
building_mask <- rast(mask_path)

# Expand string 'season' into binary features to match prediction CSV
if ("season" %in% colnames(training_data_raw)) {
  for(s in c("autumn", "spring", "summer", "winter")) {
    training_data_raw[[paste0("season_", s)]] <- as.numeric(training_data_raw$season == s)
  }
}

training_data_na_free <- na.omit(training_data_raw)
prediction_features <- na.omit(prediction_features_raw)

# Feature alignment
outcome_col <- "label"
feature_cols <- names(prediction_features)
training_data_sampled <- training_data_na_free[sample(1:nrow(training_data_na_free), min(nrow(training_data_na_free), 1000)), c(outcome_col, feature_cols)]

# Context Model
training_data_sampled[[outcome_col]] <- as.factor(make.names(training_data_sampled[[outcome_col]]))
model <- train(as.formula(paste(outcome_col, "~ .")), data = training_data_sampled, method = "ranger", 
               trControl = trainControl(method = "none"), num.trees = 100, importance = "permutation")

# Uncertainty Calculation
non_na_idx <- which(complete.cases(prediction_features_raw))
aoa_out <- aoa(newdata = prediction_features, model = model, LPD = TRUE)

# Raster reconstruction
meta <- read_json(meta_path, simplifyVector = TRUE)
template <- rast(nrows=meta$height, ncols=meta$width, crs=meta$crs_wkt, 
                 extent=ext(meta$transform[1], meta$transform[1] + meta$width*meta$transform[2], 
                            meta$transform[4] + meta$height*meta$transform[6], meta$transform[4]))

save_raster <- function(vals, name) {
  r <- template
  values(r) <- NA
  r[non_na_idx] <- vals
  r[building_mask == 0] <- NA
  writeRaster(r, paste0("aoi_", name, ".tif"), overwrite=TRUE)
}

save_raster(aoa_out$DI, "di")
save_raster(aoa_out$LPD, "lpd")
message("✅ DI/LPD calculation successful.")