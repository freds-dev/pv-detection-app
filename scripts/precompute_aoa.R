library(CAST)
library(caret)
library(readr)
library(here)

# 1. Load your existing model and original training data
model <- readRDS(here("data/artifacts/rf/model_final.rds"))
train_data <- read_csv(here("data/training/training.csv")) |> na.omit()

# 2. Precompute the DI reference
# This is the "clean" way: it calculates the multidimensional scaling
# using the exact variable importance from your saved model.
message("Precomputing Train DI Reference...")
train_DI_reference <- trainDI(model, train_data)

# 3. Save this small object (it's only a few KBs)
saveRDS(train_DI_reference, here("data/artifacts/rf/train_di_ref.rds"))
message("Success! Saved AOA reference to: data/artifacts/rf/train_di_ref.rds")