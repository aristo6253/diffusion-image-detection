#!/bin/bash

# Define source and destination directories
ROOT_DIR="/scratch/izar/dimitrio"
SRC_DIR="$ROOT_DIR/DMTestSet_1000"
DEST_DIR="$ROOT_DIR/DMTestSet_1000_new"

# Create new root directory
mkdir -p "$DEST_DIR"

# Array of model names
MODELS=("LDM" "PNDM" "StyleGAN2")

# Choose one real dataset (e.g., CelebAHQ)
REAL_DATASET="CelebAHQ"

# Loop through each model
for model in "${MODELS[@]}"; do
    # Create 0_real and 1_fake directories for each model
    mkdir -p "$DEST_DIR/$model/0_real"
    mkdir -p "$DEST_DIR/$model/1_fake"

    # Copy real images to 0_real directory of each model
    cp "$SRC_DIR/Real/$REAL_DATASET"/* "$DEST_DIR/$model/0_real/"

    # Move fake images to 1_fake directory of each model
    cp "$SRC_DIR/Fake/$model"/* "$DEST_DIR/$model/1_fake/"
done

# Optional: Uncomment the following line to remove the original directory after verification
# rm -rf "$SRC_DIR"

echo "Directory restructuring complete."
