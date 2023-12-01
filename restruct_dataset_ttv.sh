#!/bin/bash

# Define source and destination directories
ROOT_DIR="/scratch/izar/dimitrio"
SRC_DIR="$ROOT_DIR/DMTestSet_1000"
DEST_DIR="$ROOT_DIR/DMTestSet_1000_ttv2"

# Array of model names
MODELS=("LDM" "PNDM" "StyleGAN2")

# Choose one real dataset (e.g., CelebAHQ)
REAL_DATASET="CelebAHQ"

# Function to split data into train, val, and test directories
split_data() {
    local src_dir=$1
    local dest_dir_base=$2
    local array=($(ls "$src_dir"))
    local total=${#array[@]}
    local train_end=$(($total * 8 / 10))  # 80% for training
    local val_end=$(($train_end + $total / 10)) # Next 10% for validation

    # Split data
    for i in "${!array[@]}"; do
        if [ $i -lt $train_end ]; then
            dest_dir="$dest_dir_base/train"
        elif [ $i -lt $val_end ]; then
            dest_dir="$dest_dir_base/val"
        else
            dest_dir="$dest_dir_base/test"
        fi
        mkdir -p "$dest_dir"
        cp "$src_dir/${array[$i]}" "$dest_dir/"
    done
}

# Loop through each split (train, val, test)
for split in "train" "val" "test"; do
    # Loop through each model
    for model in "${MODELS[@]}"; do
        # Create and split real data
        split_data "$SRC_DIR/Real/$REAL_DATASET" "$DEST_DIR/$split/$model/0_real"

        # Create and split fake data
        split_data "$SRC_DIR/Fake/$model" "$DEST_DIR/$split/$model/1_fake"
    done
done

echo "Directory restructuring complete."
