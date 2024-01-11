import os
import shutil
import random
from pathlib import Path

def create_subfolders(base_dir, subfolders):
    for subfolder in subfolders:
        Path(base_dir, subfolder).mkdir(parents=True, exist_ok=True)

def split_and_organize_data(source_dir, dest_dir, train_val_set, test_set, real_set):
    def split_and_copy_files(source_folder, dest_base_folder, train_ratio, val_ratio, dataset_name="", is_real=False):
        files = os.listdir(source_folder)
        n = len(files)
        print(f"Processing {source_folder}: Total {n} images")

        shuffled_files = random.sample(files, len(files))

        train_end = int(train_ratio * n)
        val_end = train_end + int(val_ratio * n)

        # Copy files to respective folders
        for idx, file in enumerate(shuffled_files):
            subdir = '0_real' if is_real else '1_fake'
            if idx < train_end:
                dest_path = os.path.join(dest_base_folder, 'train', subdir, file)
            elif train_end <= idx < val_end:
                dest_path = os.path.join(dest_base_folder, 'val', subdir, file)
            else:
                dest_path = os.path.join(dest_base_folder, 'test', dataset_name, subdir, file)

            # Ensure destination directory exists
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            # print(f'Dest: {dest_path}')
            # print(f'Source: {source_folder}')
            shutil.copy(os.path.join(source_folder, file), dest_path)

        print(f"Split into Train: {train_end}, Val: {val_end - train_end}, Test: {n - val_end}")

    # Create the base directory and subfolders
    subfolders = ['train/0_real', 'train/1_fake', 'val/0_real', 'val/1_fake']
    for dataset in test_set:
        subfolders.extend([f"test/{dataset}/0_real", f"test/{dataset}/1_fake"])
    create_subfolders(dest_dir, subfolders)

    # Process real dataset
    real_source_dir = os.path.join(source_dir, real_set)
    if os.path.exists(real_source_dir):
        print("Processing Real Dataset...")
        split_and_copy_files(real_source_dir, dest_dir, 0.8, 0.1, is_real=True)

    # Process fake datasets for training and validation
    for dataset in train_val_set:
        dataset_source_dir = os.path.join(source_dir, dataset)
        if os.path.exists(dataset_source_dir):
            print(f"Processing Training/Validation Dataset: {dataset}")
            split_and_copy_files(dataset_source_dir, dest_dir, 0.8, 0.1, dataset)

    # Process datasets for testing
    for dataset in test_set:
        dataset_source_dir = os.path.join(source_dir, dataset)
        if os.path.exists(dataset_source_dir):
            print(f"Processing Test Dataset: {dataset}")
            split_and_copy_files(dataset_source_dir, dest_dir, 0.8, 0.1, dataset)

# Paths and set definitions
source_dir = '../data/'
dest_dir = './data_restruct_ProGAN_PNDM'
train_val_set = ['ProGAN', 'PNDM_200']
test_set = ['PNDM_200', 'DDIM_200', 'DDPM_200', 'LDM_200', 'ProGAN', 'StyleGAN2_tmp']
real_set = 'CelebA-HQ-img'

# Run the function
split_and_organize_data(source_dir, dest_dir, train_val_set, test_set, real_set)
