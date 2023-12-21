import os, shutil
from pathlib import Path

def create_subfolders(base_dir, subfolders):
    for subfolder in subfolders:
        Path(base_dir, subfolder).mkdir(parents=True, exist_ok=True)

def copy_images(source_folder, dest_folder, is_real):
    prefix = '0_real' if is_real else '1_fake'
    count = 0
    for file in os.listdir(source_folder):
        shutil.copy(os.path.join(source_folder, file), os.path.join(dest_folder, prefix, file))
        count += 1
    return count

def count_files(directory):
    return sum([len(files) for _, _, files in os.walk(directory)])

# Define sets
train_val_set = ['ProGAN', 'PNDM_200']
test_set = ['DDIM_200', 'PNDM_200', 'ProGAN']
real_set = 'CelebA-HQ-img'

# Define source and destination directories
source_dir = '../data/'
dest_dir = './data_restruct_ProGAN'

# Create base directories and initialize counters
print("Creating base directories...")
create_subfolders(dest_dir, ['train/0_real', 'train/1_fake', 'val/0_real', 'val/1_fake', 'test'])
image_counts = {'train': {'0_real': 0, '1_fake': 0}, 'val': {'0_real': 0, '1_fake': 0}, 'test': {}}

# Process test set
print("Processing test set...")
for dataset in test_set:
    print(f"Processing {dataset} for test set...")
    dataset_source_dir = os.path.join(source_dir, dataset)
    dataset_dest_dir = os.path.join(dest_dir, 'test', dataset)
    create_subfolders(dataset_dest_dir, ['0_real', '1_fake'])
    image_counts['test'][dataset] = {'0_real': 0, '1_fake': 0}

    # Copy real images (from CelebA-HQ-img) to each test dataset's 0_real folder
    if os.path.exists(os.path.join(source_dir, real_set)):
        print(f"Copying real images to {dataset_dest_dir}/0_real")
        image_counts['test'][dataset]['0_real'] += copy_images(os.path.join(source_dir, real_set), dataset_dest_dir, True)

    # Copy fake images to 1_fake folder
    print(f"Copying fake images to {dataset_dest_dir}/1_fake")
    image_counts['test'][dataset]['1_fake'] += copy_images(dataset_source_dir, dataset_dest_dir, False)

# Process train and val sets
print("Processing train and validation sets...")
for dataset in train_val_set:
    print(f"Processing {dataset} for train and val sets...")
    # Copy real images to train/val 0_real folder
    if os.path.exists(os.path.join(source_dir, real_set)):
        print(f"Copying real images to train and val 0_real folders")
        image_counts['train']['0_real'] += copy_images(os.path.join(source_dir, real_set), os.path.join(dest_dir, 'train'), True)
        image_counts['val']['0_real'] += copy_images(os.path.join(source_dir, real_set), os.path.join(dest_dir, 'val'), True)

    # Copy fake images to train/val 1_fake folder
    print(f"Copying fake images to train and val 1_fake folders")
    dataset_source_dir = os.path.join(source_dir, dataset)
    image_counts['train']['1_fake'] += copy_images(dataset_source_dir, os.path.join(dest_dir, 'train'), False)
    image_counts['val']['1_fake'] += copy_images(dataset_source_dir, os.path.join(dest_dir, 'val'), False)

print("Data restructuring completed.")

# Display the count of images in each folder
for set_type, datasets in image_counts.items():
    print(f"\n{set_type.upper()} SET:")
    for dataset, counts in datasets.items():
        if isinstance(counts, dict):
            for subfolder, count in counts.items():
                print(f"  {dataset}/{subfolder}: {count} images")
        else:
            print(f"  {dataset}: {counts} images")
