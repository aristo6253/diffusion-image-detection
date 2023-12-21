import os
import random
import shutil
from pathlib import Path

def sub_sample(source_dirs, base_dir, num_images_per_dir=1000, file_types=('.jpg', '.png')):
    """
    Selects a specified number of images randomly from each source directory 
    and copies them to corresponding subdirectories in a base target directory.

    :param source_dirs: List of source directories
    :param base_dir: Base target directory path
    :param num_images_per_dir: Number of images to select from each directory
    :param file_types: Tuple of image file extensions to consider
    """
    for source_dir in source_dirs:
        # Define the target subdirectory
        target_subdir = os.path.join(base_dir, os.path.basename(source_dir))
        
        # Create the target subdirectory if it doesn't exist
        Path(target_subdir).mkdir(parents=True, exist_ok=True)

        # List all image files in the source directory
        image_files = [file for file in Path(source_dir).glob('*') if file.suffix in file_types]
        
        # Randomly select the specified number of images
        selected_images = random.sample(image_files, min(num_images_per_dir, len(image_files)))

        # Copy the selected images to the target subdirectory
        for image in selected_images:
            shutil.copy(image, target_subdir)

# Define the base target directory
source_dirs = ['../data/DDIM_200', '../data/PNDM_200', '../data/ProGAN', '../data/CelebA-HQ-img']
base_target_dir = '../data_sampled'

# Call the function to perform the operation
sub_sample(source_dirs, base_target_dir)
