import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

# Datasets dictionary
datasets = {
    'celeb': 'CelebA-HQ-img',
    'ddim': 'DDIM_200',
    'pndm': 'PNDM_200',
    'progan': 'ProGAN',
    'ddpm': 'DDPM_200',
    'ldm': 'LDM_200',
    'stylegan': 'StyleGAN2_tmp',
}

# Function to apply Laplacian filter to detect edges
def apply_laplacian_filter(image_gray):
    """
    Apply Laplacian filter to detect edges.
    """
    laplacian = cv2.Laplacian(image_gray, cv2.CV_64F)
    return laplacian

# Function to process a directory of images and update the heatmap
def process_directory_images(directory_path):
    """
    Process all images in a directory to create a heatmap based on Laplacian edge detection.
    """
    laplacian_pixel_heatmap = None
    total_images = 0

    for filename in os.listdir(directory_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(directory_path, filename)
            image = cv2.imread(image_path)
            if image is None:
                continue

            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            laplacian_edges = apply_laplacian_filter(image_gray)

            # Convert to edge strength
            edge_strength = np.abs(laplacian_edges)

            if laplacian_pixel_heatmap is None:
                laplacian_pixel_heatmap = edge_strength
            else:
                laplacian_pixel_heatmap += edge_strength

            total_images += 1

    if total_images > 0:
        laplacian_pixel_heatmap /= total_images  # Normalize the heatmap

    return laplacian_pixel_heatmap

# Main function
def main(dataset):
    # Define the directory path that contains the images
    directory_path = f'../data/{datasets[dataset]}'

    # Process the images in the directory using Laplacian edge detection
    heatmap = process_directory_images(directory_path)

    # Plot and save the heatmap
    plt.imshow(heatmap, cmap='hot', interpolation='nearest')
    plt.title('Laplacian Edge Detection Heatmap')
    plt.colorbar()
    plt.savefig(f'laplacian_heatmaps/heatmap_{dataset}.png')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate a heatmap using Laplacian edge detection on images.")

    # Add arguments
    parser.add_argument('--dataset', type=str, choices=['celeb', 'ddim', 'pndm', 'ddpm', 'ldm', 'progan', 'stylegan'], help='Dataset to be processed')

    # Parse the arguments
    args = parser.parse_args()

    main(args.dataset)
