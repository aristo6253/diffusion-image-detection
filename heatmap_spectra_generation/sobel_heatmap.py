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
    'progan': 'ProGAN'
}

# Function to apply Sobel filter to detect edges
def apply_sobel_filter(image_gray):
    """
    Apply Sobel filter to detect horizontal and vertical edges.
    """
    sobel_x = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = cv2.magnitude(sobel_x, sobel_y)
    return sobel_combined

# Function to process a directory of images and update the heatmap
def process_directory_images(directory_path):
    """
    Process all images in a directory to create a heatmap based on Sobel edge detection.
    """
    sobel_pixel_heatmap = None
    total_images = 0

    for filename in os.listdir(directory_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(directory_path, filename)
            image = cv2.imread(image_path)
            if image is None:
                continue

            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            sobel_edges = apply_sobel_filter(image_gray)
            edge_strength = np.sqrt(sobel_edges) # Convert to edge strength

            if sobel_pixel_heatmap is None:
                sobel_pixel_heatmap = edge_strength
            else:
                sobel_pixel_heatmap += edge_strength

            total_images += 1

    if total_images > 0:
        sobel_pixel_heatmap /= total_images  # Normalize the heatmap

    return sobel_pixel_heatmap

# Main function
def main(dataset):
    # Define the directory path that contains the images
    directory_path = f'../data/{datasets[dataset]}'

    # Process the images in the directory using Sobel edge detection
    heatmap = process_directory_images(directory_path)

    # Plot and save the heatmap
    plt.imshow(heatmap, cmap='hot', interpolation='nearest')
    plt.title('Sobel Edge Detection Heatmap')
    plt.colorbar()
    plt.savefig(f'sobel_heatmaps/heatmap_{dataset}.png')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate a heatmap using Sobel edge detection on images.")

    # Add arguments
    parser.add_argument('--dataset', type=str, choices=['celeb', 'ddim', 'pndm', 'progan'], help='Dataset to be processed')

    # Parse the arguments
    args = parser.parse_args()

    main(args.dataset)
