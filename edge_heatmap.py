import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse


datasets = {
    'celeb': 'CelebA-HQ-img',
    'ddim': 'DDIM_200',
    'pndm': 'PNDM_200',
    'progan': 'ProGAN'
}

def apply_edge_detection(image_gray):
    """
    Apply Canny edge detection to a grayscale image.
    """
    # Use Canny edge detection
    edges = cv2.Canny(image_gray, 100, 200)
    return edges

def process_directory_images(directory_path):
    """
    Process all images in a directory to create a heatmap based on edge detection.
    """
    edge_pixel_heatmap = None
    total_images = 0

    for filename in os.listdir(directory_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(directory_path, filename)
            image = cv2.imread(image_path)
            if image is None:
                continue
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            edges = apply_edge_detection(image_gray)
            edge_positions = (edges > 0).astype(int)

            # Update heatmap
            if edge_pixel_heatmap is None:
                edge_pixel_heatmap = edge_positions
            else:
                edge_pixel_heatmap += edge_positions

            total_images += 1

    # Normalize heatmap by the number of images
    edge_pixel_heatmap = edge_pixel_heatmap / total_images

    return edge_pixel_heatmap

def main(dataset):

    # Define the directory path that contains the images
    directory_path = f'../data/{datasets[dataset]}'

    # Process the images in the directory using edge detection
    heatmap = process_directory_images(directory_path)

    # Plot and save the heatmap
    plt.imshow(heatmap, cmap='hot', interpolation='nearest')
    plt.title('Edge Detection Heatmap of Pixel Frequency')
    plt.colorbar()
    plt.savefig(f'edge_heatmaps/heatmap_{dataset}.png')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process images and generate a heatmap using edge detection.")

    # Add arguments
    parser.add_argument('--dataset', type=str, choices=['celeb', 'ddim', 'pndm', 'progan'], help='Dataset to be processed')

    # Parse the arguments
    args = parser.parse_args()

    main(args.dataset)
