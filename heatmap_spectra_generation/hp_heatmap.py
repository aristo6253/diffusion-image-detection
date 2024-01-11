import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse


datasets = {
    'celeb': 'CelebA-HQ-img',
    'ddim': 'DDIM_200',
    'pndm': 'PNDM_200',
    'progan': 'ProGAN',
    'ddpm': 'DDPM_200',
    'ldm': 'LDM_200',
    'stylegan': 'StyleGAN2_tmp',
}

def apply_high_pass(image):
    """
    Apply high-pass filtering to an image.
    """
    median_blurred = cv2.medianBlur(image, 5)
    return cv2.subtract(image, median_blurred)

def process_directory_images(directory_path, threshold_value=1, thresholding_enabled=True):
    """
    Process all images in a directory to create a heatmap.
    If thresholding_enabled is True, use the thresholded approach; otherwise, use non-thresholded.
    """
    non_zero_pixel_heatmap = None
    total_images = 0

    for filename in os.listdir(directory_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(directory_path, filename)
            image = cv2.imread(image_path)
            if image is None:
                continue
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            high_pass_image = apply_high_pass(image_gray)

            if thresholding_enabled:
                # Thresholded approach
                _, binary_image = cv2.threshold(high_pass_image, threshold_value, 255, cv2.THRESH_BINARY)
                non_zero_positions = (binary_image > 0).astype(int)
            else:
                # Non-thresholded approach
                # Normalize the high-pass image to range 0-1
                norm_image = cv2.normalize(high_pass_image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                non_zero_positions = norm_image

            # Update heatmap
            if non_zero_pixel_heatmap is None:
                non_zero_pixel_heatmap = non_zero_positions
            else:
                non_zero_pixel_heatmap += non_zero_positions

            total_images += 1

    # Normalize heatmap by the number of images
    non_zero_pixel_heatmap = non_zero_pixel_heatmap / total_images

    return non_zero_pixel_heatmap

def main(dataset, threshold, no_thresh):

    # Define the directory path that contains the images
    directory_path = f'../data/{datasets[dataset]}'

    # Process the images in the directory with thresholding enabled
    heatmap = process_directory_images(directory_path, threshold_value=threshold, thresholding_enabled=not no_thresh)

    suffix = '_' + dataset
    if no_thresh:
        suffix += '_nt'
    else:
        suffix += '_t' + str(threshold)
    # Plot and save the thresholded heatmap
    plt.imshow(heatmap, cmap='hot', interpolation='nearest')
    plt.title(no_thresh*'Non-'+'Thresholded Heatmap of Non-Zero Pixel Frequency')
    plt.colorbar()
    plt.savefig(f'highpass_heatmaps/heatmap{suffix}.png')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")

    # Add arguments
    parser.add_argument('--thresh_val', type=int, default=1, help='Threshold used for binarization')
    parser.add_argument('--dataset', type=str, choices=['celeb', 'ddim', 'pndm', 'ddpm', 'ldm', 'progan', 'stylegan'], help='Dataset to be processed')
    parser.add_argument('--no_thresh', action='store_true', help='Opt to not use a threshold')

    # Parse the arguments
    args = parser.parse_args()

    main(args.dataset, args.thresh_val, args.no_thresh)


