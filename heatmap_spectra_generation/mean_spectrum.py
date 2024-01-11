import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from scipy.fft import fft2, dctn
from scipy import ndimage

datasets = {
    'celeb': 'CelebA-HQ-img',
    'ddim': 'DDIM_200',
    'pndm': 'PNDM_200',
    'progan': 'ProGAN',
    'ddpm': 'DDPM_200',
    'ldm': 'LDM_200',
    'stylegan': 'StyleGAN2_tmp',
}

def apply_high_pass_filter(image):
    """
    Apply a high-pass filter by subtracting the median blurred image from the original image.
    """
    median_blurred = cv2.medianBlur(image, 5)
    high_pass_image = cv2.subtract(image, median_blurred)
    return high_pass_image

def apply_low_pass_filter(image):
    """
    Apply a low-pass filter using median blur.
    """
    return cv2.medianBlur(image, 5)

def apply_edge_detection(image):
    """
    Apply Canny edge detection to a grayscale image.
    """
    return cv2.Canny(image, 100, 200)

def sharpen_image(image):
    """
    Apply sharpening to an image using a kernel that enhances edges.
    """
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

def apply_transform(image, transform_type):
    """
    Apply FFT or DCT to the image based on the specified transform type.
    """
    if transform_type == 'fft':
        return np.abs(fft2(image))
    elif transform_type == 'dct':
        return np.abs(dctn(image, type=2, norm='ortho'))
    else:
        raise ValueError("Invalid transform type. Choose 'fft' or 'dct'.")

def process_directory_images(directory_path, filter_type, transform_type):
    """
    Process all images in a directory based on the selected filter and transform type.
    """
    spectrum_sum = None
    total_images = 0

    for filename in os.listdir(directory_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(directory_path, filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            
            # Apply the selected filter
            if filter_type == 'high_pass':
                filtered_image = apply_high_pass_filter(image)
            elif filter_type == 'low_pass':
                filtered_image = apply_low_pass_filter(image)
            elif filter_type == 'edge_detection':
                filtered_image = apply_edge_detection(image)
            elif filter_type == 'sharp_edge_detection':
                sharpened_image = sharpen_image(image)
                filtered_image = apply_edge_detection(sharpened_image)
            else:
                raise ValueError("Invalid filter type. Choose 'high_pass', 'low_pass', 'edge_detection', or 'sharp_edge_detection'.")

            # Apply FFT or DCT
            spectrum = apply_transform(filtered_image, transform_type)

            # Sum up the spectrum
            if spectrum_sum is None:
                spectrum_sum = spectrum
            else:
                spectrum_sum += spectrum

            total_images += 1

    # Calculate mean spectrum
    mean_spectrum = spectrum_sum / total_images

    return mean_spectrum

def main(dataset, filter_type, transform_type):
    # Define the directory path that contains the images
    directory_path = f'../data/{datasets[dataset]}'

    # Process the images in the directory
    mean_spectrum = process_directory_images(directory_path, filter_type, transform_type)

    # Plot and save the mean spectrum
    plt.imshow(np.log1p(mean_spectrum), cmap='hot', interpolation='nearest')
    # plt.title(f'Mean Spectrum ({transform_type.upper()}) with {filter_type.replace('_', ' ').title()}')
    plt.colorbar()
    plt.savefig(f'mean_spectrums/{filter_type}_{transform_type}_spectrum_{dataset}.png')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process images with different filters and generate a mean spectrum using FFT or DCT.")

    # Add arguments
    parser.add_argument('--dataset', type=str, choices=['celeb', 'ddim', 'ddpm', 'ldm', 'pndm', 'progan', 'stylegan'], help='Dataset to be processed')
    parser.add_argument('--filter', type=str, choices=['high_pass', 'low_pass', 'edge_detection', 'sharp_edge_detection'], help='Type of filter to apply')
    parser.add_argument('--transform', type=str, choices=['fft', 'dct'], help='Type of transform to apply: fft or dct')

    # Parse the arguments
    args = parser.parse_args()

    main(args.dataset, args.filter, args.transform)
