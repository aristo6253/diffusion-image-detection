import argparse
import cv2
import numpy as np
import os
from tqdm import tqdm

def apply_high_pass(image):
    """
    Apply high-pass filtering to an image.

    Args:
    image (numpy.ndarray): The input image.

    Returns:
    numpy.ndarray: The high-pass filtered image.
    """
    # Apply median blur
    median_blurred = cv2.medianBlur(image, 5)

    # Perform high pass filtering
    return cv2.subtract(image, median_blurred)

def apply_fft(image):
    """
    Apply Fast Fourier Transform to an image.

    Args:
    image (numpy.ndarray): The input image.

    Returns:
    numpy.ndarray: The FFT transformed image.
    """
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Compute the 2-dimensional FFT
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))

    return magnitude_spectrum

def process_images(input_directory, output_directory, use_high_pass):
    """
    Process images in the input directory and save them to the output directory.

    Args:
    input_directory (str): The path to the directory containing the images.
    output_directory (str): The path to the directory where the processed images will be saved.
    use_high_pass (bool): Whether to apply high-pass filtering before FFT.
    """
    # Check if the input directory exists
    if not os.path.exists(input_directory):
        print("Input directory does not exist.")
        return

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Get all image filenames
    image_filenames = [f for f in os.listdir(input_directory) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]

    # Process each image with a progress bar
    for filename in tqdm(image_filenames, desc="Processing Images"):
        file_path = os.path.join(input_directory, filename)

        # Read the image
        image = cv2.imread(file_path)

        # Apply high pass filter if flag is set
        if use_high_pass:
            image = apply_high_pass(image)

        # Apply FFT
        fft_image = apply_fft(image)

        # Save the processed image
        save_path = os.path.join(output_directory, f"fft_{filename}")
        cv2.imwrite(save_path, fft_image)

# Parsing command line arguments
parser = argparse.ArgumentParser(description="Image Processing with FFT and optional High-Pass Filter")
parser.add_argument("-i", "--input", required=True, help="Input directory containing images")
parser.add_argument("-o", "--output", required=True, help="Output directory for processed images")
parser.add_argument("--high-pass", action="store_true", help="Apply high-pass filter before FFT")

args = parser.parse_args()

# Process images
process_images(args.input, args.output, args.high_pass)
