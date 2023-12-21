import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def apply_high_pass(image):
    """
    Apply high-pass filtering to an image.
    """
    median_blurred = cv2.medianBlur(image, 5)
    return cv2.subtract(image, median_blurred)

def process_directory_images(directory_path, threshold_value=1):
    """
    Process all images in a directory to find the average number of non-zero pixels
    and create a heatmap of non-zero pixel frequency.
    """
    non_zero_counts_non_thresholded = []
    non_zero_counts_thresholded = []
    non_zero_pixel_heatmap = None
    total_images = 0

    for filename in os.listdir(directory_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Add other file types if necessary
            image_path = os.path.join(directory_path, filename)
            image = cv2.imread(image_path)
            if image is None:
                continue  # Skip files that are not images
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            high_pass_image = apply_high_pass(image_gray)

            # Non-thresholded
            non_zero_mask_non_thresholded = high_pass_image > 0
            non_zero_counts_non_thresholded.append(np.count_nonzero(non_zero_mask_non_thresholded))

            # Thresholded
            _, binary_image = cv2.threshold(high_pass_image, threshold_value, 255, cv2.THRESH_BINARY)
            threshold_mask = binary_image > 0
            non_zero_counts_thresholded.append(np.count_nonzero(threshold_mask))

            # Update heatmap
            non_zero_positions = threshold_mask.astype(int)
            if non_zero_pixel_heatmap is None:
                non_zero_pixel_heatmap = non_zero_positions
            else:
                non_zero_pixel_heatmap += non_zero_positions

            total_images += 1

    # Convert counts to averages
    avg_non_zero_non_thresholded = np.mean(non_zero_counts_non_thresholded)
    avg_percentage_non_thresholded = (avg_non_zero_non_thresholded / (image_gray.size)) * 100

    avg_non_zero_thresholded = np.mean(non_zero_counts_thresholded)
    avg_percentage_thresholded = (avg_non_zero_thresholded / (image_gray.size)) * 100

    # Normalize heatmap by the number of images
    non_zero_pixel_heatmap = non_zero_pixel_heatmap / total_images

    return avg_non_zero_non_thresholded, avg_percentage_non_thresholded, avg_non_zero_thresholded, avg_percentage_thresholded, non_zero_pixel_heatmap

datasets = ['CelebA-HQ-img', 'DDIM_200', 'PNDM_200', 'ProGAN']

# Define the directory path that contains the images
directory_path = f'../data/{datasets[3]}'  # Replace with the path to your directory

# Process the images in the directory
results = process_directory_images(directory_path)

print(f"Average Non-Zero Pixels (Non-Thresholded): {results[0]}")
print(f"Average Percentage of Non-Zero Pixels (Non-Thresholded): {results[1]}%")
print(f"Average Non-Zero Pixels (Thresholded): {results[2]}")
print(f"Average Percentage of Non-Zero Pixels (Thresholded): {results[3]}%")

# Plot the heatmap
plt.imshow(results[4], cmap='hot', interpolation='nearest')
plt.title('Heatmap of Non-Zero Pixel Frequency')
plt.colorbar()
plt.show()

# Optionally save the heatmap to a file
heatmap_save_path = 'highpass_heatmaps_total/heatmap_progan-t1.png'  # Replace with the path where you want to save the heatmap
plt.savefig(heatmap_save_path)
