import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_high_pass(image):
    """
    Apply high-pass filtering to an image.
    """
    # Apply median blur
    median_blurred = cv2.medianBlur(image, 5)

    # Perform high pass filtering
    return cv2.subtract(image, median_blurred)

def highlight_non_zero_values(image, threshold_value=1):
    """
    Apply high-pass filter, highlight non-zero values with and without thresholding, 
    and return the images.
    """
    # Apply the high-pass filter
    high_pass_image = apply_high_pass(image)

    # Highlight non-zero values
    non_zero_mask = high_pass_image > 0

    # Highlighted non-zero values image
    highlighted_non_zero_image = np.zeros_like(image)
    highlighted_non_zero_image[non_zero_mask] = image[non_zero_mask]
    
    # Apply threshold
    gray_high_pass_image = cv2.cvtColor(high_pass_image, cv2.COLOR_RGB2GRAY)
    _, binary_image = cv2.threshold(gray_high_pass_image, threshold_value, 255, cv2.THRESH_BINARY)
    
    # Highlight thresholded non-zero values
    threshold_mask = binary_image > 0
    highlighted_threshold_image = np.zeros_like(image)
    highlighted_threshold_image[threshold_mask] = image[threshold_mask]

    return high_pass_image, highlighted_non_zero_image, highlighted_threshold_image

# Load the image from the uploaded file
image_path = '../data_sampled/ProGAN/003918.png'  # Replace with the path to your input image
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for matplotlib

# Apply the high-pass filter and get both highlighted images
high_pass_image, highlighted_non_zero_image, highlighted_threshold_image = highlight_non_zero_values(image_rgb)

# Calculate the total number of non-zero pixels for both images
non_zero_count_non_thresholded = np.count_nonzero(highlighted_non_zero_image)
non_zero_count_thresholded = np.count_nonzero(highlighted_threshold_image)

# Display the total amount of non-zero pixels
print(f"Non-zero pixels (Non-Thresholded): {non_zero_count_non_thresholded}")
print(f"Non-zero pixels (Thresholded): {non_zero_count_thresholded}")

# Display the images using matplotlib
plt.figure(figsize=(18, 6))

# High-pass filtered image
plt.subplot(1, 3, 1)
plt.imshow(high_pass_image)
plt.title('High-Pass Filtered Image')
plt.axis('off')

# Highlighted non-zero values image
plt.subplot(1, 3, 2)
plt.imshow(highlighted_non_zero_image)
plt.title('Highlighted Non-Zero Values')
plt.axis('off')

# Highlighted thresholded image
plt.subplot(1, 3, 3)
plt.imshow(highlighted_threshold_image)
plt.title('Highlighted Thresholded Values')
plt.axis('off')

plt.tight_layout()

# Save the figure with the subplots
figure_save_path = 'celeb.png'
plt.savefig(figure_save_path, dpi=300)

# Close the figure to avoid displaying it inline
plt.close()
