import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image from the correct path
image_path = '../data/CelebA-HQ-img/1002.jpg'
image = cv2.imread(image_path)

# Prepare the output path
output_path = './celeb_viz.png'

# Check if the image was loaded properly
if image is None:
    print("Error: Image not loaded.")
else:
    # Apply high-pass filtering
    median_blurred = cv2.medianBlur(image, 5)
    high_pass_filtered = cv2.subtract(image, median_blurred)

    # Apply low-pass filtering
    low_pass_filtered = cv2.medianBlur(image, 5)

    # Apply edge detection using Canny
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)

    # Apply histogram equalization
    gray_hist_eq = cv2.equalizeHist(gray)

    # Apply sharpening
    kernel_sharpening = np.array([[-1, -1, -1],
                                  [-1, 9, -1],
                                  [-1, -1, -1]])
    sharpened = cv2.filter2D(image, -1, kernel_sharpening)

    # Convert to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Erosion
    kernel = np.ones((5, 5), np.uint8)
    eroded = cv2.erode(image, kernel, iterations=1)

    # Dilation
    dilated = cv2.dilate(image, kernel, iterations=1)

    # Make a figure with subplots
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle('Image Pre-processing Techniques')

    # Original image
    axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    # High-pass filtered image
    axes[0, 1].imshow(cv2.cvtColor(high_pass_filtered, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title('High-Pass Filtered')
    axes[0, 1].axis('off')

    # Low-pass filtered image
    axes[0, 2].imshow(cv2.cvtColor(low_pass_filtered, cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title('Low-Pass Filtered')
    axes[0, 2].axis('off')

    # Edge Detection
    axes[1, 0].imshow(edges, cmap='gray')
    axes[1, 0].set_title('Edge Detection (Canny)')
    axes[1, 0].axis('off')

    # Histogram Equalization
    axes[1, 1].imshow(gray_hist_eq, cmap='gray')
    axes[1, 1].set_title('Histogram Equalization')
    axes[1, 1].axis('off')

    # Sharpened image
    axes[1, 2].imshow(cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB))
    axes[1, 2].set_title('Sharpened Image')
    axes[1, 2].axis('off')

    # HSV image
    axes[2, 0].imshow(cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB))
    axes[2, 0].set_title('HSV Color Space')
    axes[2, 0].axis('off')

    # Eroded image
    axes[2, 1].imshow(cv2.cvtColor(eroded, cv2.COLOR_BGR2RGB))
    axes[2, 1].set_title('Eroded Image')
    axes[2, 1].axis('off')

    # Dilated image
    axes[2, 2].imshow(cv2.cvtColor(dilated, cv2.COLOR_BGR2RGB))
    axes[2, 2].set_title('Dilated Image')
    axes[2, 2].axis('off')

    # Save the figure
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Figure saved as {output_path}")

