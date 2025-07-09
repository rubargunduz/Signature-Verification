import cv2
import numpy as np
from scipy.spatial.distance import pdist

# Paths
INPUT_IMAGE_PATH = 'assets/signature.jpg'
OUTPUT_IMAGE_PATH = 'signature_cc_removed.png'

# Load image
image = cv2.imread(INPUT_IMAGE_PATH)
if image is None:
    print("Failed to load image.")
else:
    h, w = image.shape[:2]
    thresh_dist = 0.5 * ((w + h) / 2)  # threshold

    # Convert to grayscale and binary mask (non-white pixels)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = (gray < 128).astype(np.uint8) * 255

    # Connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    # Prepare mask for final output: start with all pixels white
    output_mask = np.zeros_like(binary)

    for label in range(1, num_labels):  # skip background label 0
        # Get pixel coordinates for this component
        ys, xs = np.where(labels == label)
        points = np.column_stack((xs, ys))

        if len(points) < 2:
            # Single pixel component: remove it
            continue

        # Calculate pairwise distances
        distances = pdist(points)
        max_dist = distances.max()

        # Keep component only if max_dist >= threshold
        if max_dist >= thresh_dist:
            output_mask[labels == label] = 255

    # Create white background image
    white_bg = np.full_like(image, 255)

    # Apply mask: keep original pixels where output_mask=255, else white
    final = np.where(output_mask[:, :, None] == 255, image, white_bg)

    # Save and show
    cv2.imwrite(OUTPUT_IMAGE_PATH, final)
    print(f"Saved connected components cleaned image to: {OUTPUT_IMAGE_PATH}")

    cv2.imshow("Original", image)
    cv2.imshow("CC Cleaned", final)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
