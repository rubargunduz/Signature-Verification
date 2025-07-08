# identify.py
import cv2
import numpy as np
import pickle
from scipy.spatial.distance import euclidean

def preprocess_image(path, target_size=300):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    
    # Threshold to binary (black signature on white)
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

    # Get current size
    h, w = img.shape[:2]
    
    # Determine padding to make it square
    if h > w:
        pad_left = (h - w) // 2
        pad_right = h - w - pad_left
        pad_top = pad_bottom = 0
    else:
        pad_top = (w - h) // 2
        pad_bottom = w - h - pad_top
        pad_left = pad_right = 0

    # Apply padding
    img_padded = cv2.copyMakeBorder(
        img, pad_top, pad_bottom, pad_left, pad_right,
        borderType=cv2.BORDER_CONSTANT, value=0  # black background
    )

    # Resize to target size
    img_resized = cv2.resize(img_padded, (target_size, target_size))
    
    return img_resized



def extract_features(img):
    moments = cv2.moments(img)
    hu_moments = cv2.HuMoments(moments).flatten()
    return -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)

def identify_signature(test_img_path, database_path="signature_features.pkl", threshold=2.5):
    with open(database_path, "rb") as f:
        db = pickle.load(f)

    test_img = preprocess_image(test_img_path)
    test_feats = extract_features(test_img)

    best_match = None
    best_score = float('inf')
    for person, feats_list in db.items():
        for feats in feats_list:
            score = euclidean(test_feats, feats)
            if score < best_score:
                best_score = score
                best_match = person

    if best_score <= threshold:
        return best_match, best_score
    else:
        return "Unknown", best_score

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    test_img_path = "image.png"  # Replace with a valid path

    who, score = identify_signature(test_img_path)

    print(who, score)

    preprocessed_img = preprocess_image(test_img_path)

    # Show original and preprocessed image
    original = cv2.imread(test_img_path, cv2.IMREAD_GRAYSCALE)
    print(f"Original size: {original.shape}")
    print(f"Preprocessed size: {preprocessed_img.shape}")

    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(original, cmap='gray')

    plt.subplot(1, 2, 2)
    plt.title("Preprocessed (300x300)")
    plt.imshow(preprocessed_img, cmap='gray')

    plt.tight_layout()
    plt.show()
