# identify.py
import cv2
import numpy as np
import pickle
from scipy.spatial.distance import euclidean

def preprocess_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (300, 300))
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    return img


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

