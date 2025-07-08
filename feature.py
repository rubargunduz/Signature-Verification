# feature.py
import cv2
import os
import numpy as np
import pickle

from identify import preprocess_image


def extract_features(img):
    moments = cv2.moments(img)
    hu_moments = cv2.HuMoments(moments).flatten()
    return -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)  # Log transform for stability

def build_signature_database(folder_path):
    db = {}
    for person_name in os.listdir(folder_path):
        person_folder = os.path.join(folder_path, person_name)
        if not os.path.isdir(person_folder):
            continue
        features = []
        for file in os.listdir(person_folder):
            img_path = os.path.join(person_folder, file)
            try:
                img = preprocess_image(img_path)
                feats = extract_features(img)
                features.append(feats)
            except:
                print(f"Error processing {img_path}")
        if features:
            db[person_name] = features
    return db

# Build and save database
signature_db_path = "signature_db"
database = build_signature_database(signature_db_path)

# Save to file for later use
with open("signature_features.pkl", "wb") as f:
    pickle.dump(database, f)

print("Database created with people:", list(database.keys()))
