from flask import Flask, request, jsonify
import os
import tempfile
import base64
import cv2
import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from signature import match
from pdf2image import convert_from_path
import requests


app = Flask(__name__)


# Load YOLO signature detector
repo_id = "mdefrance/yolos-base-signature-detection"
processor = AutoImageProcessor.from_pretrained(repo_id)
model = AutoModelForObjectDetection.from_pretrained(repo_id)
model.eval()


def detect_signature_and_crop(pil_image, full_image_cv, temp_dir, prefix="sig"):
    inputs = processor(images=pil_image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    target_size = pil_image.size[::-1]
    detections = processor.post_process_object_detection(
        outputs, threshold=0.5, target_sizes=[target_size]
    )[0]

    if len(detections["scores"]) == 0:
        return []

    crops = []
    for idx, box in enumerate(detections["boxes"]):
        x0, y0, x1, y1 = box.int().tolist()
        crop = full_image_cv[y0:y1, x0:x1]
        path = os.path.join(temp_dir, f"{prefix}_crop_{idx}.png")
        cv2.imwrite(path, crop)
        crops.append(path)
    return crops




@app.route("/verify-signature", methods=["POST"])
def verify_signature():
    data = request.get_json()

    if not data or "pdf_url" not in data or "signature_url" not in data:
        return jsonify({"error": "Missing URL fields"}), 400

    try:
        pdf_response = requests.get(data["pdf_url"])
        sig_response = requests.get(data["signature_url"])
        pdf_response.raise_for_status()
        sig_response.raise_for_status()
    except Exception as e:
        return jsonify({"error": f"Failed to fetch files: {str(e)}"}), 400

    with tempfile.TemporaryDirectory() as temp_dir:
        # Save fetched PDF and signature
        pdf_path = os.path.join(temp_dir, "input.pdf")
        sig_image_path = os.path.join(temp_dir, "signature_input.png")
        with open(pdf_path, "wb") as f:
            f.write(pdf_response.content)
        with open(sig_image_path, "wb") as f:
            f.write(sig_response.content)

        # Optional signature detection (disabled by default)
        # given_crop = sig_image_path
        # If you want to enable YOLO detection on the given image:
        '''
        given_pil = Image.open(sig_image_path).convert("RGB")
        w, h = given_pil.size
        pad_w = int(w * 0.25)
        pad_h = int(h * 0.25)
        given_pil_padded = ImageOps.expand(given_pil, border=(pad_w, pad_h, pad_w, pad_h), fill=(255, 255, 255))
        given_cv = cv2.cvtColor(np.array(given_pil_padded), cv2.COLOR_RGB2BGR)
        given_crop_paths = detect_signature_and_crop(given_pil_padded, given_cv, temp_dir, prefix="given")
        if not given_crop_paths:
            return jsonify({"message": "No signature detected in the given image"}), 200
        given_crop = given_crop_paths[0]
        '''
        given_crop = sig_image_path

        # Convert PDF to image
        pages = convert_from_path(
            pdf_path,
            dpi=300,
            first_page=1,
            last_page=1
        )
        if not pages:
            return jsonify({"message": "No pages found in PDF"}), 400

        pdf_pil = pages[0]
        pdf_cv = cv2.cvtColor(np.array(pdf_pil), cv2.COLOR_RGB2BGR)
        pdf_crop_paths = detect_signature_and_crop(pdf_pil, pdf_cv, temp_dir, prefix="pdf")

        if not pdf_crop_paths:
            return jsonify({"message": "No signatures found in PDF"}), 200

        # Compare
        found_match = False
        best_score = 0.0
        for crop_path in pdf_crop_paths:
            score = float(match(given_crop, crop_path))
            if score > 80.0:
                found_match = True
                best_score = max(best_score, score)

        if found_match:
            return jsonify({"message": "Signature match found", "score": best_score}), 200
        else:
            return jsonify({"message": "Signatures found, but none matched"}), 200

if __name__ == "__main__":
    app.run(debug=True)
