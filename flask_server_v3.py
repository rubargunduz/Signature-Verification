from flask import Flask, request, jsonify
import os
import tempfile
import cv2
import torch
import numpy as np
from PIL import Image, ImageOps
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


def process_file(file_url, temp_dir, prefix):
    response = requests.get(file_url)
    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "")
    if "pdf" in content_type:
        # Handle PDF
        pdf_path = os.path.join(temp_dir, f"{prefix}.pdf")
        with open(pdf_path, "wb") as f:
            f.write(response.content)

        pages = convert_from_path(pdf_path, dpi=300)
        if not pages:
            return []

        crops = []
        for idx, page in enumerate(pages):
            page_cv = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
            page_crops = detect_signature_and_crop(page, page_cv, temp_dir, prefix=f"{prefix}_page_{idx}")
            crops.extend(page_crops)
        return crops
    elif "image" in content_type:
        # Handle Image
        image_path = os.path.join(temp_dir, f"{prefix}.png")
        with open(image_path, "wb") as f:
            f.write(response.content)

        pil_image = Image.open(image_path).convert("RGB")
        w, h = pil_image.size
        pad_w = int(w * 0.25)
        pad_h = int(h * 0.25)
        pil_image_padded = ImageOps.expand(pil_image, border=(pad_w, pad_h, pad_w, pad_h), fill=(255, 255, 255))
        image_cv = cv2.cvtColor(np.array(pil_image_padded), cv2.COLOR_RGB2BGR)
        return detect_signature_and_crop(pil_image_padded, image_cv, temp_dir, prefix=prefix)
    else:
        raise ValueError("Unsupported file type")


@app.route("/verify-signature", methods=["POST"])
def verify_signature():
    data = request.get_json()

    if not data or "pdf_url" not in data or "signature_url" not in data:
        return jsonify({"error": "Missing URL fields"}), 400

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Process PDF or image file from pdf_url
            pdf_crops = process_file(data["pdf_url"], temp_dir, prefix="pdf")
            if not pdf_crops:
                return jsonify({"message": "No signatures found in the provided PDF or image"}), 200

            # Process PDF or image file from signature_url
            signature_crops = process_file(data["signature_url"], temp_dir, prefix="signature")
            if not signature_crops:
                return jsonify({"message": "No signatures found in the provided signature file"}), 200

            # Compare signatures
            found_match = False
            best_score = 0.0
            for sig_crop in signature_crops:
                for pdf_crop in pdf_crops:
                    score = float(match(sig_crop, pdf_crop))
                    if score > 80.0:
                        found_match = True
                        best_score = max(best_score, score)

            if found_match:
                return jsonify({"message": "Signature match found", "score": best_score}), 200
            else:
                return jsonify({"message": "Signatures found, but none matched"}), 200
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 400


if __name__ == "__main__":
    app.run(debug=True)