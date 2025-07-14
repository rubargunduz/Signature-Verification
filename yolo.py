import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection
import cv2
import numpy as np

# Load YOLOS signature detection model
repo_id = "mdefrance/yolos-base-signature-detection"
processor = AutoImageProcessor.from_pretrained(repo_id)
model = AutoModelForObjectDetection.from_pretrained(repo_id)
model.eval()

def detect_and_show(image_path):
    # Load image and convert to NumPy
    pil_image = Image.open(image_path).convert("RGB")
    np_image = np.array(pil_image)

    # Calculate 25% padding
    h, w, _ = np_image.shape
    pad_top = pad_bottom = int(h * 0.25)
    pad_left = pad_right = int(w * 0.25)

    # Apply padding
    padded_np = cv2.copyMakeBorder(np_image, pad_top, pad_bottom, pad_left, pad_right,
                                   borderType=cv2.BORDER_CONSTANT, value=[255, 255, 255])

    # Convert back to PIL
    padded_pil = Image.fromarray(padded_np)
    padded_cv = cv2.cvtColor(padded_np, cv2.COLOR_RGB2BGR)

    # Run YOLOS detection
    inputs = processor(images=padded_pil, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    target_size = padded_pil.size[::-1]  # (H, W)
    results = processor.post_process_object_detection(
        outputs, threshold=0.5, target_sizes=[target_size]
    )[0]

    boxes = results["boxes"]
    scores = results["scores"]
    labels = results["labels"]

    print(f"\n{len(boxes)} signature(s) found:")
    for idx, (box, score) in enumerate(zip(boxes, scores), 1):
        x0, y0, x1, y1 = map(int, box.tolist())
        print(f" - Box {idx}: (x0={x0}, y0={y0}, x1={x1}, y1={y1}), score={score:.2f}")
        cv2.rectangle(padded_cv, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.putText(padded_cv, f"{score:.2f}", (x0, y0 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show image
    cv2.imshow("YOLO Signature Detection with Padding", padded_cv)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python debug_signature_detection.py <image_path>")
    else:
        detect_and_show(sys.argv[1])
