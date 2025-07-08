# yolo.py
import cv2
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from pdf2image import convert_from_path
from tkinter import Tk, filedialog
import numpy as np
import torch
import os

from identify import identify_signature

def browse_pdf():
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select PDF document",
        filetypes=[("PDF files", "*.pdf")]
    )
    return file_path

def pdf_to_image(pdf_path):
    pages = convert_from_path(
        pdf_path,
        dpi=300,
        first_page=1,
        last_page=1,
        poppler_path=r"C:\poppler-24.08.0\Library\bin"
    )
    if pages:
        return pages[0]
    else:
        raise ValueError("No pages found in PDF")

def main():
    pdf_path = browse_pdf()
    if not pdf_path:
        print("No file selected. Exiting.")
        return
    print(f"Selected PDF: {pdf_path}")

    pil_image = pdf_to_image(pdf_path)
    cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    # Load YOLOS signature detector
    repo_id = "mdefrance/yolos-base-signature-detection"
    processor = AutoImageProcessor.from_pretrained(repo_id)
    model = AutoModelForObjectDetection.from_pretrained(repo_id)
    model.eval()

    # Prepare input tensor
    inputs = processor(images=pil_image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process to get detections
    target_size = pil_image.size[::-1]  # (height, width)
    detections_raw = processor.post_process_object_detection(
        outputs, threshold=0.5, target_sizes=[target_size]
    )[0]

    if len(detections_raw["scores"]) == 0:
        print("No signatures detected.")
        return

    # Annotate image and show crops
    annotated = cv_image.copy()
    num = len(detections_raw["scores"])
    for idx in range(num):
        box = detections_raw["boxes"][idx].int().tolist()
        x0, y0, x1, y1 = box
        # Crop signature region
        crop = annotated[y0:y1, x0:x1]
        crop_path = f"crop_{idx}.png"
        cv2.imwrite(crop_path, crop)

        # Identify signature
        person, score = identify_signature(crop_path)

        # Draw rectangle and label
        cv2.rectangle(annotated, (x0, y0), (x1, y1), (0, 255, 0), 2)
        label = f"{person} ({score:.2f})"
        cv2.putText(
            annotated,
            label,
            (x0, max(y0 - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
        # Display each crop
        cv2.imshow(f"Signature {idx}: {label}", crop)

    # Show full annotated image scaled down
    h, w = annotated.shape[:2]
    resized = cv2.resize(annotated, (w // 4, h // 4))
    cv2.imshow("All Signatures", resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Cleanup temp crops
    for idx in range(num):
        try:
            os.remove(f"crop_{idx}.png")
        except OSError:
            pass

if __name__ == "__main__":
    main()
