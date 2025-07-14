import base64
import requests
from tkinter import Tk, filedialog

# Initialize tkinter (without showing the root window)
Tk().withdraw()

# Prompt user to select the PDF file
pdf_path = filedialog.askopenfilename(title="Select PDF file", filetypes=[("PDF Files", "*.pdf")])
if not pdf_path:
    print("No PDF file selected.")
    exit()

# Prompt user to select the Signature image
sig_path = filedialog.askopenfilename(title="Select Signature Image", filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
if not sig_path:
    print("No signature image selected.")
    exit()

# Read and encode files
with open(pdf_path, "rb") as f:
    pdf_b64 = base64.b64encode(f.read()).decode("utf-8")

with open(sig_path, "rb") as f:
    sig_b64 = base64.b64encode(f.read()).decode("utf-8")

# Prepare and send request
payload = {
    "pdf_base64": pdf_b64,
    "signature_base64": sig_b64
}

response = requests.post("http://localhost:5000/verify-signature", json=payload)
print(response.json())
