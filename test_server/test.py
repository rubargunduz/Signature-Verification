import base64
import requests

with open("0285.pdf", "rb") as f:
    pdf_b64 = base64.b64encode(f.read()).decode("utf-8")

with open("signature.jpg", "rb") as f:
    sig_b64 = base64.b64encode(f.read()).decode("utf-8")

payload = {
    "pdf_base64": pdf_b64,
    "signature_base64": sig_b64
}

response = requests.post("http://localhost:5000/verify-signature", json=payload)
print(response.json())
