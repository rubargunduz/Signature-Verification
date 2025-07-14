import base64
import requests

# URLs of the PDF and Signature Image
pdf_url = "https://example.com/sample.pdf"
sig_url = "https://example.com/signature.png"

def download_and_encode(url):
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to download file from {url}")
        exit()
    return base64.b64encode(response.content).decode("utf-8")

# Encode the files
pdf_b64 = download_and_encode(pdf_url)
sig_b64 = download_and_encode(sig_url)

# Prepare and send request
payload = {
    "pdf_base64": pdf_b64,
    "signature_base64": sig_b64
}

response = requests.post("http://localhost:5000/verify-signature", json=payload)
print(response.json())
