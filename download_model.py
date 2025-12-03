import os
import requests

MODEL_URL = "https://huggingface.co/punitpratapsingh30/siglip-base-onnx/resolve/main/siglip_vision.onnx"
MODEL_PATH = "models/siglip_base.onnx"

def download_model():
    # Skip if already downloaded
    if os.path.exists(MODEL_PATH):
        print("Model already exists. Skipping download.")
        return

    os.makedirs("models", exist_ok=True)
    print("Downloading ONNX model from:", MODEL_URL)

    with requests.get(MODEL_URL, stream=True) as r:
        r.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    print("Model downloaded to:", MODEL_PATH)


if __name__ == "__main__":
    download_model()
