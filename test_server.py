# test_server.py

import argparse
import requests
import base64
import os

# === Required ===
API_URL = "https://YOUR_CEREBRIUM_DEPLOYMENT_URL/predict"  # e.g., https://xyz.cerebrium.ai/predict
API_KEY = "YOUR_CEREBRIUM_API_KEY"                          # Paste your key or load from env

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}


def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()


def predict(image_path: str):
    encoded = encode_image_to_base64(image_path)
    payload = {"image": encoded}

    try:
        res = requests.post(API_URL, headers=HEADERS, json=payload)
        res.raise_for_status()
        result = res.json()
        print(f"✅ Image: {image_path} → Predicted class: {result.get('predicted_class')}")
    except Exception as e:
        print(f"❌ Error for {image_path}: {e}")


def run_preset_tests():
    test_images = {
        "image1.jpg": 0,
        "image2.jpg": 35
    }
    for filename, expected in test_images.items():
        image_path = os.path.join("assets", filename)
        print(f"Running preset test for: {filename} (expected class: {expected})")
        predict(image_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Cerebrium-Deployed Model")
    parser.add_argument("--img", help="Path to image file", type=str)
    parser.add_argument("--preset", action="store_true", help="Run preset tests")
    args = parser.parse_args()

    if args.preset:
        run_preset_tests()
    elif args.img:
        predict(args.img)
    else:
        print("⚠️ Please provide either --img <path> or --preset")
