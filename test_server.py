# test_server.py

import argparse
import requests
import base64
import os

# === Cerebrium Deployment Configuration ===
API_URL = "https://api.cortex.cerebrium.ai/v4/p-704ad0ee/mtailor-takehometask/run"
API_KEY = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJwcm9qZWN0SWQiOiJwLTcwNGFkMGVlIiwibmFtZSI6IiIsImRlc2NyaXB0aW9uIjoiIiwiZXhwIjoyMDY0MjYxNDI5fQ.eKz-LMb3xQVzrzIjsXsklHwhWa723gs8wE11FxQChGFRC4-WWSjgK7H2BJv3uddlaz0StUkHvORmwaPXVNkXAdvuCv__ELr9IcGTkupJ0Gwo0Dsu-NaAfAfpWQIx4Ev6uWMT5zkXUQUGcypIFyCZK2cweaHofqb2SgkTLwKrdgwv770jGrKRyjsBt78ercTLOoNbEpEDt_8CZJbqqOeln-Gsg52B9a1YdE6pdX4hZuFvO0DS8Uk8UWVCa0JZ0BSYMOi18VUPO6s4DMYiOlCYJpiwFGyJYIiICDjuZbQrvvL__Q1uylHFa-Byez33vR6qTd_yJEKnjvupQ7lM7RVxBw"

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
        return res.json()
    except Exception as e:
        print(f"❌ Error for {image_path}: {e}")
        return None


# def run_preset_tests():
def run_preset_tests():
    test_images = {
        "n01440764_tench.jpeg": 0,
        "n01667114_mud_turtle.JPEG": 35
    }
    for filename, expected in test_images.items():
        image_path = os.path.join("assets", filename)
        print(f"Running preset test for: {filename} (expected class: {expected})")
        result = predict(image_path)
        if result:
            print(f"✅ Image: {image_path} → Predicted class: {result.get('param_1')}")


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
