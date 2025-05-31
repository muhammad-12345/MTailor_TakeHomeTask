# MTailor_TakeHomeTask

# Image Classification Model Deployment (ONNX + Cerebrium)

This repository contains all the components to deploy an image classification model (trained on ImageNet) using ONNX and serve it via Cerebrium’s serverless GPU platform using a custom Docker image.

---

## Project Overview

- ✅ Convert a trained **PyTorch model** to **ONNX**
- ✅ Build an **inference API** using **FastAPI**
- ✅ Package and deploy the service using a **custom Docker image**
- ✅ Host the model on **Cerebrium GPU serverless platform**
- ✅ Include **local and remote tests** using provided sample images


---

## 🔧 Environment Setup (Local)

### 1. Clone this Repo
git clone https://github.com/YOUR_USERNAME/mtailor-cerebrium-classifier.git
cd mtailor-cerebrium-classifier

###2. Create Python Virtual Environment
python3 -m venv env
source env/bin/activate

###3. Install Dependencies
pip install --upgrade pip
pip install -r requirements.txt

###4. Download Model Weights
https://www.dropbox.com/s/b7641ryzmkceoc9/pytorch_model_weights.pth?dl=0

python convert_to_onnx.py

Input: PyTorch weights (assets/pytorch_model_weights.pth)

Output: assets/model.onnx

ONNX model will be used in deployment — do not deploy PyTorch model directly.

### Run local Test
python test.py

This validates:

ONNX model loads

Image preprocessing is applied correctly

Inference returns valid class IDs

Test images are correctly predicted (0 and 35)

### Start Local API Server
python app.py

###Docker BUILD
docker build -t classifier-api .

###Deploy to Cerebrium
1. Sign in to: https://www.cerebrium.ai/
Use free credits — no card required

Connect GitHub repo 

2. Create a New Deployment
Select Docker-based

Use default Dockerfile

Set entrypoint: app.py

Set project type: Python

3. Deploy
Wait for successful logs

Copy API endpoint URL

Generate API key (required for testing)

###Test Deployment via test_server.py
Run Inference on Deployed Model
python test_server.py --img assets/image1.jpg

This:
Reads the image
Converts it to base64
Sends it to Cerebrium API with API key
Returns predicted class ID

###Where to Add API Key and Endpoint
Inside test_server.py:
API_KEY = "your_cerebrium_api_key"
URL = "https://your-cerebrium-url/predict"

###A Short Description of what each file does:

app.py
This file implements a FastAPI server that exposes a /predict endpoint. It accepts a base64-encoded image, decodes and preprocesses it, then performs inference using the ONNX model and returns the predicted class ID as JSON.

convert_to_onnx.py
This script loads the PyTorch Classifier model from pytorch_model.py, loads its weights, and converts it to ONNX format using a dummy input. The resulting ONNX model is saved as assets/model.onnx.

model.py
This contains three modular classes: Preprocessor for handling image resizing and normalization, OnnxModel for loading the ONNX model and running inference using ONNX Runtime, and Imageloader for loading images from disk. These classes are reused across the entire project.

test.py
This file includes unit tests for the local inference pipeline. It checks that image loading works, preprocessing outputs the correct shape, and predictions fall within valid class ID ranges. It uses Python's built-in unittest framework.

test_server.py
This script is used to test the deployed model on Cerebrium. It reads an image, encodes it in base64, and sends it via POST request to the /predict endpoint of your deployed API.

Dockerfile
This file defines how to build a Docker image for the project. It installs all dependencies, copies your codebase, and sets up the container to run the FastAPI server. This Docker image is deployed on Cerebrium.

requirements.txt
This lists all Python dependencies needed for the project, such as FastAPI, onnxruntime, Pillow, NumPy, etc. It is used both locally and in the Docker image to install necessary packages.

pytorch_model.py
This is the provided PyTorch model definition file. It includes the architecture of the Classifier model and the preprocess_numpy method for image preprocessing. It is used only in the model conversion step (convert_to_onnx.py) and not for inference or deployment.

pytorch_model_weights.pth: The pretrained PyTorch model weights.
model.onnx: The exported ONNX model used in actual inference and deployment.