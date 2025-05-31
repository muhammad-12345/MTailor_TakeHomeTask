# app.py

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import base64
import io
from PIL import Image
from model import Preprocessor, OnnxModel

# Initialize FastAPI app
app = FastAPI(title="MTailor Image Classifier API")

# Load model and preprocessor once at startup
model = OnnxModel("assets/model.onnx")
preprocessor = Preprocessor()

# Define input data model
class ImageRequest(BaseModel):
    image: str  # base64 encoded image

@app.post("/predict")
async def predict(req: ImageRequest):
    try:
        # Decode base64 string to bytes
        image_data = base64.b64decode(req.image)
        image = Image.open(io.BytesIO(image_data))

        # Preprocess image
        input_array = preprocessor.process(image)

        # Run inference
        class_id = model.predict(input_array)

        return {"predicted_class": class_id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
