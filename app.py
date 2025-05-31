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
    param_1: str 

@app.post("/run")
async def run(req: ImageRequest):
    try:
        image_data = base64.b64decode(req.image)
        image = Image.open(io.BytesIO(image_data))

        input_array = preprocessor.process(image)
        class_id = model.predict(input_array)

        # ✅ return using param_1 format
        return {"param_1": class_id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


