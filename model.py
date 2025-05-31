# model.py

import onnxruntime as ort
import numpy as np
from PIL import Image
from torchvision import transforms
import torch


class Preprocessor:
    def __init__(self, target_size=(224, 224)):
        self.transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def process(self, img: Image.Image) -> np.ndarray:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        tensor = self.transform(img).unsqueeze(0) 
        return tensor.numpy()


class OnnxModel:
    def __init__(self, model_path: str = "assets/model.onnx"):
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict(self, input_array: np.ndarray) -> int:
        outputs = self.session.run([self.output_name], {self.input_name: input_array})
        prediction = np.argmax(outputs[0])
        return int(prediction)


class ImageLoader:
    @staticmethod
    def load(image_path: str) -> Image.Image:
        try:
            return Image.open(image_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load image '{image_path}': {str(e)}")



if __name__ == "__main__":
    preprocessor = Preprocessor()
    model = OnnxModel("assets/model.onnx")
    image = ImageLoader.load("assets/n01667114_mud_turtle.jpeg")

    processed = preprocessor.process(image)
    pred_class = model.predict(processed)

    print(f"Predicted class: {pred_class}")
