# convert_to_onnx.py

import torch
from torchvision import transforms
from PIL import Image
from pytorch_model import Classifier, BasicBlock

if __name__ == "__main__":
    model = Classifier(BasicBlock, [2, 2, 2, 2])
    model.load_state_dict(torch.load("assets/pytorch_model_weights.pth", map_location=torch.device('cpu')))
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)

    torch.onnx.export(
        model,
        dummy_input,
        "assets/model.onnx",
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        },
        opset_version=11
    )

    print("✅ ONNX model exported to assets/model.onnx")
