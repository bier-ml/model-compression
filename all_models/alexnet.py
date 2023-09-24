from io import BytesIO
from typing import Any, Optional

import pandas as pd
import requests
import torch
from torchvision import transforms
from PIL import Image

from all_models import CHECKPOINTS_DIR
from all_models.abstract_model import BaseModel


class AlexNet(BaseModel):
    def __init__(self):
        self.model = self.init_model()

    @property
    def parameters_number(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    @property
    def model_size_mb(self) -> int:
        param_size, buffer_size = 0, 0
        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()

        for buffer in self.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_all_mb = (param_size + buffer_size) / 1024 ** 2
        return size_all_mb

    @staticmethod
    def init_model() -> Any:
        torch.hub.set_dir(CHECKPOINTS_DIR)
        model = torch.hub.load(
            'pytorch/vision:v0.10.0',
            'alexnet', pretrained=True
        )
        model.eval()
        return model

    def sample_infer(self) -> Any:
        return self.infer()

    def infer(self, images: Optional[list[Image]] = None) -> pd.DataFrame:
        res = requests.get("https://github.com/pytorch/hub/raw/master/images/dog.jpg")
        img = Image.open(BytesIO(res.content))

        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(img)
        input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
            self.model.to('cuda')

        with torch.no_grad():
            output = self.model(input_batch)

        return output  # results.pandas().xyxy[0]


if __name__ == '__main__':
    model = AlexNet()
    print(model.parameters_number)
    print(model.infer())
