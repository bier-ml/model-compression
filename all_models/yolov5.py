from typing import Any, Optional

import pandas as pd
import torch
from PIL import Image

from all_models import CHECKPOINTS_DIR
from all_models.abstract_model import BaseModel


class YoloV5(BaseModel):
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
            'ultralytics/yolov5:master',
            'custom', CHECKPOINTS_DIR / 'yolov5s'
        )
        model.eval()
        return model

    def sample_infer(self) -> Any:
        return self.infer()

    def infer(self, images: Optional[list[Image]] = None) -> pd.DataFrame:
        imgs = ['https://ultralytics.com/images/zidane.jpg']
        results = self.model(imgs)
        return results.pandas().xyxy[0]


if __name__ == '__main__':
    yolo = YoloV5()
    print(yolo.parameters_number)
    print(yolo.infer())
