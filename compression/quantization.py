from io import BytesIO

import requests
import torch
from PIL import Image

from all_models.yolov5 import YoloV5
from compression.time_benchmark import ExecutionTimeBenchmark

if __name__ == '__main__':
    response = requests.get("https://ultralytics.com/images/zidane.jpg")
    img = Image.open(BytesIO(response.content))
    benchmark = ExecutionTimeBenchmark()
    yolo = YoloV5()

    #  ===================================  #
    #  Quantization example should be here  #
    #  ===================================  #
    # yolo_int8 = magic(yolo)
    # print(benchmark.run(yolo), benchmark.run(yolo_int8))
