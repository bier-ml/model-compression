from io import BytesIO

import requests
import torch.quantization
from PIL import Image

from all_models.alexnet import AlexNet
from compression.time_benchmark import ExecutionTimeBenchmark

if __name__ == '__main__':
    response = requests.get("https://ultralytics.com/images/zidane.jpg")
    img = Image.open(BytesIO(response.content))
    benchmark = ExecutionTimeBenchmark()

    alexnet = AlexNet()

    model = alexnet.model

    # model.qconfig = torch.quantization.get_default_qconfig('fbgemm')

    quantized_model = torch.ao.quantization.quantize_dynamic(
        model,
        qconfig_spec={
            torch.nn.Linear,
            torch.nn.Conv2d,
            # torch.nn.Conv3d,
            # torch.nn.UpsamplingNearest2d
        },
        dtype=torch.qint8
    )

    param_size = 0
    for param in quantized_model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in quantized_model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    print('model size: {:.10f}MB'.format(alexnet.model_size_mb))
    size_all_mb = (param_size + buffer_size) / 1024.0000 ** 2.0000
    print('quantized_model size: {:.10f}MB'.format(size_all_mb))

    torch.save(model.state_dict(), "yolov5.pth")
    torch.save(quantized_model.state_dict(), "quantized_yolov5.pth")
