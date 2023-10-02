import time

from all_models.abstract_model import BaseModel
from all_models.yolov5 import YoloV5


class ExecutionTimeBenchmark:
    @staticmethod
    def run(model: BaseModel, samples: int = 10) -> float:
        start_time = time.time()

        for _ in range(samples):
            model.sample_infer()

        return (time.time() - start_time) / samples


if __name__ == '__main__':
    benchmark = ExecutionTimeBenchmark()
    yolo = YoloV5()
    execution_time = benchmark.run(yolo)
    print(f"Average inference time is {execution_time}")
