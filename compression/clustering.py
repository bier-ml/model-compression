import numpy as np
import onnx
import tensorflow as tf
import tensorflow_probability as tfp
import torch.quantization
from onnx2pytorch import ConvertModel
from onnx_tf.backend import prepare
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from onnx import numpy_helper

from all_models.alexnet_mnist import MNISTAlexNet

if __name__ == '__main__':
    print("Tensorflow version: ", tf.__version__)
    print("Tensorflow Probability version: ", tfp.__version__)
    device = torch.device("cpu")
    alexnet = MNISTAlexNet(num_classes=10).to(device)
    alexnet.load_state_dict(torch.load('base_model.pth'))

    dummy_input = torch.randn(1, 3, 224, 224)
    print(dummy_input.shape)

    output_file = "alexnet.onnx"  # Define the output ONNX file name

    torch.onnx.export(alexnet, dummy_input, output_file, input_names=['input'],
                      output_names=['output'])

    onnx_model = onnx.load("alexnet.onnx")
    tf_model = prepare(onnx_model)

    # Define the number of clusters
    num_clusters = 2


    # Function to cluster weights of a specific initializer
    def cluster_weights(initializer):
        flattened_w = numpy_helper.to_array(initializer)
        flattened_w = flattened_w.reshape(-1, 1)
        kmeans = KMeans(n_clusters=num_clusters, random_state=0)
        clustered_w = kmeans.fit_predict(flattened_w)
        return clustered_w.astype(np.float32).flatten()


    # Iterate through all initializers in the model and cluster them
    for initializer in onnx_model.graph.initializer:
        if initializer.name.endswith(".weight"):
            clustered_weights = cluster_weights(initializer)
            # initializer.ClearField("float_data")
            tensor = numpy_helper.from_array(clustered_weights.reshape(initializer.dims))
            initializer.CopyFrom(tensor)
            # initializer.float_data.extend(clustered_weights)

    onnx.save(onnx_model, "alexnet_clustered.onnx")

    device = torch.device("cuda")
    onnx_model2 = onnx.load('alexnet_clustered.onnx')
    pytorch_model = ConvertModel(onnx_model2).to(device)

    torch.manual_seed(42)
    batch_size = 1
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    test_dataset = datasets.MNIST(root='../all_models/data', train=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = pytorch_model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100.0 * correct / total

    print(accuracy)
