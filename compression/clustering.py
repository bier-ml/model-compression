import torch

from all_models.alexnet_mnist import MNISTAlexNet

if __name__ == '__main__':
    device = torch.device("cuda")
    alexnet = MNISTAlexNet(num_classes=10).to(device)
    alexnet.load_state_dict(torch.load('base_model.pth'))

    n_clusters = 100
    alexnet.cluster_model_weights(n_clusters)

    clustered_accuracy = alexnet.infer()
    print(f"Number of clusters: {n_clusters}")
    print(f"Clusterized model accuracy: {clustered_accuracy:.3f}%")
