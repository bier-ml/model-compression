import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


class MNISTAlexNet(nn.Module):

    def __init__(self, num_classes):
        super(MNISTAlexNet, self).__init__()
        self.device = torch.device('cuda')
        alexnet = models.alexnet()

        for param in alexnet.features.parameters():
            param.requires_grad = False

        self.features = alexnet.features

        self.classifier = nn.Sequential(
            nn.Linear(256 * 6 * 6, 100),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(100, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def infer(self):
        torch.manual_seed(42)
        batch_size = 32
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
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100.0 * correct / total
        return accuracy

    def model_size_mb(self):
        param_size, buffer_size = 0, 0
        for param in self.parameters():
            param_size += param.nelement() * param.element_size()

        for buffer in self.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_all_mb = (param_size + buffer_size) / 1024 ** 2
        return size_all_mb

    def train_save(self):
        torch.manual_seed(42)
        batch_size = 32

        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)

        num_epochs = 3
        for epoch in tqdm(range(num_epochs)):
            self.train()
            running_loss = 0.0
            for images, labels in tqdm(train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self(images).to(self.device)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")

        torch.save(self.state_dict(), 'alexnet_mnist.pth')

    def cluster_model_weights(self, n_clusters=256):
        # Clustering Model Weights
        for name, module in self.named_modules():
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                original_weights = module.weight.data.cpu().numpy()
                reshaped_weights = original_weights.reshape(-1, 1)  # collapse to 1D
                # Fit k-means
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=3, max_iter=10).fit(reshaped_weights)
                # Replace weights with cluster center values
                cluster_labels = kmeans.labels_
                new_weights = kmeans.cluster_centers_[cluster_labels]
                module.weight.data = torch.from_numpy(new_weights.reshape(original_weights.shape)).to(self.device)

    def knowledge_distillation(self, teacher_model, temperature=3.0, alpha=0.5):
        self.teacher = teacher_model
        self.temperature = temperature
        self.alpha = alpha

        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        batch_size = 32

        train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)

        num_epochs = 3
        for epoch in tqdm(range(num_epochs)):
            self.train()
            running_loss = 0.0
            for images, labels in tqdm(train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                student_outputs = self(images)

                with torch.no_grad():
                    teacher_outputs = self.teacher(images)

                loss = (1 - self.alpha) * F.cross_entropy(student_outputs, labels)
                loss += self.alpha * (self.temperature ** 2) * F.kl_div(
                    F.log_softmax(student_outputs / self.temperature, dim=-1),
                    F.softmax(teacher_outputs / self.temperature, dim=-1))
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")

        torch.save(self.state_dict(), 'distillated.pth')


if __name__ == '__main__':
    device = torch.device("cuda")
    # model = MNISTAlexNet(num_classes=10).to(device)

    teacher_model = MNISTAlexNet(num_classes=10).to(device)
    # teacher_model.train_save()
    teacher_model.load_state_dict(torch.load('alexnet_mnist.pth'))


    student_model = MNISTAlexNet(num_classes=10).to(device)
    student_model.knowledge_distillation(teacher_model)

    # student_model.load_state_dict(torch.load('distillated.pth'))
    # teacher_model.infer()
    # student_model.infer()
