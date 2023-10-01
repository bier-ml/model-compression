import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


class MNISTAlexNet(nn.Module):

    def __init__(self, num_classes):
        super(MNISTAlexNet, self).__init__()
        self.device = torch.device('cuda')
        alexnet = models.alexnet(pretrained=True)

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
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

        num_epochs = 3
        for epoch in tqdm(range(num_epochs)):
            model.train()
            running_loss = 0.0
            for images, labels in tqdm(train_loader):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images).to(device)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")

        # torch.save(model.state_dict(), 'alexnet_mnist.pth')


if __name__ == '__main__':
    device = torch.device("cuda")
    model = MNISTAlexNet(num_classes=10).to(device)

    model.train_save()
    # model.infer()
