import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from all_models.alexnet import AlexNet


def print_sparsity(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            sparsity = torch.sum(module.weight == 0) / module.weight.nelement() * 100
            print(f"Sparsity of {name}.weight: {sparsity:.2f}%")


if __name__ == '__main__':

    # Instantiate your model
    alex = AlexNet()
    model = alex.model

    # Print the model's initial sparsity (before pruning)

    print("Initial sparsity:")
    print_sparsity(model)

    # Add pruning to the model
    for module in model.modules():
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            prune.l1_unstructured(module, name="weight", amount=0.2)  # Prunes 20% of weights

    # Remove pruned weights
    for module in model.modules():
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            prune.remove(module, "weight")

    # Print the model's sparsity after pruning
    print("\nSparsity after pruning:")
    print_sparsity(model)
