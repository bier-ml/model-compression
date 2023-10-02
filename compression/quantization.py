import torch.quantization

from all_models.alexnet_mnist import MNISTAlexNet

if __name__ == '__main__':
    device = torch.device("cpu")
    model = MNISTAlexNet(num_classes=10).to(device)
    model.load_state_dict(torch.load('../all_models/alexnet_mnist.pth'))
    model.eval()

    # model.qconfig = torch.quantization.get_default_qconfig('fbgemm')

    quantized_model = torch.ao.quantization.quantize_dynamic(
        model,
        qconfig_spec={
            torch.nn.Linear
        },
        dtype=torch.qint8
    )

    print('Base model size: {:.10f}MB'.format(model.model_size_mb()))
    print('Quantized model size: {:.10f}MB'.format(quantized_model.model_size_mb()))

    base_accuracy = model.infer()
    print(f"Base model (test accuracy): {base_accuracy:.5f}%")

    quantized_accuracy = quantized_model.infer()
    print(f"Quantized model (test accuracy): {quantized_accuracy:.5f}%")

    # torch.save(model.state_dict(), "base_model.pth")
    # torch.save(quantized_model.state_dict(), "quantized_model.pth")
