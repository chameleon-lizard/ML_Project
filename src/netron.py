import netron
from nn import UNet, ResNet_D
import torch
import numpy as np
import os

def plot_network_struct():
    resnet = ResNet_D(16, nc=3).eval()
    unet = UNet(3, 3).eval()
    x = torch.Tensor(np.random.normal(size=(16, 3, 16, 16, 16)))

    resnet_fig = os.path.join('schemes',  'ResNet.onnx')
    torch.onnx.export(resnet, x, resnet_fig, input_names=['input'], output_names=['output'], opset_version=10)
    unet_fig = os.path.join('schemes',  'UNet.onnx')
    torch.onnx.export(unet, x, unet_fig, input_names=['input'], output_names=['output'], opset_version=10)

if __name__ == "__main__":
    plot_network_struct()

