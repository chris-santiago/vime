import torch
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, transforms

from vime import Constants


def get_mnist_datasets():
    flatten = Compose([transforms.ToTensor(), transforms.Lambda(torch.flatten)])
    train = MNIST(Constants.DATA, train=True, download=True, transform=flatten)
    test = MNIST(
        Constants.DATA, train=False, download=True, transform=transforms.ToTensor()
    )
    return train, test
