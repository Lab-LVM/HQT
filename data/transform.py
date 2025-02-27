import torch
import numpy as np
import random
import torchvision.transforms as transforms
from PIL import ImageFilter


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def encode_onehot(labels, num_classes=10):
    """
    one-hot labels

    Args:
        labels (numpy.ndarray): labels.
        num_classes (int): Number of classes.

    Returns:
        onehot_labels (numpy.ndarray): one-hot labels.
    """
    onehot_labels = np.zeros((len(labels), num_classes))

    for i in range(len(labels)):
        onehot_labels[i, labels[i]] = 1

    return onehot_labels

class Onehot(object):
    def __call__(self, sample, num_classes=10):
        target_onehot = torch.zeros(num_classes)
        target_onehot[sample] = 1

        return target_onehot

def train_transforms(mean, std, img_size):
    transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    return transform

def train_aug_transforms(mean, std, img_size, scale):
    color_jitter = transforms.ColorJitter(0.4,0.4,0.4,0.1)
    transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=scale),
        transforms.RandomApply([color_jitter], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    return transform

def test_transforms(mean, std, img_size):
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    return transform

