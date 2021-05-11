import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import random
import numpy as np


# computes total variation for an image
def TV(x):
    batch_size = x.size()[0]
    h_x = x.size()[2]
    w_x = x.size()[3]
    count_h = _tensor_size(x[:, :, 1:, :])
    count_w = _tensor_size(x[:, :, :, 1:])
    h_tv = torch.pow(x[:, :, 1:, :] - x[:, :, :h_x - 1, :], 2).sum()
    w_tv = torch.pow(x[:, :, :, 1:] - x[:, :, :, :w_x - 1], 2).sum()
    return (h_tv / count_h + w_tv / count_w) / batch_size


def l2loss(x):
    return (x ** 2).mean()


def _tensor_size(t):
    return t.size()[1] * t.size()[2] * t.size()[3]


# get random examples from a dataset
def get_random_example(set, count=1, batch_size=1):
    indices = []
    for i in range(count):
        if i not in indices:
            indices.append(random.randrange(len(set)))
    subset = torch.utils.data.Subset(set, indices)
    subsetloader = torch.utils.data.DataLoader(subset, batch_size=batch_size, num_workers=0, shuffle=False)
    return subsetloader


# get the first `count` examples of the `target` class from dataset
def get_examples_by_class(dataset, target, count=1):
    result = []
    for image, label in dataset:
        if label == target:
            if count == 1:
                return image
            result.append(image)
        if len(result) == count:
            break
    return result


def normalize(result):
    min_v = torch.min(result)
    range_v = torch.max(result) - min_v
    if range_v > 0:
        normalized = (result - min_v) / range_v
    else:
        normalized = torch.zeros(result.size())
    return normalized


def get_test_score(m1, m2, dataset, split=0):
    score = 0
    imageloader = get_random_example(dataset, count=2000)
    for image, label in imageloader:
        pred = m2(m1(image, end=split), start=split+1)
        if torch.argmax(pred) == label.detach():
            score += 1
    return 100 * score / len(imageloader)


def display_imagelist(images, height, width):
    fig, ax = plt.subplots(1, len(images))
    for index, image in enumerate(images):
        ax[index].axis('off')
        ax[index].imshow(image.cpu().detach().reshape(height, width))
    plt.show()