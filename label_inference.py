import sys

import numpy as np
import torch
from torchvision import transforms, datasets
from torchvision.utils import save_image

import attacks
from models import *
from util import *


dataset = sys.argv[1]
count = int(sys.argv[2])

# load datasets and initialize client, server, and clone models
# set the split values so that the client model (second part) has depth of one.
if dataset == 'mnist':
    trainset = datasets.MNIST('data/mnist', download=True, train=True, transform=transforms.ToTensor())
    testset = datasets.MNIST('data/mnist', download=True, train=False, transform=transforms.ToTensor())
    client, server, clone = MnistNet(), MnistNet(), MnistNet()
    split_layer = 9
    grad_index = 8
elif dataset == 'f_mnist':
    trainset = datasets.FashionMNIST('data/f_mnist', download=True, train=True, transform=transforms.ToTensor())
    testset = datasets.FashionMNIST('data/f_mnist', download=True, train=False, transform=transforms.ToTensor())
    client, server, clone = MnistNet(), MnistNet(), MnistNet()
    split_layer = 9
    grad_index = 8
elif dataset == 'cifar':
    trainset = datasets.CIFAR10('data/cifar', download=True, train=True, transform=transforms.ToTensor())
    testset = datasets.CIFAR10('data/cifar', download=True, train=False, transform=transforms.ToTensor())
    client, server, clone = CifarNet(), CifarNet(), CifarNet()
    split_layer = 17
    grad_index = 14

trainloader = torch.utils.data.DataLoader(trainset, shuffle=True, batch_size=64)
testloader = torch.utils.data.DataLoader(testset, shuffle=True)


# -- LABEL INFERENCE ATTACK --
client_opt = torch.optim.Adam(client.parameters(), lr=0.001, amsgrad=True)
server_opt = torch.optim.Adam(server.parameters(), lr=0.001, amsgrad=True)
clone_opt = torch.optim.Adam(clone.parameters(), lr=0.001, amsgrad=True)

criterion = torch.nn.CrossEntropyLoss()

results = []
for idx, (image, label) in enumerate(testloader):
    if idx == count:
        break

    # enumerate possible label values
    label_vals = [i * torch.ones(len(label)).long() for i in range(10)] 

    # obtain gradient values from client
    client_opt.zero_grad()
    server_opt.zero_grad()
    server_out = server(image, end=split_layer)
    pred = client(server_out, start=split_layer+1)
    loss = criterion(pred, label)
    loss.backward(retain_graph=True)

    target_grad = [param.grad for param in client.parameters()][grad_index]

    # obtain clone model's output
    clone_opt.zero_grad()
    clone_pred = clone(server_out, start=split_layer+1)

    # try out all possible labels and pick the one that produces the closest gradient values
    pred_label = attacks.label_inference(clone_pred, clone, target_grad, label_vals, grad_index)

    results.append(label.item() == pred_label.item())
    print(f'Label: {label.item()} - Predicted: {pred_label.item()}')

    # perform training updates
    clone_loss = criterion(clone_pred, pred_label)
    clone_loss.backward()
    client_opt.step()
    clone_opt.step()
    server_opt.step()

print('Run complete.')
print(f'Label inference accuracy: {sum(results) / count}')

