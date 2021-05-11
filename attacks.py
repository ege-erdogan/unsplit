import numpy as np
import torch
from torchvision import transforms, datasets

from models import *
from util import *

def model_inversion_stealing(clone_model, split_layer, target, input_size, 
                            lambda_tv=0.1, lambda_l2=1, main_iters=1000, input_iters=100, model_iters=100):
    x_pred = torch.empty(input_size).fill_(0.5).requires_grad_(True)
    input_opt = torch.optim.Adam([x_pred], lr=0.001, amsgrad=True)
    model_opt = torch.optim.Adam(clone_model.parameters(), lr=0.001, amsgrad=True)
    mse = torch.nn.MSELoss()

    for main_iter in range(main_iters):
        for input_iter in range(input_iters):
            input_opt.zero_grad()
            pred = clone_model(x_pred, end=split_layer)
            loss = mse(pred, target) + lambda_tv*TV(x_pred) + lambda_l2*l2loss(x_pred)
            loss.backward(retain_graph=True)
            input_opt.step()
        for model_iter in range(model_iters):
            model_opt.zero_grad()
            pred = clone_model(x_pred, end=split_layer)
            loss = mse(pred, target) 
            loss.backward(retain_graph=True)
            model_opt.step()

    return x_pred.detach()


def label_inference(pred, clone_model, target_grad, label_vals, grad_index):
    pred_losses = [torch.nn.CrossEntropyLoss()(pred, cd_label) for cd_label in label_vals]
    pred_grads = [torch.autograd.grad(loss, clone_model.parameters(), allow_unused=True, retain_graph=True)[grad_index] for loss in pred_losses]
    grad_losses = [torch.nn.MSELoss()(pred_grad, target_grad) for pred_grad in pred_grads]
    return torch.LongTensor([grad_losses.index(min(grad_losses))])