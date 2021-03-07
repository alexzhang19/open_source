#!/usr/bin/env python3
# coding: utf-8

"""
@File      : train.py
@Author    : alex
@Date      : 2021/1/23
@Desc      : 
"""

import time
import torch
from models.unet import UNet
import torch.optim as optim
import torch.nn.functional as F
from collections import defaultdict
from torch.optim import lr_scheduler
from data_loader import train_loader, valid_loader, n_cls


def dice_loss(pred, target, smooth=1.):
    pred = pred.contiguous()
    target = target.contiguous()
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    return loss.mean()


def calc_loss(pred, target, metrics, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)

    # pred = F.sigmoid(pred)
    pred = torch.sigmoid(pred)
    dice = dice_loss(pred, target)
    loss = bce * bce_weight + dice * (1 - bce_weight)

    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)
    return loss


def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))
    print("{}: {}".format(phase, ", ".join(outputs)))


def valid(model, device):
    metrics = defaultdict(float)

    model.eval()  # Set models to evaluate mode
    epoch_samples = 0
    for inputs, labels in valid_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss = calc_loss(outputs, labels, metrics)
        epoch_samples += inputs.size(0)

    print_metrics(metrics, epoch_samples, "val")
    epoch_loss = metrics['loss'] / epoch_samples
    return epoch_loss


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(n_cls).to(device)
    model.load_state_dict(torch.load("best_model_wts.pth", map_location=device))
    # Observe that all parameters are being optimized
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)

    best_loss = 1e10
    num_epochs = 500
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        start_time = time.time()
        # Each epoch has a training and validation phase
        scheduler.step()
        for param_group in optimizer.param_groups:
            print("LR", param_group['lr'])
        model.train()  # Set models to training mode

        metrics = defaultdict(float)
        epoch_samples = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                loss = calc_loss(outputs, labels, metrics)
                loss.backward()
                optimizer.step()

            # statistics
            epoch_samples += inputs.size(0)

        print_metrics(metrics, epoch_samples, "train")
        train_loss = metrics['loss'] / epoch_samples
        end_time = time.time()
        print(f"train loss = %.3f, train time = %.1fs" % (train_loss, end_time - start_time))

        val_loss = valid(model, device)
        if val_loss < best_loss:
            print(f"epoch = {epoch}, saving best model")
            best_loss = val_loss
            torch.save(model.state_dict(), "best_model_wts.pth")


if __name__ == "__main__":
    train()
