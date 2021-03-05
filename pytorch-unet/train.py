#!/usr/bin/env python3
# coding: utf-8

"""
@File      : train.py
@Author    : alex
@Date      : 2021/3/5
@Desc      : 
"""

import time
import copy
import torch
import os, sys
import numpy as np
import helper
import simulation
from torchvision import models
from torch import nn
import torchsummary
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy

class FCN(nn.Module):

    def __init__(self, base_model, n_class):
        super().__init__()

        self.base_model = models.resnet18(pretrained=True)
        layers = list(base_model.children())
        self.layer1 = nn.Sequential(*layers[:5])  # size=(N, 64, x.H/2, x.W/2)
        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.layer2 = layers[5]  # size=(N, 128, x.H/4, x.W/4)
        self.upsample2 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.layer3 = layers[6]  # size=(N, 256, x.H/8, x.W/8)
        self.upsample3 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        self.layer4 = layers[7]  # size=(N, 512, x.H/16, x.W/16)
        self.upsample4 = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)
        self.conv1k = nn.Conv2d(64 + 128 + 256 + 512, n_class, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        up1 = self.upsample1(x)
        x = self.layer2(x)
        up2 = self.upsample2(x)
        x = self.layer3(x)
        up3 = self.upsample3(x)
        x = self.layer4(x)
        up4 = self.upsample4(x)
        merge = torch.cat([up1, up2, up3, up4], dim=1)
        merge = self.conv1k(merge)
        out = self.sigmoid(merge)
        return out


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            batch_size = 10
            epoch_steps = 10
            for i in range(epoch_steps):
                input_images, target_masks = simulation.generate_random_data(192, 192, count=batch_size)

                inputs = torch.from_numpy(input_images)
                labels = torch.from_numpy(target_masks)
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / (batch_size * epoch_steps)

            print('{} Loss: {:.4f}'.format(
                phase, epoch_loss))

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def main():
    imgs, gt_masks = simulation.generate_random_data(192, 192, count=3)
    print(imgs.shape, gt_masks.shape)

    base_model = models.resnet18(pretrained=True)

    def find_last_layer(layer):
        children = list(layer.children())

        if len(children) == 0:
            return layer
        else:
            return find_last_layer(children[-1])

    list(base_model.children())
    model_wo_avgpool = nn.Sequential(*list(base_model.children())[:-2])

    model = FCN(base_model, 6).to(device)
    # torchsummary.summary(fcn_model, input_size=(3, 224, 224), device='cpu')
    criterion = nn.BCELoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=25)

    pass


if __name__ == "__main__":
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = "cpu"

    main()
