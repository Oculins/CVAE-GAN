import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as models



class Discriminator(nn.Module):
    def __init__(self, num_classes=1, pretrained=False):
        super().__init__()
        self.features_extractor = models.resnet50(pretrained)

        self.classifier = nn.Sequential(
            nn.Linear(1000, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
            nn.Sigmoid()
        )

        self.criterion = nn.BCELoss()

    def forward(self, images):
        h = self.features_extractor(images)
        h = self.classifier(h)
        # h = torch.softmax(h, dim=1)

        # _, pred = torch.max(h, 1)

        return h
    
    def loss_function(self, r1, r2):
        return self.criterion(r1, r2)


class Regressor(nn.Module):
    def __init__(self, num_classes=1, pretrained=False):
        super().__init__()
        self.backbone = models.resnet50(pretrained)

        self.regression = nn.Sequential(
            nn.Linear(1000, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
            nn.ReLU()
        )

        self.loss_func = nn.MSELoss()


    def forward(self, images):
        x = self.backbone(images)
        x = self.regression(x)

        return x

    def loss_function(self, r1, r2):
        return self.loss_func(r1, r2)


