import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import sys

class LeNet_5(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.cnn_model = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),        # (N, 1, 32, 32) -> (N, 6, 28, 28)
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),                      # (N, 6, 28, 28) -> (N, 6, 14, 14)
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),       # (N, 6, 14, 14) -> (N, 16, 10, 10)
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)                       # (N, 16, 10, 10) -> (N, 16, 5, 5)
        )
        
        # current shape of the image is 16, 5, 5
        # flatten to 16*5*5 = 400
        self.flatten = nn.Flatten()
        
        self.fc_model = nn.Sequential(
            nn.Linear(400, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )
        
    def forward(self, x):
        x = self.cnn_model(x)
        x = self.flatten(x)
        out = self.fc_model(x)
        return out

model = LeNet_5()
print(model)