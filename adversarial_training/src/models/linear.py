import torch
import torch.nn as nn

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

def LinearRegression():
    model = nn.Sequential(
        Flatten(), nn.Linear(784, 10))
    return model
