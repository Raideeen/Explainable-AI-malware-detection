import torch.nn as nn

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class MaxPool2D(nn.Module):
    def forward(self, x):
        return nn.functional.max_pool2d(x, 2)

def MNIST(): 
    model = nn.Sequential(
        nn.Conv2d(1, 32, 5, stride=1, padding=2),
        MaxPool2D(),
        nn.ReLU(),
        nn.Conv2d(32, 64, 5, stride=1, padding=2),
        MaxPool2D(),
        nn.ReLU(),
        Flatten(),
        nn.Linear(64*7*7,1024),
        nn.ReLU(),
        nn.Linear(1024, 10)
    )
    return model
