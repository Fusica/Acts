import torch
import torch.nn as nn


class DSigmoid(nn.Module):
    def __init__(self):
        super(DSigmoid, self).__init__()
        self.alpha = nn.Parameter(torch.randn(1), requires_grad=True)
        nn.init.constant_(self.alpha, torch.pi / 2)

    def forward(self, x):
        return torch.atan(x / self.alpha) / torch.pi + 1 / 2


class DReLU(nn.Module):
    def __init__(self, c1):
        super().__init__()
        self.p1 = nn.Parameter(torch.randn(1, c1, 1, 1), requires_grad=True)
        self.p2 = nn.Parameter(torch.randn(1, c1, 1, 1), requires_grad=True)
        self.beta = nn.Parameter(torch.ones(1, c1, 1, 1), requires_grad=True)
        self.sigmoid = DSigmoid()

    def forward(self, x):
        dpx = (self.p1 - self.p2) * x
        return dpx * self.sigmoid(self.beta * dpx) + self.p2 * x