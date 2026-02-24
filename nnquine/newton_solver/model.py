import torch
import torch.nn as nn

torch.set_default_dtype(torch.float64)


class Quine(nn.Module):
    def __init__(self, alpha=0.25):
        super().__init__()
        self.alpha = alpha

        self.l1 = nn.Linear(8, 20)
        self.l2 = nn.Linear(20, 20)
        self.l3 = nn.Linear(20, 20)

        self.P = sum(p.numel() for p in self.parameters())

        proj = torch.randn(20, self.P) / 20.0
        self.register_buffer("proj", proj)

        self.activation = nn.Tanh()

    def forward(self, z):
        x = self.activation(self.l1(z))
        x = self.activation(self.l2(x))
        x = self.activation(self.l3(x))
        out = x @ self.proj
        return self.alpha * out


def flatten_params(model):
    return torch.cat([p.view(-1) for p in model.parameters()])


def set_params(model, theta_vec):
    idx = 0
    with torch.no_grad():
        for p in model.parameters():
            num = p.numel()
            p.copy_(theta_vec[idx : idx + num].view_as(p))
            idx += num
