import torch
import torch.nn as nn

torch.set_default_dtype(torch.float64)

"""
In this case, we will be following something dead similar to this :
https://arxiv.org/abs/1803.05859

Earlier, we were modelling the whole quine problem as a regression problem with
multiple outputs ... this paper does something clever and interesting. The 
method described in this paper sorta makes the input "control" the output. The 
way it does that is by making the weights of the WHOLE neural network QUERYABLE.

So, inp[i][j] = W_ij .. this is made clear by using a one-hot encoder to one 
which weight is currently being queried.

Again, this DOES NOT solve the 0-quine problem. To bypass that issue, quoting 
from https://evanfletcher42.com/2022/10/31/neural-quines/ :

"We can forcibly avoid the zero quine by adding a parameter-free normalization layer after the trained dense layer. Both instance normalization and batch normalization (with all learned & running parameters disabled - no extra weights!) produce imperfect, but non-trivial, quines with relatively low error"

Hence, we add the nn.InstanceNorm1d between each layer
"""

class Quine(nn.Module):
    def __init__(self, alpha=0.25):
        super().__init__()
        self.alpha = alpha

        self.l1 = nn.Linear(8, 20)
        self.l2 = nn.Linear(20, 20)
        self.l3 = nn.Linear(20, 20)
        

        self.norm1 = nn.InstanceNorm1d(20, affine=False)
        self.norm2 = nn.InstanceNorm1d(20, affine=False)
        self.norm3 = nn.InstanceNorm1d(20, affine=False)

        self.P = sum(p.numel() for p in self.parameters())

        # Fixed random projection
        proj = torch.randn(20, self.P) / 20.0
        self.register_buffer("proj", proj)

        self.activation = nn.Tanh()

    def forward(self, z):
        x = self.activation(self.l1(z))
        x = self.norm1(x.unsqueeze(0)).squeeze(0)

        x = self.activation(self.l2(x))
        x = self.norm2(x.unsqueeze(0)).squeeze(0)

        x = self.activation(self.l3(x))
        x = self.norm3(x.unsqueeze(0)).squeeze(0)

        out = x @ self.proj
        return self.alpha * out


def flatten_params(model):
    return torch.cat([p.view(-1) for p in model.parameters()])


def set_params(model, theta_vec):
    idx = 0
    with torch.no_grad():
        for p in model.parameters():
            num = p.numel()
            p.copy_(theta_vec[idx:idx + num].view_as(p))
            idx += num