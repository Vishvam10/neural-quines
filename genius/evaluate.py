import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from .model import Quine, flatten_params

torch.set_default_dtype(torch.float64)
device = "cpu"

################################################################################
# LOAD MODEL
################################################################################

checkpoint = torch.load("quine_fixed.pth", map_location=device)

model = Quine(alpha=0.25).to(device)
model.load_state_dict(checkpoint["model_state"])
model.eval()

theta = flatten_params(model).detach()

################################################################################
# FIRST LAYER INSPECTION
################################################################################

with torch.no_grad():
    W = model.l1.weight
    b = model.l1.bias

print("First layer weight stats:")
print("Mean : ", W.mean().item())
print("Std  : ", W.std().item())
print("Max  : ", W.abs().max().item())

print("\nFirst layer bias stats: ")
print("Mean : ", b.mean().item())
print("Std  : ", b.std().item())
print("Max  : ", b.abs().max().item())

print("\nFirst few weight rows:")
print(W[:5])

################################################################################
# FIXED POINT CHECK
################################################################################

z_fixed = checkpoint["z_fixed"]

out_fixed = model(z_fixed).view(-1).detach()
print("\nFixed-point diff (training z):")
print("||f(z_fixed) - theta|| =", torch.norm(out_fixed - theta).item())

################################################################################
# RANDOM INPUT TEST
################################################################################

num_tests = 200
outs = []

for _ in range(num_tests):
    z_rand = torch.randn(1, 8)
    out = model(z_rand).view(-1).detach()
    outs.append(out)

outs = torch.stack(outs)

diffs = torch.norm(outs - theta, dim=1)

print("\nAcross random inputs:")
print("Mean diff : ", diffs.mean().item())
print("Max diff  : ", diffs.max().item())
print("Min diff  : ", diffs.min().item())

################################################################################
# VARIANCE ANALYSIS
################################################################################

output_variance = outs.var(dim=0).mean().item()
max_output_change = (outs.max(dim=0).values - outs.min(dim=0).values).abs().max().item()

print("\nOutput variance across inputs (mean over dims) : ", output_variance)
print("Max absolute change in any output dim : ", max_output_change)

################################################################################
# INPUT GRADIENT TEST
################################################################################

z_test = torch.randn(1, 8, requires_grad=True)
out_test = model(z_test)
loss = out_test.norm()
loss.backward()

grad_norm = z_test.grad.norm().item()

print("\n||df/dz|| (input sensitivity) : ", grad_norm)


def plot_weight_matrix(W, title, clip=None):
    W_np = W.detach().cpu().numpy()

    if clip is None:
        max_abs = np.max(np.abs(W_np))
    else:
        max_abs = clip

    plt.figure(figsize=(6, 4))
    plt.imshow(
        W_np,
        aspect='auto',
        cmap='coolwarm',
        vmin=-max_abs,
        vmax=max_abs
    )
    plt.colorbar()
    plt.title(f"{title} (±{max_abs:.2e})")
    plt.show()

plot_weight_matrix(model.l1.weight, "Layer 1 Weights", clip=1e-18)
plot_weight_matrix(model.l2.weight, "Layer 2 Weights")
plot_weight_matrix(model.l3.weight, "Layer 3 Weights")