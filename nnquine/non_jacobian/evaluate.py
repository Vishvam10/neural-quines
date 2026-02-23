import torch
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import catppuccin

from catppuccin.extras.matplotlib import get_colormap_from_list

from nnquine.non_jacobian.model import Quine, flatten_params

mpl.style.use(catppuccin.PALETTE.macchiato.identifier)

torch.set_default_dtype(torch.float64)
device = "cpu"

################################################################################
# LOAD MODEL
################################################################################

checkpoint = torch.load("non_jacobian.pth", map_location=device)

model = Quine(alpha=0.25).to(device)
model.load_state_dict(checkpoint["model_state"])
model.eval()

theta = flatten_params(model).detach()
input_probe = checkpoint["input_probe"]

with torch.no_grad():
    pred_theta = model(input_probe).view(-1).detach()


cmap = get_colormap_from_list(
    catppuccin.PALETTE.macchiato.identifier,
   ["red", "surface2", "blue"]
)


################################################################################
# HEATMAP UTILS
################################################################################

def plot_comparison(actual, pred, title, vmin=None, vmax=None):
    diff = pred - actual

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    # Add overall plot title
    fig.suptitle(title, fontsize=12, fontweight="bold")

    if vmin is None or vmax is None:
        absmax = max(np.abs(actual).max(), np.abs(pred).max())
        vmin, vmax = -absmax, absmax

    axs[0].imshow(actual, cmap=cmap, vmin=vmin, vmax=vmax)
    axs[0].set_title("Actual")

    axs[1].imshow(pred, cmap=cmap, vmin=vmin, vmax=vmax)
    axs[1].set_title("Predicted")

    im2 = axs[2].imshow(diff, cmap=cmap, vmin=-absmax, vmax=absmax)
    axs[2].set_title("Difference")

    fig.colorbar(im2, ax=axs, orientation="vertical")
    plt.show()

################################################################################
# COMPARE LAYER WEIGHTS
################################################################################

with torch.no_grad():
    for layer_name in ["l1", "l2", "l3"]:
        W_act = getattr(model, layer_name).weight.detach().cpu().numpy()
        pred_full = pred_theta.cpu().numpy().reshape(-1)

        start = 0
        W_pred = None
        for name, param in model.named_parameters():
            if name.endswith(f"{layer_name}.weight"):
                end = start + param.numel()
                W_pred = pred_full[start:end].reshape(param.size())
                break
            start += param.numel()

        if W_pred is not None:
            plot_comparison(W_act, W_pred, "Queried Weights")
