import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from nnquine.non_jacobian.model import Quine, flatten_params

torch.set_default_dtype(torch.float64)
device = "cpu"

##############################################################################
# LOAD MODEL
##############################################################################

checkpoint = torch.load("non_jacobian.pth", map_location=device)

model = Quine(alpha=0.25).to(device)
model.load_state_dict(checkpoint["model_state"])
model.eval()

theta = flatten_params(model).detach()
input_probe = checkpoint["input_probe"]

with torch.no_grad():
    pred_theta = model(input_probe).view(-1).detach()

##############################################################################
# CATPPUCCIN FRAPPE COLORS
##############################################################################

# Hard‑coded Catppuccin Frappe palette (hex colors) :contentReference[oaicite:1]{index=1}
FRAPPE_HEX = {
    "rosewater": "#f2d5cf",
    "flamingo":  "#eebebe",
    "pink":      "#f4b8e4",
    "mauve":     "#ca9ee6",
    "red":       "#e78284",
    "maroon":    "#ea999c",
    "peach":     "#ef9f76",
    "yellow":    "#e5c890",
    "green":     "#a6d189",
    "teal":      "#81c8be",
    "sky":       "#99d1db",
    "sapphire":  "#85c1dc",
    "blue":      "#8caaee",
    "lavender":  "#babbf1",
    "text":      "#c6d0f5",
    "base":      "#303446",
    "mantle":    "#292c3c",
    "crust":     "#232634",
}

plt.style.use("dark_background")
plt.rcParams["figure.facecolor"] = FRAPPE_HEX["base"]
plt.rcParams["axes.facecolor"] = FRAPPE_HEX["mantle"]

# Custom colormap from Frappe palette
cmap = plt.get_cmap("coolwarm")

##############################################################################
# HEATMAP UTILS
##############################################################################

def plot_comparison(actual, pred, title, vmin=None, vmax=None):
    """Plot three heatmaps: actual, predicted, and their difference."""
    
    diff = pred - actual
    
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(title, color=FRAPPE_HEX["text"])

    if vmin is None or vmax is None:
        absmax = max(np.abs(actual).max(), np.abs(pred).max())
        vmin, vmax = -absmax, absmax

    im0 = axs[0].imshow(actual, cmap=cmap, vmin=vmin, vmax=vmax)
    axs[0].set_title("Actual", color=FRAPPE_HEX["sky"])
    im1 = axs[1].imshow(pred, cmap=cmap, vmin=vmin, vmax=vmax)
    axs[1].set_title("Predicted", color=FRAPPE_HEX["sky"])
    im2 = axs[2].imshow(diff, cmap=cmap, vmin=-absmax, vmax=absmax)
    axs[2].set_title("Difference", color=FRAPPE_HEX["red"])

    for ax in axs:
        ax.tick_params(colors=FRAPPE_HEX["text"])
    fig.colorbar(im2, ax=axs, orientation="vertical")
    plt.show()

    # show correlation coefficient
    corr = np.corrcoef(actual.flatten(), pred.flatten())[0, 1]
    print(f"Correlation (actual vs pred) for {title} = {corr:.4f}")

##############################################################################
# COMPARE LAYER WEIGHTS
##############################################################################

with torch.no_grad():
    for layer_name in ["l1", "l2", "l3"]:
        W_act = getattr(model, layer_name).weight.detach().cpu().numpy()
        
        # predicted full theta
        pred_full = pred_theta.cpu().numpy().reshape(-1)
        
        # extract corresponding slice for this layer
        start = 0
        for name, param in model.named_parameters():
            if name.endswith(f"{layer_name}.weight"):
                end = start + param.numel()
                W_pred = pred_full[start:end].reshape(param.size())
                break
            start += param.numel()

        plot_comparison(W_act, W_pred, f"Layer {layer_name} weights")
