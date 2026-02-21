import torch
import matplotlib.pyplot as plt

from tqdm import tqdm

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)

device = "cpu"

from nnquine.non_jacobian.model import Quine, set_params, flatten_params

##############################################################################
# SETUP
##############################################################################

model = Quine(alpha=0.25).to(device)

theta = flatten_params(model).detach().clone().requires_grad_(True)

# Fixed random seed / fixed z
input_probe = torch.randn(1, 8).to(device)

print("Total parameters : ", theta.numel())

# Choose optimizer
optimizer = torch.optim.RMSprop([theta], lr=1e-3)
print("Optimizer:", optimizer)

loss_list = []

##############################################################################
# TRAINING LOOP
##############################################################################

num_steps = 200000

for step in tqdm(range(num_steps), desc="Training Progress", unit="step"):
    optimizer.zero_grad()

    set_params(model, theta)

    pred = model(input_probe).view(-1)
    loss = torch.norm(pred - theta) ** 2

    loss.backward()
    optimizer.step()

    loss_list.append(loss.item())

    if step % 500 == 0:
        tqdm.write(f"Step {step:04d} | Loss = {loss.item():.3e}")

set_params(model, theta.detach())

print("\nFinal loss :", loss.item())

##############################################################################
# SAVE
##############################################################################

torch.save({
    "model_state": model.state_dict(),
    "theta": theta.detach(),
    "input_probe": input_probe
}, "non_jacobian.pth")

print("\nModel saved to non_jacobian.pth")

##############################################################################
# PLOT LOSS
##############################################################################

plt.figure(figsize=(6, 4))
plt.plot(loss_list, linewidth=1)
plt.yscale("log")
plt.title("Training Loss over Iterations")
plt.xlabel("Iteration")
plt.ylabel("Loss (log scale)")
plt.grid(True, linestyle="--", alpha=0.4)

plt.savefig("non_jacobian_training_loss.png", dpi=300)
print("Saved loss plot as non_jacobian_training_loss.png")
plt.show()