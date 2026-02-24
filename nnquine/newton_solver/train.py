import torch
import catppuccin
import matplotlib as mpl
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch.func import functional_call

from nnquine.gradient_solver.model import Quine, flatten_params, set_params

mpl.style.use(catppuccin.PALETTE.mocha.identifier)

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)

device = "cpu"


################################################################################
# SETUP
################################################################################

model = Quine(alpha=0.25).to(device)
fixed_input = torch.randn(1, 8)

P = sum(p.numel() for p in model.parameters())
print("Total parameters : ", P)


def F(theta_vec):
    param_dict = {}
    idx = 0
    for name, p in model.named_parameters():
        num = p.numel()
        param_dict[name] = theta_vec[idx:idx+num].view_as(p)
        idx += num

    return functional_call(model, param_dict, (fixed_input,)).view(-1)


def g(theta_vec):
    return F(theta_vec) - theta_vec


theta = flatten_params(model).detach().clone().requires_grad_(True)
fixed_input = torch.randn(1, 8)

# This takes more than an hour on Macbook M4 and it oscillates.So, sticking 
# with a lower number
# num_steps = 200000
num_steps = 1000
tol = 1e-16

residual_norms = []

print("\nRunning Newton solver ...\n")

for it in tqdm(range(num_steps), desc="Newton Progress"):
    with torch.no_grad():
        g_val = g(theta).detach()
        norm = torch.norm(g_val).item()
        residual_norms.append(norm)
        

    tqdm.write(f"Step {it:02d} | ||g|| = {norm:.6e}")

    if norm < tol:
        print("Converged")
        break

    # Compute Jacobian and solve linear update
    J = torch.autograd.functional.jacobian(F, theta)
    A = J - torch.eye(P, dtype=J.dtype)

    with torch.no_grad():
        eigvals = torch.linalg.eigvals(J)
        tqdm.write(f"max |eig(J_F)| : {eigvals.abs().max().item()}")

    delta = torch.linalg.solve(A, g_val)
    theta = (theta - delta).detach().requires_grad_(True)

# Finalize model
set_params(model, theta.detach())
final_out = model(fixed_input).view(-1).detach()
final_theta = flatten_params(model).detach()
diff = final_out - final_theta

print("\nFinal ||difference|| : ", torch.norm(diff).item())

################################################################################
# SAVE PROGRESS DATA
################################################################################

torch.save(
    {
        "model_state": model.state_dict(),
        "fixed_input": fixed_input,
        "final_diff_norm": torch.norm(diff).item(),
        "residual_norms": residual_norms,
    },
    "newton_solver.pth",
)

print("\nModel saved to newton_solver.pth")

################################################################################
# PLOT RESIDUALS
################################################################################

plt.figure(figsize=(6, 4))
plt.plot(residual_norms)
plt.yscale("log")
plt.title("Newton Residual Norm ||f(θ) - θ|| vs Iteration")
plt.xlabel("Iteration")
plt.ylabel("Residual Norm (log scale)")
plt.grid(True, linestyle="--", alpha=0.4)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("newton_solver_training_loss.png", dpi=300)
print("Saved residual plot as newton_solver_training_loss.png")
plt.show()
