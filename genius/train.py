import torch
import torch.nn as nn

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)

device = "cpu"

from .model import Quine, set_params, flatten_params

################################################################################
# Setup
################################################################################

model = Quine(alpha=0.25).to(device)
z_fixed = torch.randn(1, 8)

P = sum(p.numel() for p in model.parameters())
print("Total parameters:", P)


def F(theta_vec):
    set_params(model, theta_vec)
    return model(z_fixed).view(-1)


def g(theta_vec):
    return F(theta_vec) - theta_vec


theta = flatten_params(model).detach().clone().requires_grad_(True)
z_fixed = theta.view(1, -1)[:, :8]


max_iters = 100
tol = 1e-16

print("\nRunning Newton solver ...\n")

for it in range(max_iters):
    g_val = g(theta).detach()
    norm = torch.norm(g_val).item()
    print(f"Iter {it:02d} | ||g|| = {norm:.6e}")

    if norm < tol:
        print("Converged")
        break

    J = torch.autograd.functional.jacobian(F, theta)
    A = J - torch.eye(P, dtype=J.dtype)
    
    # print("cond(A) =", torch.linalg.cond(A).item())

    delta = torch.linalg.solve(A, g_val)
    theta = (theta - delta).detach().requires_grad_(True)


# Finalize model
set_params(model, theta.detach())
final_out = model(z_fixed).view(-1).detach()
final_theta = flatten_params(model).detach()

diff = final_out - final_theta
print("\nFinal ||difference||:", torch.norm(diff).item())


################################################################################
# Save everything
################################################################################

torch.save({
    "model_state": model.state_dict(),
    "z_fixed": z_fixed,
    "final_diff_norm": torch.norm(diff).item()
}, "quine_fixed.pth")

print("\nModel saved to quine_fixed.pth")