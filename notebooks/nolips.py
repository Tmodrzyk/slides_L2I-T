# %%
import torch
from tqdm import tqdm

import deepinv as dinv
import matplotlib.pyplot as plt
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"


def kl(x, y, bkg=0, eps=1e-20, reduction: str = "none"):
    """
    KL(y || Ax + bkg) for Poisson data.

    Parameters
    ----------
    y : Tensor     (B, …)  observed sinogram / counts
    x : Tensor     (B, …)  reconstruction image(s)
    physics : obj   must expose `A` (forward projector) that maps x → sinogram
    bkg : Tensor or float, broadcastable to sinogram shape
    eps : float     small positive number to avoid log(0)
    reduction : {'none', 'sum', 'mean'}
        - 'none' → 1D tensor, one KL per sample
        - 'sum'  → scalar, sum over batch and pixels
        - 'mean' → scalar, average over batch

    Returns
    -------
    Tensor        KL divergence, shape depends on `reduction`
    """
    # Forward model λ = A x + bkg  (ensure strictly positive)
    lam = x.clamp_min(eps)  # avoids λ = 0 → log NaN

    # Clamp y as well so y log y is finite even when y = 0
    y_safe = y.clamp_min(eps)

    # Element-wise KL and reduce
    kl_pix = lam - y_safe + y_safe * torch.log(y_safe / lam)

    if reduction == "none":
        return kl_pix.flatten(1).sum(dim=1)  # (B,)
    elif reduction == "sum":
        return kl_pix.sum()  # scalar
    elif reduction == "mean":
        return kl_pix.flatten(1).sum(dim=1).mean()  # scalar
    else:
        raise ValueError("reduction must be 'none', 'sum' or 'mean'")


def mlem(
    y: torch.Tensor,
    x0: torch.Tensor,
    stepsize: float,
    physics: dinv.physics.LinearPhysics,
    steps: int,
    verbose: bool = True,
    keep_inter: bool = False,
    filter_epsilon: float = 1e-20,
) -> torch.Tensor:
    """
    Performs Richardson-Lucy deconvolution on an observed image.

    Args:
        y (torch.Tensor): The observed image
        x_0 (torch.Tensor): The initial estimate
        physics (deepinv.physics.LinearPhysics): The physics operator
        steps (int): Number of iterations
        verbose (bool): Whether to show progress bar
        keep_inter (bool): Whether to keep intermediate results

    Returns:
        torch.Tensor or tuple: The deconvolved image, and if keep_inter=True, list of intermediate results
    """
    xs = [x0.cpu().clone()] if keep_inter else None

    with torch.no_grad():
        recon = x0.clone()
        recon = recon.clamp(min=filter_epsilon)
        s = physics.A_adjoint(torch.ones_like(y))

        for step in tqdm(range(steps), desc="MLEM", disable=not verbose):
            mlem = (recon / s) * physics.A_adjoint(
                y / physics.A(recon).clamp(min=filter_epsilon)
            )
            recon = (1 - stepsize) * recon + stepsize * mlem

            if keep_inter:
                xs.append(recon.cpu().clone())

        return (recon, xs) if keep_inter else recon


def nolips(
    y: torch.Tensor,
    x0: torch.Tensor,
    physics: dinv.physics.LinearPhysics,
    stepsize: float,
    steps: int,
    verbose: bool = True,
    keep_inter: bool = False,
    backtracking: bool = False,
    gamma: float = 0.5,
    eta: float = 0.5,
    filter_epsilon: float = 1e-20,
) -> torch.Tensor:

    xs = [x0.cpu().clone()] if keep_inter else None

    with torch.no_grad():
        recon = x0.clone()
        recon = recon.clamp(min=filter_epsilon)
        s = physics.A_adjoint(torch.ones_like(y))

        for step in tqdm(range(steps), desc="NoLips", disable=not verbose):
            backtracked = False
            grad = s - physics.A_adjoint(y / physics.A(recon).clamp(min=filter_epsilon))
            recon_next = recon / (1 + stepsize * recon * grad)

            if backtracking:
                # Backtracking line search
                obj_prev = kl_divergence(
                    physics.A(recon).clamp(min=filter_epsilon),
                    y.clamp(min=filter_epsilon),
                )
                obj_next = kl_divergence(
                    physics.A(recon_next).clamp(min=filter_epsilon),
                    y.clamp(min=filter_epsilon),
                )
                while obj_prev - obj_next < (gamma / stepsize) * itakura_saito(
                    recon_next.clamp(min=filter_epsilon),
                    recon.clamp(min=filter_epsilon),
                ):
                    stepsize *= eta
                    recon_next = recon / (1 + stepsize * recon * grad)
                    obj_next = kl_divergence(
                        physics.A(recon_next).clamp(min=filter_epsilon),
                        y.clamp(min=filter_epsilon),
                    )
                    backtracked = True
                if backtracked:
                    print(f"Step size at iteration {step}: {stepsize}")

            recon = recon_next

            if keep_inter:
                xs.append(recon.cpu().clone())

        return (recon, xs) if keep_inter else recon


# %%
# You can also use other noise models but my algorithm is designed for Poisson noise
gain = 1 / 5
noise_model = dinv.physics.PoissonNoise(gain=gain)

img_width = 128

physics = dinv.physics.Tomography(
    img_width=img_width,
    angles=120,
    noise_model=noise_model,
    device=device,
)
x = dinv.utils.demo.load_example(
    "SheppLogan.png",
    device=device,
    img_size=img_width,
    grayscale=True,
    resize_mode="resize",
)
# Create a batch of 20 projections of x
batch_size = 5
x = x.repeat(batch_size, 1, 1, 1).to(device)
# %%
# Run baseline experiments without regularization

# for batch in tqdm((dataloader)):
y = physics(x)

steps = 1000
x0 = torch.ones_like(x)
stepsize = 1

x_mlem, xs_mlem = mlem(
    y=y,
    x0=x0,
    stepsize=1,
    physics=physics,
    steps=steps,
    verbose=True,
    keep_inter=True,
)

x_nolips_base, xs_nolips_base = nolips(
    y=y,
    x0=x0,
    physics=physics,
    stepsize=1,
    steps=steps,
    keep_inter=True,
    verbose=True,
)


def cost_function_base(x, y, physics):
    Ax = physics.A(x)
    return kl(y, Ax)


mae = dinv.loss.metric.MAE()

# Process each image in the batch
costs_mlem = []
costs_nolips_base = []
mae_vals_mlem = []
mae_vals_nolips_base = []

for x_inter in xs_mlem:
    cost = cost_function_base(x_inter.to(device), y, physics).detach().cpu().numpy()
    costs_mlem.append(cost)
    mae_val = mae(x_inter.to(device), x).detach().cpu().numpy()
    mae_vals_mlem.append(mae_val)

for x_inter in xs_nolips_base:
    cost = cost_function_base(x_inter.to(device), y, physics).detach().cpu().numpy()
    costs_nolips_base.append(cost)
    mae_val = mae(x_inter.to(device), x).detach().cpu().numpy()
    mae_vals_nolips_base.append(mae_val)


# %%
import seaborn as sns

sns.set_theme(context="talk", style="whitegrid")
mean_mlem = np.mean(costs_mlem, axis=1)
std_mlem = np.std(costs_mlem, axis=1)
mean_nolips_base = np.mean(costs_nolips_base, axis=1)
std_nolips_base = np.std(costs_nolips_base, axis=1)

# Compute mean and std for MAE
mean_mae_mlem = np.mean(mae_vals_mlem, axis=1)
std_mae_mlem = np.std(mae_vals_mlem, axis=1)
mean_mae_nolips_base = np.mean(mae_vals_nolips_base, axis=1)
std_mae_nolips_base = np.std(mae_vals_nolips_base, axis=1)

# Plot baseline results - KL Divergence
plt.figure(figsize=(10, 6))
iterations = range(len(mean_mlem))

plt.plot(iterations, mean_mlem, label="MLEM", linestyle="-")
plt.fill_between(iterations, mean_mlem - std_mlem, mean_mlem + std_mlem, alpha=0.3)

plt.plot(iterations, mean_nolips_base, label="NoLips", linestyle="-")
plt.fill_between(
    iterations,
    mean_nolips_base - std_nolips_base,
    mean_nolips_base + std_nolips_base,
    alpha=0.3,
)

plt.xlabel("Iteration", fontsize=20)
plt.ylabel(r"$f(x^{(k)})$", fontsize=20)
plt.legend(fontsize=24)
plt.title("Tomography on the Shepp-Logan Phantom", fontsize=20)
plt.savefig(
    "../assets/img/section4/nolips_vs_mlem_tomo.pdf", dpi=96, bbox_inches="tight"
)
plt.show()

dinv.utils.plot(
    [x, x_mlem, x_nolips_base],
    titles=["Ground Truth", "MLEM", "NoLips"],
    rescale_mode="clip",
    vmin=0,
    vmax=1,
    cmap="gray_r",
)
# %%
# Find best iterations for each algorithm based on MAE
best_iter_mlem = np.argmin(mean_mae_mlem)
best_iter_nolips_base = np.argmin(mean_mae_nolips_base)

print(
    f"Best iteration for MLEM: {best_iter_mlem} (MAE: {mean_mae_mlem[best_iter_mlem]:.4f})"
)
print(
    f"Best iteration for NoLips: {best_iter_nolips_base} (MAE: {mean_mae_nolips_base[best_iter_nolips_base]:.4f})"
)
# Get the best reconstructions
x_mlem_best = xs_mlem[best_iter_mlem].mean(dim=0)
x_nolips_base_best = xs_nolips_base[best_iter_nolips_base].mean(dim=0)

# Get the standard deviations
x_mlem_best_std = xs_mlem[best_iter_mlem].std(dim=0)
x_nolips_base_best_std = xs_nolips_base[best_iter_nolips_base].std(dim=0)

dinv.utils.plot(
    img_list=[
        x.mean(dim=0),
        x_mlem_best,
        x_nolips_base_best,
        x_mlem_best_std,
        x_nolips_base_best_std,
    ],
    titles=[
        "Ground Truth",
        f"MLEM (iter {best_iter_mlem})",
        f"NoLips (iter {best_iter_nolips_base})",
        f"MLEM Std (iter {best_iter_mlem})",
        f"NoLips Std (iter {best_iter_nolips_base})",
    ],
    rescale_mode="clip",
    vmin=0,
    vmax=1,
    cmap="gray_r",
)
# %%
