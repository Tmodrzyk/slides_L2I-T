# %%
import torch
from tqdm import tqdm
import deepinv
import numpy as np

device = "cuda"


def mlem(
    y: torch.Tensor,
    x0: torch.Tensor,
    stepsize: float,
    physics: deepinv.physics.LinearPhysics,
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


# Load data
# %%
import torchvision.transforms.functional as F
import seaborn as sns

sns.set_theme(context="talk", style="whitegrid")

data = np.load("../data/brainweb/data-1.npy", allow_pickle=True).item()
x = torch.from_numpy(data["imgHD"]).float().unsqueeze(0).unsqueeze(0).to(device)
x = (x - x.min()) / (x.max() - x.min())
x = F.center_crop(x, (128, 128))

# Convert to PIL image with gray_r colormap
import matplotlib.pyplot as plt
import matplotlib.cm as cm

x_np = x.squeeze().cpu().numpy()
x_colored = cm.get_cmap("gray_r")(x_np)
F.to_pil_image((x_colored[:, :, :3] * 255).astype(np.uint8)).save(
    "../assets/img/section2/brainweb_groundtruth_1.png"
)


# %%
# Setup Tomography physics
physics = deepinv.physics.Tomography(
    angles=180,
    img_width=128,
    device=device,
    noise_model=deepinv.physics.PoissonNoise(gain=1 / 200),
    normalize=True,
)
y = physics(x)
# Reconstruct with MLEM
x0 = torch.ones_like(physics.A_adjoint(y))
x_mlem, xs_mlem = mlem(
    y=y, x0=x0, stepsize=0.5, physics=physics, steps=200, verbose=True, keep_inter=True
)
x_mlem_earlystop, xs_earlystop = mlem(
    y=y, x0=x0, stepsize=0.5, physics=physics, steps=30, verbose=True, keep_inter=True
)
psnr = deepinv.loss.metric.PSNR()

# Calculate PSNR for each iteration
psnr_values_mlem = [psnr(x, xs.to(device)).item() for xs in xs_mlem]
psnr_values_earlystop = [psnr(x, xs.to(device)).item() for xs in xs_earlystop]

# Plot PSNR evolution
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 2))
plt.plot(
    psnr_values_mlem,
    label="MLEM PSNR",
)
plt.axvline(x=30, color="orange", linestyle="--", label="Early Stop (30 iterations)")
plt.xlabel("Iteration")
plt.ylabel("PSNR")
plt.ylim(0, 25)
plt.legend()
plt.savefig(
    "../assets/img/section2/psnr_mlem_brainweb.svg", dpi=96, bbox_inches="tight"
)
plt.show()

fig = deepinv.utils.plot(
    [
        x,
        y,
        physics.A_dagger(y),
        x_mlem,
        x_mlem_earlystop,
    ],
    titles=[
        "Ground-truth",
        "Measurements",
        "Filtered-Backprojection",
        "MLEM 200 Iterations",
        "MLEM 30 Iterations",
    ],
    figsize=(12, 6),
    fontsize=18,
    cmap="gray_r",
    return_fig=True,
    rescale_mode="clip",
    vmin=0,
    vmax=2,
)

fig.savefig(
    "../assets/img/section2/mlem_reconstruction_brainweb.svg",
    dpi=96,
    bbox_inches="tight",
)
# %%
