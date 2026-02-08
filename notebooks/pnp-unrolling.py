# %%
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt

sns.set_style("whitegrid")
plt.figure(figsize=(15, 5))

x = np.linspace(-2, 2, 1000)
y1 = x**2
y2 = x**4
y3 = np.abs(x)

fontsize = 18
plt.subplot(1, 3, 1)
sns.lineplot(x=x, y=y1, linewidth=2.5, color="steelblue")
plt.title(
    "Convex differentiable $L$-smooth Function\n$f(x) = x^2$",
    fontsize=fontsize,
    fontweight="bold",
)
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
sns.lineplot(x=x, y=y2, linewidth=2.5, color="coral")
plt.title(
    "Convex differentiable non $L$-smooth\n$f(x) = x^4$",
    fontsize=fontsize,
    fontweight="bold",
)
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
sns.lineplot(x=x, y=y3, linewidth=2.5, color="mediumseagreen")
plt.title(
    "Convex non-differentiable\n$f(x) = |x|$", fontsize=fontsize, fontweight="bold"
)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(
    "../assets/img/section2/regularization_smooth_nonsmooth.pdf",
    dpi=300,
    bbox_inches="tight",
)
plt.show()
# %%
import deepinv
import torch

device = "cuda"
# Load example images
img = deepinv.utils.demo.load_example("butterfly.png", img_size=256, device=device)
img_leaves = deepinv.utils.demo.load_example("leaves.png", img_size=256, device=device)
img_starfish = deepinv.utils.demo.load_image(
    "../assets/img/section2/starfish.png",
    resize_mode="resize",
    img_size=256,
    device=device,
)

# Define blur operator
psf = deepinv.physics.blur.gaussian_blur(sigma=2.0)
physics = deepinv.physics.Blur(
    filter=psf,
    padding="circular",
    noise_model=deepinv.physics.GaussianNoise(sigma=15 / 255),
    device=device,
)
# Create batch of images
imgs = torch.stack([img, img_leaves, img_starfish], dim=1)[0]
y = physics(imgs)
deepinv.utils.plot(
    [imgs, y],
)
# %%
data_fidelity = deepinv.optim.L2()
tv = deepinv.optim.TVPrior(n_it_max=50)

# gd_noprior = deepinv.optim.GD(
#     data_fidelity=data_fidelity,
# )
pgd_tv = deepinv.optim.PGD(
    data_fidelity=data_fidelity,
    prior=tv,
    lambda_reg=0.025,
)
dpir = deepinv.optim.DPIR(
    sigma=15 / 255,
    denoiser=deepinv.models.DRUNet(device=device),
    device=device,
)

# x_noprior = gd_noprior(y, physics=physics)
x_tv = pgd_tv(y, physics=physics)
x_dpir = dpir(y, physics=physics)

# %%
psnr = deepinv.loss.metric.PSNR()

psnr_y = psnr(imgs, y)
psnr_tv = psnr(imgs, x_tv)
psnr_dpir = psnr(imgs, x_dpir)

# Plot results for all three images
titles = ["Butterfly", "Leaves", "Starfish"]
fig = deepinv.utils.plot(
    [imgs, y, x_tv, x_dpir],
    titles=[
        r"Ground-Truth $x$",
        r"Measurement $y$",
        r"TV",
        r"Plug-and-Play",
    ],
    subtitles=[
        [
            "PSNR:",
            f"{psnr_y[0].item():.2f} dB",
            f"{psnr_tv[0].item():.2f} dB",
            f"{psnr_dpir[0].item():.2f} dB",
        ],
        [
            "PSNR:",
            f"{psnr_y[1].item():.2f} dB",
            f"{psnr_tv[1].item():.2f} dB",
            f"{psnr_dpir[1].item():.2f} dB",
        ],
        [
            "PSNR:",
            f"{psnr_y[2].item():.2f} dB",
            f"{psnr_tv[2].item():.2f} dB",
            f"{psnr_dpir[2].item():.2f} dB",
        ],
    ],
    fontsize=16,
    figsize=(16, 6),
    return_fig=True,
)

fig.savefig(
    "../assets/img/section2/regularization_tv_pnp.pdf",
    dpi=300,
    bbox_inches="tight",
)
# %%
