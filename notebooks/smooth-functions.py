# %%
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt

# Set style
sns.set_style("whitegrid")
plt.figure(figsize=(15, 5))

# Define x range
x = np.linspace(-2, 2, 1000)

# 1. Convex L-smooth function: f(x) = x^2
y1 = x**2

# 2. Convex differentiable but non L-smooth: f(x) = x^4
y2 = x**4

# 3. Convex non-differentiable: f(x) = |x|
y3 = np.abs(x)

# Plot 1: L-smooth function
fontsize = 18
plt.subplot(1, 3, 1)
sns.lineplot(x=x, y=y1, linewidth=2.5, color="steelblue")
plt.title(
    "Convex differentiable $L$-smooth Function\n$f(x) = x^2$",
    fontsize=fontsize,
    fontweight="bold",
)
plt.grid(True, alpha=0.3)

# Plot 2: Convex differentiable but non L-smooth
plt.subplot(1, 3, 2)
sns.lineplot(x=x, y=y2, linewidth=2.5, color="coral")
plt.title(
    "Convex differentiable non $L$-smooth\n$f(x) = x^4$",
    fontsize=fontsize,
    fontweight="bold",
)
plt.grid(True, alpha=0.3)

# Plot 3: Convex non-differentiable
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

device = "cuda"
# Load example image
img = deepinv.utils.demo.load_example("butterfly.png", img_size=256, device=device)

# Define blur operator
psf = deepinv.physics.blur.gaussian_blur(sigma=2.0)
physics = deepinv.physics.Blur(
    filter=psf,
    padding="circular",
    noise_model=deepinv.physics.GaussianNoise(sigma=15 / 255),
    device=device,
)
data_fidelity = deepinv.optim.L2()
tikhonov = deepinv.optim.Tikhonov()
tv = deepinv.optim.TVPrior(n_it_max=50)
# %%
gd_noprior = deepinv.optim.GD(
    data_fidelity=data_fidelity,
)
gd_tik = deepinv.optim.GD(
    data_fidelity=data_fidelity,
    prior=tikhonov,
    lambda_reg=0.1,
)
pgd_tv = deepinv.optim.PGD(
    data_fidelity=data_fidelity,
    prior=tv,
    lambda_reg=0.05,
)
y = physics(img)
x_noprior = gd_noprior(y, physics=physics)
x_tik = gd_tik(y, physics=physics)
x_tv = pgd_tv(y, physics=physics)

psnr = deepinv.loss.metric.PSNR()
deepinv.utils.plot(
    [img, y, x_noprior, x_tik, x_tv],
    titles=[
        r"x",
        r"y",
        r"g(x) = 0",
        r"$g(x) = \|x\|^2$",
        r"$g(x) = \operatorname{TV}(x)$",
    ],
    subtitles=[
        "PSNR:",
        f"{psnr(img, y).item():.2f} dB",
        f"{psnr(img, x_noprior).item():.2f} dB",
        f"{psnr(img, x_tik).item():.2f} dB",
        f"{psnr(img, x_tv).item():.2f} dB",
    ],
    fontsize=20,
    figsize=(12, 3),
    save_fn="../assets/img/section2/smooth_nonsmooth_reconstruction.pdf",
)

# %%
