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
    noise_model=deepinv.physics.PoissonNoise(gain=1 / 5, normalize=True),
    device=device,
)
# Create batch of images
imgs = torch.stack([img, img_leaves, img_starfish], dim=1)[0]
y = physics(imgs)
deepinv.utils.plot(
    [imgs, y],
    rescale_mode="clip",
    vmin=0,
    vmax=1.0,
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
    lambda_reg=0.09,
)
dpir = deepinv.optim.DPIR(
    sigma=55 / 255,
    denoiser=deepinv.models.DRUNet(device=device),
    device=device,
)

# x_noprior = gd_noprior(y, physics=physics)
x_tv = pgd_tv(y, physics=physics)
x_dpir = dpir(y, physics=physics)

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
    rescale_mode="clip",
    vmin=0,
    vmax=1.0,
)

fig.savefig(
    "../assets/img/section2/regularization_tv_pnp_poisson_mismatched.pdf",
    dpi=300,
    bbox_inches="tight",
)
# %%
