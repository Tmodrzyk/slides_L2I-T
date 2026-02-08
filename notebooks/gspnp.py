# %%
import deepinv
import torch
import seaborn as sns

sns.set_theme(style="whitegrid")

device = "cuda"
# Load example images
x = deepinv.utils.demo.load_image(
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
y = physics(x)
deepinv.utils.plot(
    [x, y],
)


# %%
# The GSPnP prior corresponds to a RED prior with an explicit `g`.
# We thus write a class that inherits from RED for this custom prior.
class GSPnP(deepinv.optim.prior.RED):
    r"""
    Gradient-Step Denoiser prior.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.explicit_prior = True

    def forward(self, x, *args, **kwargs):
        r"""
        Computes the prior :math:`g(x)`.

        :param torch.Tensor x: Variable :math:`x` at which the prior is computed.
        :return: (torch.Tensor) prior :math:`g(x)`.
        """
        return self.denoiser.potential(x, *args, **kwargs)


prior = GSPnP(denoiser=deepinv.models.GSDRUNet(pretrained="download").to(device))


# we want to output the intermediate PGD update to finish with a denoising step.
def custom_output(X):
    return X["est"][1]


data_fidelity = deepinv.optim.L2()
sigma_denoiser = 15 / 255
lambda_reg = 0.9
stepsize = 1.0
early_stop = True
max_iter = 200
backtracking = True

psnr = deepinv.loss.metric.PSNR()

# instantiate the algorithm class to solve the IP problem.
model_gspnp = deepinv.optim.PGD(
    prior=prior,
    g_first=True,
    data_fidelity=data_fidelity,
    sigma_denoiser=sigma_denoiser,
    lambda_reg=lambda_reg,
    stepsize=stepsize,
    early_stop=early_stop,
    max_iter=max_iter,
    backtracking=False,
    get_output=custom_output,
    verbose=True,
)


x_gspnp, metrics_gspnp = model_gspnp(y, physics=physics, compute_metrics=True, x_gt=x)

# %%
dpir = deepinv.optim.DPIR(
    sigma=sigma_denoiser,
    denoiser=deepinv.models.DRUNet(device=device),
    device=device,
)
x_dpir = dpir(y, physics=physics)

# %%
fig = deepinv.utils.plot(
    [x, y, x_dpir, x_gspnp],
    titles=[
        r"Ground-Truth $x$",
        r"Measurement $y$",
        r"PnP",
        r"GS-PnP",
    ],
    subtitles=[
        "PSNR:",
        f"{psnr(y, x).item():.2f} dB",
        f"{psnr(x_dpir, x).item():.2f} dB",
        f"{psnr(x_gspnp, x).item():.2f} dB",
    ],
    figsize=(12, 6),
    fontsize=20,
    return_fig=True,
)

fig.savefig(
    "../assets/img/section2/gspnp_reconstruction_starfish.svg",
    dpi=96,
    bbox_inches="tight",
)
# %%
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Plot cost
axes[0].plot(metrics_gspnp["cost"][0])
axes[0].set_xlabel("Iteration", fontsize=20)
axes[0].set_ylabel(r"$(f + g_\sigma) (x^{(k)})$", fontsize=20)
axes[0].tick_params(labelsize=20)
axes[0].grid(True)

# Plot PSNR
axes[1].plot(metrics_gspnp["psnr"][0])
axes[1].set_xlabel("Iteration", fontsize=20)
axes[1].set_ylabel("PSNR (dB)", fontsize=20)
axes[1].tick_params(labelsize=20)
axes[1].grid(True)

plt.tight_layout()
plt.savefig(
    "../assets/img/section2/gspnp_metrics_starfish.svg",
    dpi=96,
)
plt.show()

# %%
