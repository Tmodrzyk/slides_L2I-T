# %%
import deepinv as dinv
import torch
from deepinv.utils import load_url_image

import matplotlib.pyplot as plt

# Set device
device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

# Load the butterfly test image
x = dinv.utils.load_example(
    "butterfly.png", device=device, resize_mode="resize", img_size=256
)
# %%


# Define Gamma noise physics (multiplicative noise)
# Gamma noise with a given gain parameter (l controls noise level, lower = more noise)
gamma_noise_level = 2.0  # shape parameter of the Gamma distribution

physics = dinv.physics.Tomography(
    angles=180,
    img_width=256,
    normalize=True,
    noise_model=dinv.physics.GaussianNoise(sigma=20 / 255),
    device=device,
)
# Apply the gamma noise to the image
y = physics(x)
x_dagger = physics.A_dagger(y, max_iter=100)

# Load pretrained DRUNet denoiser
# DRUNet expects an estimate of the noise level as input
# For Gamma noise with parameter l, the variance is 1/l and mean is 1,
# so we approximate the equivalent Gaussian sigma

# Load DRUNet model
denoiser = dinv.models.DRUNet(
    in_channels=3, out_channels=3, pretrained="download", device=device
)

# Denoise the noisy image
with torch.no_grad():
    x_hat = denoiser(x_dagger, sigma=50 / 255)
    y_hat = denoiser(y, sigma=50 / 255)

# Clip to valid range
x_hat = x_hat.clamp(0, 1)


# Plot results
imgs = [y, y_hat, physics.A_dagger(y), physics.A_dagger(y_hat)]
dinv.utils.plot(imgs, figsize=(16, 10))
plt.show()
print(y.min(), y.max())

# %%
