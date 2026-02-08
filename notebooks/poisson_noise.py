# %%

import numpy as np
import deepinv as dinv

import matplotlib.pyplot as plt

# Load image
img = dinv.utils.load_image(
    "../assets/img/section2/smiley.png",
    grayscale=True,
    img_size=256,
    resize_mode="resize",
).clamp(min=0.0)

# Create noise operators
gaussian_noise = dinv.physics.GaussianNoise(sigma=0.8)
poisson_noise = dinv.physics.PoissonNoise(gain=1.0, normalize=True)

# Apply noise
y_gaussian = gaussian_noise(img)
y_poisson = poisson_noise(img)

# Plot results
fig = dinv.utils.plot(
    [img, y_gaussian, y_poisson],
    titles=[
        "Original Image",
        "Gaussian Noise",
        "Poisson Noise",
    ],
    figsize=(8, 4),
    fontsize=20,
    rescale_mode="clip",
    vmin=0.0,
    vmax=2,
    return_fig=True,
    show=False,
)

fig.savefig("../assets/img/section2/poisson_noise.pdf", dpi=300, bbox_inches="tight")

# %%
