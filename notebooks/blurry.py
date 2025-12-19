# %%
import deepinv as dinv
from pathlib import Path
from PIL import Image
import numpy as np
import torch

x = dinv.utils.demo.load_image(
    "../assets/img/section1/slide1/moon.jpg",
    resize_mode="crop",
    img_size=(800, 800),
    grayscale=True,
)
kernel = dinv.physics.blur.gaussian_blur(sigma=10.0)
physics = dinv.physics.Blur(filter=kernel)
y = physics(x)
dinv.utils.plot([x, y])


def save_png(img, path):
    # handle torch tensors
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy().squeeze()
    img = np.asarray(img)

    # convert channels-first (C,H,W) to (H,W,C)
    if img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[0] != img.shape[2]:
        img = np.transpose(img, (1, 2, 0))

    # single channel -> mode L, else RGB
    if img.ndim == 2:
        mode = "L"
    elif img.ndim == 3 and img.shape[2] == 1:
        img = img[:, :, 0]
        mode = "L"
    else:
        mode = "RGB"

    # scale to 0-255 uint8
    img = np.clip(img, 0.0, 1.0)
    img = (img * 255).astype(np.uint8)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img, mode=mode).save(path, format="PNG")


save_png(x, "../assets/img/section1/slide1/moon.png")
save_png(y, "../assets/img/section1/slide1/moon_blur.png")
# %%
