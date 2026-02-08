# %%
import deepinv as dinv

device = "cuda"
x = dinv.utils.demo.load_example(
    "cat.jpg", device=device, resize_mode="resize", img_size=256
)

kernel = dinv.physics.blur.gaussian_blur(sigma=2.0)
noise = dinv.physics.GaussianNoise(sigma=5 / 255)
physics = dinv.physics.Blur(
    filter=kernel, noise_model=noise, padding="circular", device=device
)
y_noisy = physics(x)
y_noiseless = physics.A(x)

pseudoinverse_noisy = physics.A_dagger(y_noisy, max_iter=100)
pseudoinverse_noiseless = physics.A_dagger(y_noiseless, max_iter=100)
psnr = dinv.metric.PSNR()

fig = dinv.utils.plot(
    [x, y_noiseless, pseudoinverse_noiseless, y_noisy, pseudoinverse_noisy],
    figsize=(20, 12),
    rescale_mode="clip",
    vmin=0,
    vmax=1,
    return_fig=True,
)

fig.savefig(
    "../assets/img/section1/ill-conditioning-cat.svg", dpi=96, bbox_inches="tight"
)
