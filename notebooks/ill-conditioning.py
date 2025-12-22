# %%
import deepinv as dinv

device = "cuda"
x = dinv.utils.demo.load_image(
    "../assets/img/section1/slide1/moon.png",
    resize_mode="crop",
    img_size=(256, 256),
    grayscale=True,
    device=device,
)

dinv.utils.plot(x)

kernel = dinv.physics.blur.gaussian_blur(sigma=2.0)
noise = dinv.physics.GaussianNoise(sigma=3 / 255)
physics = dinv.physics.Blur(
    filter=kernel, noise_model=noise, padding="circular", device=device
)
y_noisy = physics(x)
y_noiseless = physics.A(x)

pseudoinverse_noisy = physics.A_dagger(y_noisy)
pseudoinverse_noiseless = physics.A_dagger(y_noiseless)
psnr = dinv.metric.PSNR()

fig = dinv.utils.plot(
    [x, y_noiseless, y_noisy, pseudoinverse_noiseless, pseudoinverse_noisy],
    titles=[
        "Ground-truth $x$",
        "Noiseless blurry $\\tilde{y}$",
        "Noisy blurry $y$",
        "Pseudoinverse\non noiseless data $A^\\dagger \\tilde{y}$",
        "Pseudoinverse\non noisy data $A^\\dagger y$",
    ],
    subtitles=[
        f"PSNR:",
        f"{psnr(x, y_noiseless).item():.2f} dB",
        f"{psnr(x, y_noisy).item():.2f} dB",
        f"{psnr(x, pseudoinverse_noiseless).item():.2f} dB",
        f"{psnr(x, pseudoinverse_noisy).item():.2f} dB",
    ],
    figsize=(12, 4),
    return_fig=True,
)
fig.savefig("../assets/img/section1/slide4/moon-pseudoinverse.png", dpi=300)

# %%
x = dinv.utils.demo.load_example(
    "SheppLogan.png",
    img_size=(256, 256),
    grayscale=True,
    device=device,
    resize_mode="resize",
)
physics = dinv.physics.TomographyWithAstra(
    img_size=(256, 256),
    num_angles=180,
    num_detectors=512,
    geometry="parallel",
    device=device,
    normalize=True,
    noise_model=noise,
)
y_noisy = physics(x)
y_noiseless = physics.A(x)
pseudoinverse_noisy = physics.A_dagger(y_noisy)
pseudoinverse_noiseless = physics.A_dagger(y_noiseless)

fig = dinv.utils.plot(
    [x, pseudoinverse_noiseless, pseudoinverse_noisy],
    titles=[
        "Ground-truth $x$",
        "Pseudoinverse\non noiseless data $A^\\dagger \\tilde{y}$",
        "Pseudoinverse\non noisy data $A^\\dagger y$",
    ],
    subtitles=[
        f"PSNR:",
        f"{psnr(x, pseudoinverse_noiseless).item():.2f} dB",
        f"{psnr(x, pseudoinverse_noisy).item():.2f} dB",
    ],
    figsize=(12, 4),
    return_fig=True,
)

fig.savefig("../assets/img/section1/slide4/tomo-pseudoinverse.png", dpi=300)
# %%
