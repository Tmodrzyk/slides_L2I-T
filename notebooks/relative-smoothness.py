# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(context="talk", style="whitegrid", font_scale=1)


def f(x, y=1.0):
    return x - y * np.log(x)


def grad_f(x, y=1.0):
    return 1.0 - y / x


def phi(x):
    return -np.log(x)


def D_phi(y, x):
    return phi(y) - phi(x) + (y - x) / x  # Burg Bregman


y_obs = 1.0
xk = 0.5
tau = 0.2

L_tight = y_obs
L = 3 * L_tight  # inflate to make the gap visible (try 1.2–3.0)

x = np.linspace(1e-3, 1.2, 800)

fx = f(x, y_obs)
quad = f(xk, y_obs) + grad_f(xk, y_obs) * (x - xk) + (1.0 / (2.0 * tau)) * (x - xk) ** 2
breg = f(xk, y_obs) + grad_f(xk, y_obs) * (x - xk) + L * D_phi(x, xk)

fontsize = 30
colors = sns.color_palette("deep")

plt.figure(figsize=(8, 4))

# Curves
plt.plot(x, fx, linewidth=2, label=r"$f$", color=colors[0])
plt.plot(x, quad, "--", linewidth=2, label=r"$L$-smoothness", color=colors[3])
plt.plot(x, breg, "--", linewidth=2, label=r"$L$-relative smoothness", color=colors[2])

# Shade invalid Euclidean region: where quad is NOT an upper bound
invalid = quad < fx
plt.fill_between(
    x,
    quad,
    fx,
    where=invalid,
    interpolate=True,
    alpha=0.20,
    color=colors[3],
)

# Optional: mark intersections (nice for posters)
# Find sign changes of (quad - fx)
diff = quad - fx
cross_idx = np.where(np.sign(diff[:-1]) != np.sign(diff[1:]))[0]

plt.ylim(0.5, 6)

ax = plt.gca()
ax.tick_params(labelbottom=False, labelleft=False)

plt.legend()  # poster: keep legend readable but not huge
plt.savefig(
    "../assets/img/section4/relative-smoothness.pdf", dpi=96, bbox_inches="tight"
)
plt.show()


# %%
