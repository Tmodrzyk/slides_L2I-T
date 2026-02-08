# %%
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt

# Set seaborn talk context for presentation
sns.set_context("talk")
sns.set_style("whitegrid")


# Define a simple quadratic objective function
def f(x):
    """Convex quadratic objective function."""
    return 0.5 * x**2 - 2 * x + 3


def f_prime(x):
    """Derivative of f."""
    return x - 2


# Define a quadratic majorant that touches f at point x0
def majorant(x, x0, kappa=5.0):
    """
    Quadratic majorant g(x | x0) = f(x0) + f'(x0)(x - x0) + kappa/2 * (x - x0)^2
    """
    return f(x0) + f_prime(x0) * (x - x0) + kappa / 2.0 * (x - x0) ** 2


def majorant_minimizer(x0, kappa=5.0):
    """Minimizer of the quadratic majorant."""
    return x0 - f_prime(x0) / kappa


# Set up the domain
x = np.linspace(-2, 6, 500)
y = f(x)

# MM iterations
kappa = 2.8
x0 = -1.0  # starting point
x1 = majorant_minimizer(x0, kappa)
x2 = majorant_minimizer(x1, kappa)

colors = sns.color_palette("deep")

# --- Single big plot with two MM steps ---
fig, ax = plt.subplots(figsize=(12, 8))

ax.plot(x, y, linewidth=3, color=colors[0], label=r"$f$", zorder=5)

# First majorant around x0
mask = np.abs(x - x0) < 2.5
x_local = x[mask]
y_maj0 = majorant(x_local, x0, kappa)
y_clip0 = np.clip(y_maj0, None, np.max(y) + 2)
ax.plot(
    x_local,
    y_clip0,
    "--",
    linewidth=2.5,
    color=colors[1],
    label=r"$F(\cdot,~ x^{(0)})$",
    zorder=5,
)

# Second majorant around x1
mask = np.abs(x - x1) < 2.5
x_local = x[mask]
y_maj1 = majorant(x_local, x1, kappa)
y_clip1 = np.clip(y_maj1, None, np.max(y) + 2)
ax.plot(
    x_local,
    y_clip1,
    "--",
    linewidth=2.5,
    color=colors[2],
    alpha=1,
    label=r"$F(\cdot,~ x^{(1)})$",
    zorder=5,
)

# Mark x0
ax.plot(
    x0,
    f(x0),
    "o",
    markersize=14,
    color=colors[1],
    markeredgecolor="white",
    markeredgewidth=2,
    zorder=6,
)
ax.annotate(
    r"$x^{(0)}$",
    xy=(x0, f(x0)),
    xytext=(x0 - 0.6, f(x0) - 0.4),
    fontsize=24,
    zorder=8,
)

# Mark x1
ax.plot(
    x1,
    f(x1),
    "o",
    markersize=14,
    color=colors[2],
    markeredgecolor="white",
    markeredgewidth=2,
    zorder=6,
)
ax.plot(
    [x1, x1],
    [f(x1), majorant(x1, x0, kappa)],
    ":",
    color="gray",
    linewidth=1.5,
    zorder=3,
)
ax.annotate(
    r"$x^{(1)}$",
    xy=(x1, f(x1)),
    xytext=(x1 - 0.6, f(x1) - 0.4),
    fontsize=24,
    zorder=8,
)

# Mark x2
ax.plot(
    x2,
    f(x2),
    "s",
    markersize=14,
    color=colors[4],
    markeredgecolor="white",
    markeredgewidth=2,
    zorder=6,
)
ax.plot(
    [x2, x2],
    [f(x2), majorant(x2, x1, kappa)],
    ":",
    color="gray",
    linewidth=1.5,
    zorder=3,
)
ax.annotate(
    r"$x^{(2)}$",
    xy=(x2, f(x2)),
    xytext=(x2 - 0.6, f(x2) - 0.6),
    fontsize=24,
    zorder=8,
)


ax.legend(loc="upper right", framealpha=0.9, fontsize=24)
ax.set_ylim(0, np.max(y) + 2)
ax.set_xlim(-2, 4)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.tick_params(axis="both", length=0)

plt.tight_layout()
plt.savefig("../assets/img/section3/mm_principle.pdf", dpi=96)
plt.show()
