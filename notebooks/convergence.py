# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(context="talk", style="whitegrid")


# Objective
def f(x):
    return x**2


K = 50
k = np.arange(1, K + 1)

# Sequence A: f(x_k) converges, but x_k does NOT converge (oscillates)
# x_k alternates near +1 and -1, so x_k has no limit, but f(x_k) -> 1
xA = (-1.0) ** k + 1.0 / k
fA = f(xA)

# Sequence B: x_k converges (hence f(x_k) converges for continuous f)
xB = 1.0 / k
fB = f(xB)
# --- Plot iterates (pointwise behavior) and objective values
fig, axes = plt.subplots(2, 1, figsize=(10, 10))

axes[0].plot(
    k, xB, marker="o", markersize=5, linewidth=2, label=r"$x^{(k)} = 1/k \to 0$"
)
axes[0].plot(
    k,
    xA,
    marker="o",
    markersize=5,
    linewidth=2,
    label=r"$x^{(k)} = (-1)^k + 1/k$ (no limit)",
)
axes[0].set_ylabel("Iterate value $x^{(k)}$")
axes[0].legend(framealpha=1.0)

axes[1].plot(
    k,
    fB,
    marker="o",
    markersize=5,
    linewidth=2,
    label=r"$f(x^{(k)})\to 0$ with $x^{(k)}\to 0$",
)
axes[1].plot(
    k,
    fA,
    marker="o",
    markersize=5,
    linewidth=2,
    label=r"$f(x^{(k)})\to 1$ while $x^{(k)}$ oscillates",
)

axes[1].set_xlabel("Iteration $k$")
axes[1].set_ylabel("Objective value $f(x^{(k)})$")
axes[1].legend(framealpha=1.0)

plt.tight_layout()
plt.savefig("../assets/img/section4/convergence_illustration.pdf")
plt.show()

# %%
