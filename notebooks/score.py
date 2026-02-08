# %%
import numpy as np
from matplotlib.colors import LogNorm
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

sns.set_theme(context="talk", style="whitegrid", palette="deep")

import matplotlib.pyplot as plt

# ============================================================
# 1. Define a Gaussian Mixture Model in 2D
# ============================================================


class GaussianMixture2D:
    """Simple 2D Gaussian mixture with isotropic components."""

    def __init__(self, means, stds, weights):
        """
        means: (K, 2) array of component means
        stds: (K,) array of component standard deviations (isotropic)
        weights: (K,) array of mixture weights (must sum to 1)
        """
        self.means = np.array(means, dtype=np.float64)
        self.stds = np.array(stds, dtype=np.float64)
        self.weights = np.array(weights, dtype=np.float64)
        self.K = len(weights)

    def smoothed_component_std(self, sigma):
        """After adding N(0, sigma^2 I) noise, each component N(mu_k, s_k^2 I)
        becomes N(mu_k, (s_k^2 + sigma^2) I)."""
        return np.sqrt(self.stds**2 + sigma**2)

    def smoothed_log_density(self, y, sigma):
        """Compute log p_sigma(y) for the smoothed distribution."""
        # y: (..., 2)
        s = self.smoothed_component_std(sigma)  # (K,)
        log_components = np.zeros(y.shape[:-1] + (self.K,))
        for k in range(self.K):
            diff = y - self.means[k]  # (..., 2)
            log_components[..., k] = (
                np.log(self.weights[k])
                - np.log(2 * np.pi * s[k] ** 2)
                - 0.5 * np.sum(diff**2, axis=-1) / s[k] ** 2
            )
        # log-sum-exp for numerical stability
        max_log = np.max(log_components, axis=-1, keepdims=True)
        log_density = max_log[..., 0] + np.log(
            np.sum(np.exp(log_components - max_log), axis=-1)
        )
        return log_density

    def smoothed_density(self, y, sigma):
        """Compute p_sigma(y)."""
        return np.exp(self.smoothed_log_density(y, sigma))

    def score(self, y, sigma):
        """
        Compute the score ∇_y log p_σ(y) in closed form.

        ∇_y log p_σ(y) = Σ_k w_k(y) * (μ_k - y) / (s_k² + σ²)

        where w_k(y) = π_k N(y; μ_k, (s_k²+σ²)I) / p_σ(y)
        are the posterior responsibilities.
        """
        s = self.smoothed_component_std(sigma)  # (K,)

        # Compute unnormalized log responsibilities
        log_resps = np.zeros(y.shape[:-1] + (self.K,))
        for k in range(self.K):
            diff = y - self.means[k]
            log_resps[..., k] = (
                np.log(self.weights[k])
                - np.log(2 * np.pi * s[k] ** 2)
                - 0.5 * np.sum(diff**2, axis=-1) / s[k] ** 2
            )

        # Normalize responsibilities (softmax)
        max_log = np.max(log_resps, axis=-1, keepdims=True)
        resps = np.exp(log_resps - max_log)
        resps = resps / np.sum(resps, axis=-1, keepdims=True)  # (..., K)

        # Score = sum_k resp_k * (mu_k - y) / (s_k^2 + sigma^2)
        # which equals sum_k resp_k * (mu_k - y) / s_smoothed_k^2
        score_val = np.zeros_like(y)
        for k in range(self.K):
            diff = self.means[k] - y  # (..., 2)
            score_val += resps[..., k : k + 1] * diff / s[k] ** 2

        return score_val

    def mmse_denoiser(self, y, sigma):
        """
        MMSE denoiser E[x | y] via Tweedie's formula:
            E[x | y] = y + σ² * ∇_y log p_σ(y)

        Equivalently, this is the posterior mean:
            E[x | y] = Σ_k w_k(y) * (σ² μ_k + s_k² y) / (s_k² + σ²)
                      = Σ_k w_k(y) * posterior_mean_k
        """
        return y + sigma**2 * self.score(y, sigma)

    def sample(self, n):
        """Sample n points from the clean distribution."""
        components = np.random.choice(self.K, size=n, p=self.weights)
        samples = np.zeros((n, 2))
        for k in range(self.K):
            mask = components == k
            nk = np.sum(mask)
            samples[mask] = self.means[k] + self.stds[k] * np.random.randn(nk, 2)
        return samples


# ============================================================
# 2. Set up the mixture and noise level
# ============================================================

# A mixture of 5 Gaussians arranged in interesting pattern
means = [
    [-2.0, 2.0],
    [2.0, -2.0],
]

stds = [1, 1]
weights = [0.5, 0.5]

gmm = GaussianMixture2D(means, stds, weights)

# ============================================================
# 3. Visualization
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(20, 10))

sigma = 1.0

# Grid for density and vector field
grid_range = 5
n_density = 200
n_arrows = 20

x_lin = np.linspace(-grid_range, grid_range, n_density)
y_lin = np.linspace(-grid_range, grid_range, n_density)
X, Y = np.meshgrid(x_lin, y_lin)
grid_points = np.stack([X, Y], axis=-1)  # (n, n, 2)

x_arrows = np.linspace(-grid_range, grid_range, n_arrows)
y_arrows = np.linspace(-grid_range, grid_range, n_arrows)
Xa, Ya = np.meshgrid(x_arrows, y_arrows)
arrow_points = np.stack([Xa, Ya], axis=-1)

# --- Custom colormap: white at 0, then colored ---

cmap_custom = LinearSegmentedColormap.from_list(
    "white_to_color", ["white", "royalblue", "darkblue", "black"]
)
cmap_custom = "plasma"

# --- Left: True (unsmoothed) distribution, no vector field ---
ax = axes[0]
clean_density = gmm.smoothed_density(grid_points, 0.01)
# ax.contourf(X, Y, clean_density, levels=50, cmap=cmap_custom)
ax.contour(X, Y, clean_density, cmap="plasma", levels=50, linewidths=0.8, alpha=0.9)

ax.set_xlim(-grid_range, grid_range)
ax.set_ylim(-grid_range, grid_range)
ax.set_xticks([])
ax.set_yticks([])
ax.set_aspect("equal")
ax.set_facecolor("white")

# --- Right: Smoothed distribution + score vector field + denoiser point ---
ax2 = axes[1]

density = gmm.smoothed_density(grid_points, sigma)
# ax2.contourf(X, Y, density, levels=50, cmap=cmap_custom)
ax2.contour(X, Y, density, cmap="plasma", levels=10, linewidths=0.8, alpha=1)

# Score (vector field) in black
score_vals = gmm.score(arrow_points, sigma)
norms = np.sqrt(np.sum(score_vals**2, axis=-1, keepdims=True))
scale_factor = np.log1p(norms) / (norms + 1e-10)
scaled = score_vals * scale_factor

ax2.quiver(
    Xa,
    Ya,
    scaled[..., 0],
    scaled[..., 1],
    color=sns.color_palette("deep")[1],
    alpha=0.7,
    scale=30,
    width=0.003,
    headwidth=4,
    headlength=5,
)

# --- Big point with denoiser direction in red ---
y_point = np.array([[2.5, 1]])
denoised_point = gmm.mmse_denoiser(y_point, sigma)


ax2.set_xlim(-grid_range, grid_range)
ax2.set_ylim(-grid_range, grid_range)
ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_aspect("equal")
ax2.set_facecolor("white")

fig.patch.set_facecolor("white")
plt.tight_layout()
plt.savefig(
    "../assets/img/section2/score.svg", dpi=96, bbox_inches="tight", facecolor="white"
)
plt.show()

# %%
