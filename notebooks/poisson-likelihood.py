# %%

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid")

x = np.linspace(0.001, 5, 1000)
nll = x - np.log(x)
plt.figure(figsize=(8, 4))
plt.plot(x, nll, linewidth=3)
plt.xlabel("x", fontsize=20)
plt.ylabel("f(x)", fontsize=20)
plt.xlim([-0.25, 5.0])
plt.ylim([0.0, 5.0])
plt.xticks(np.arange(0, 5.5, 1))
plt.yticks(np.arange(0, 5.5, 1))
plt.savefig("../assets/img/section2/poisson_nll.pdf", dpi=300, bbox_inches="tight")
plt.show()

# %%
