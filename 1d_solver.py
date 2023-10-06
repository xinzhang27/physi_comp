# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal


def ml2V(y):
    # return 1000 * (y - 0.5) ** 2
    return 1000 * np.exp(-(y - 0.7) ** 2 / (2 * 0.05 ** 2))


# %%
N = 2000
y = np.linspace(0, 1, N + 1)
dy = 1 / N

d = 1 / dy ** 2 + ml2V(y)[1:-1]
e = -1 / (2 * dy ** 2) * np.ones(len(d) - 1)

# %%
w, v = eigh_tridiagonal(d, e, select='i', select_range=(0, 4))

plt.plot(v.T[0] ** 2)
plt.plot(v.T[1] ** 2)
plt.plot(v.T[2] ** 2)
plt.plot(v.T[3] ** 2)
plt.plot(v.T[4] ** 2)
plt.show()

# %%
plt.bar(np.arange(5), w)
plt.ylabel(r'$mL^2 E/\hbar^2$')
plt.show()

# %%
