import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

x_dim = 302
y_dim = 302
steps = 100000

data = {
    10: 204.35791611671448,
    49: 80.81711959838867,
    81: 68.13315320014954,
    100: 59.757325649261475,
}

fig = plt.figure(dpi=400)
x = np.array(list(data.keys()))
mlups = (x_dim * y_dim * steps) / np.array(list(data.values()))

plt.plot(x, mlups, 'x', label='Measurements')
res = stats.linregress(x, mlups)
# plt.plot(x, res.intercept + res.slope * x, 'r', label='fitted line')

# ----
from scipy.optimize import curve_fit


def func(x, a, b, c):
    # return a * np.exp(-b * x) + c
    return a * np.log(b * x) + c


popt, pcov = curve_fit(func, x, mlups)
x = np.linspace(5, 100, 500)  # changed boundary conditions to avoid division by 0
plt.plot(x, func(x, *popt), 'r-', label="Fitted Curve")

# plt.plot(list(data.keys()), list(data.values()), 'x')
plt.xlabel('Number of MPI Processes')
plt.ylabel('MLUPS')
plt.legend()
plt.tight_layout()
plt.show()
fig.savefig('mlups.jpg')
