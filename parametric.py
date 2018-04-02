import matplotlib.pyplot as plt
import numpy as np

times, lc = np.loadtxt('lc.txt', unpack=True)

from scipy.optimize import fmin_l_bfgs_b

# plt.plot(times, lc)

splits = np.argwhere(np.diff(times) > 0.1).T[0] + 1
splits = np.concatenate([[0], splits, [len(lc)]])
for i in range(len(splits)-1):
#     print(splits[i], splits[i+1])
    plt.plot(times[splits[i]:splits[i+1]], lc[splits[i]:splits[i+1]], '.', color='gray')
# plt.show()

period = 1.407#/2

def model(p, x):
    amp, x0, offset1, offset2, offset3, offset4, offset5 = p

    offsets = np.zeros_like(x)
    for i, offset in zip(range(len(splits)-1),
                         [offset1, offset2, offset3, offset4, offset5]):
        offsets[splits[i]:splits[i+1]] = offset

    return amp * np.cos(2*np.pi/period * (x - x0)) + 1 + offsets

def residuals(p, x):
    return np.sum((model(p, x) - lc)**2)

bounds = [[0, 0.01], [0, 5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5]]
initp = [0.001, 0.1, 0, 0, 0, 0, 0]
result = fmin_l_bfgs_b(residuals, initp, bounds=bounds, args=(times,),
                       approx_grad=True)

print(result[0])

# plt.scatter(times, lc, marker='.')
# plt.plot(times, model(result[0], times), color='r')
# plt.show()
for i, offset in zip(range(len(splits)-1), result[0][2:]):
    plt.plot(times[splits[i]:splits[i+1]], lc[splits[i]:splits[i+1]] - offset, '.')
    plt.scatter(np.median(times[splits[i]:splits[i+1]]), np.median(lc[splits[i]:splits[i+1]] - offset), marker='s')

x = np.linspace(times.min(), times.max(), 1000)
plt.plot(x, result[0][0] * np.cos(2*np.pi/period * (x - result[0][1])) + 1)

plt.show()