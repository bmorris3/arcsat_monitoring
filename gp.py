import matplotlib.pyplot as plt
import numpy as np

times, lc = np.loadtxt('lc.txt', unpack=True)
import george
from george import kernels

k = kernels.CosineKernel(log_period=np.log(1.407)) + kernels.Matern32Kernel(1)
# k = kernels.ExpSine2Kernel(gamma=100, log_period=np.log(1.407))
k.freeze_parameter('k1:log_period')
gp = george.GP(k, mean=1.0)
gp.compute(times - times.mean())

import scipy.optimize as op

# Define the objective function (negative log-likelihood in this case).
def nll(p):
    # Update the kernel parameters and compute the likelihood.
    gp.kernel.set_parameter_vector(p)
    ll = gp.log_likelihood(lc, quiet=True)

    # The scipy optimizer doesn't play well with infinities.
    return -ll if np.isfinite(ll) else 1e25

# And the gradient of the objective function.
def grad_nll(p):
    # Update the kernel parameters and compute the likelihood.
    gp.kernel.set_parameter_vector(p)
    return -gp.grad_log_likelihood(lc, quiet=True)

# You need to compute the GP once before starting the optimization.
gp.compute(times - times.mean())

# Run the optimization routine.
p0 = gp.kernel.get_parameter_vector()
results = op.minimize(nll, p0, jac=grad_nll)

# Update the kernel and print the final log-likelihood.
gp.kernel.set_parameter_vector(results.x)

x = np.linspace(min(times), max(times), 2000)
mu, cov = gp.predict(lc, x)
std = np.sqrt(np.diag(cov))

plt.scatter(times, lc, marker='.', color='k')
plt.plot(x, mu, 'r')
plt.show()