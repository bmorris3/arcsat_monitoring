import h5py
import matplotlib.pyplot as plt
import numpy as np

times, lc = np.loadtxt('lc.txt', unpack=True)

from gatspy import periodic

# Fit the Lomb-Scargle model
model = periodic.LombScargle(Nterms=5)
model.fit(times, lc)

# P_rot = 1.407 (Nielson 2013)
model.optimizer.period_range = (1.3, 1.5)

# Predict on a regular phase grid
period = model.best_period
tfit = np.linspace(0, period, 1000)
magfit = model.predict(tfit)

# Plot the results
phase = (times / period) % 1
phasefit = (tfit / period)

print(period)

fig, ax = plt.subplots()
ax.scatter(phase, lc, marker='o')
ax.plot(phasefit, magfit, '-', color='gray')
ax.set(xlabel='phase', ylabel='Flux')
ax.invert_yaxis()
plt.show()