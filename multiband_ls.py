import h5py
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
paths = glob('/Users/bmmorris/data/freckles/*.fits')
from astropy.io import fits
spectrum_times = [fits.getheader(p)['JD'] for p in paths]

times_g, lc_g = np.loadtxt('lc.txt', unpack=True)
times_u, lc_u = np.loadtxt('lc_u.txt', unpack=True)
err = np.std(lc_g)/2

from gatspy import periodic

# Fit the Lomb-Scargle model
model = periodic.LombScargleMultiband(Nterms_base=3)#, fit_period=False)
model.fit(np.concatenate([times_g, times_u]), np.concatenate([lc_g, lc_u]),
          filts=np.concatenate([np.zeros_like(times_g), np.ones_like(times_u)]))

P_rot = 1.407 # (Nielson 2013)
model.optimizer.period_range = (1.3, 1.5)

# Predict on a regular phase grid
period = model.best_period
tfit = np.linspace(0, period, 1000)
magfit_g = model.predict(tfit, filts=0)
magfit_u = model.predict(tfit, filts=1)

# Plot the results
phase_g = (times_g / period) % 1
phasefit_g = (tfit / period)

# Plot the results
phase_u = (times_u / period) % 1
phasefit_u = (tfit / period)

print(period)

fig, ax = plt.subplots(3, 2, figsize=(8, 8), sharex='col', sharey='row')#, sharex='col')
ax[0, 1].scatter(phase_g, lc_g, marker='.', color='gray')
ax[0, 1].plot(phasefit_g, magfit_g, '-', color='r')
ax[0, 0].set(ylabel="SDSS $g$")
ax[1, 1].scatter(phase_u, lc_u, marker='.', color='gray')
ax[1, 1].plot(phasefit_u, magfit_u, '-', color='r')
ax[2, 1].set(xlabel='Phase')

splits_g = np.argwhere(np.diff(times_g) > 0.1).T[0] + 1
splits_g = np.concatenate([[0], splits_g, [len(lc_g)]])

splits_u = np.argwhere(np.diff(times_u) > 0.1).T[0] + 1
splits_u = np.concatenate([[0], splits_u, [len(lc_u)]])

from scipy.stats import binned_statistic

for i in range(len(splits_g)-1):
    bs_g = binned_statistic(times_g[splits_g[i]:splits_g[i+1]],
                          lc_g[splits_g[i]:splits_g[i+1]], statistic='median',
                          bins=3)
    std_g = binned_statistic(times_g[splits_g[i]:splits_g[i+1]],
                           lc_g[splits_g[i]:splits_g[i+1]], statistic='std',
                           bins=3)

    bin_centers = 0.5*(bs_g.bin_edges[1:] + bs_g.bin_edges[:-1])
    ax[0, 0].errorbar(bin_centers - int(times_g.min()), bs_g.statistic,
                      std_g.statistic, fmt='s', color='k', zorder=100)

for i in range(len(splits_u)-1):
    bs_u = binned_statistic(times_u[splits_u[i]:splits_u[i+1]],
                          lc_u[splits_u[i]:splits_u[i+1]], statistic='median',
                          bins=3)
    std_u = binned_statistic(times_u[splits_u[i]:splits_u[i+1]],
                           lc_u[splits_u[i]:splits_u[i+1]], statistic='std',
                           bins=3)
    bin_centers = 0.5*(bs_u.bin_edges[1:] + bs_u.bin_edges[:-1])

    ax[1, 0].errorbar(bin_centers - int(times_g.min()), bs_u.statistic,
                      std_u.statistic, fmt='s', color='k', zorder=100)

trange = np.linspace(np.min(spectrum_times), times_g.max(), 1000)
ax[0, 0].scatter(times_g - int(times_g.min()), lc_g, marker='.', color='gray')
ax[0, 0].plot(trange - int(times_g.min()), model.predict(trange, filts=0), color='r')


ax[1, 0].scatter(times_u - int(times_u.min()), lc_u, marker='.', color='gray')
ax[1, 0].plot(trange - int(times_g.min()), model.predict(trange, filts=1), color='r')

ax[1, 0].set(ylabel='SDSS $u$')

ax[2, 0].set(xlabel='Time [JD - {0}]'.format(int(times_g.min())), ylabel='$S$-index')

for i in spectrum_times:
    ax[0, 0].axvline(i - int(times_g.min()), ls='--', color='k', alpha=0.5, lw=0.5)
    ax[1, 0].axvline(i - int(times_g.min()), ls='--', color='k', alpha=0.5, lw=0.5)



# Plot s_index measurements

import sys
sys.path.insert(0, '/Users/bmmorris/git/arces_hk')

from toolkit import json_to_stars

stars = json_to_stars('/Users/bmmorris/git/arces_hk/data/freckles/KIC9652680_apo_calibrated.json')

times = np.array([s.time.jd for s in stars])
sinds = np.array([s.s_mwo.value if not hasattr(s.s_mwo.value, '__len__') else np.mean(s.s_mwo.value) for s in stars])
sinds_err = np.array([s.s_mwo.err if not hasattr(s.s_mwo.err, '__len__') else np.std(s.s_mwo.value) for s in stars])
print(times, sinds, sinds_err)

in_time_range = ((times < trange.max()) & (times > trange.min()))

ax[2, 0].errorbar(times[in_time_range] - int(times_g.min()),
                  sinds[in_time_range], sinds_err[in_time_range],
                  fmt='o', color='k', ecolor='gray')

sind_phase = (times / period) % 1

ax[2, 1].errorbar(sind_phase[in_time_range],
                  sinds[in_time_range], sinds_err[in_time_range],
                  fmt='o', color='k', ecolor='gray')

for axis in fig.axes:
    for j in ['right', 'top']:
        axis.spines[j].set_visible(False)

    axis.grid(ls=':')

fig.savefig('plots/arcsat.pdf', bbox_inches='tight')
plt.show()