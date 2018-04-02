import numpy as np

# Iterate!
def Init(fpix, K = 5):
    '''

    '''

    # Compute the 1st order PLD model
    fsap = np.sum(fpix, axis = 1)
    A = fpix / fsap.reshape(-1,1)
    w = np.linalg.solve(np.dot(A.T, A), np.dot(A.T, fsap))
    model = np.dot(A, w)
    fdet = fsap - model + 1

    # The data matrix
    F = np.array(fpix)

    # Estimate S from PLD de-trended flux
    S = 0.5 + np.array(fdet) / 2
    S = np.ones_like(fdet)

    # Estimate A with PCA
    X = F / S.reshape(-1, 1)
    X -= np.nanmedian(X, axis = 0)
    U, _, _ = np.linalg.svd(X)
    A = U[:,:K]
    A = np.hstack((np.ones((fpix.shape[0], 1)), A))

    return F, S, A

def Step(F, S, A):
    '''

    '''

    # Dimensions
    nt = F.shape[0]

    # Update B
    ATA = np.dot(A.T, (S ** 2)[:,None] * A)
    ATF = np.dot(A.T, S[:,None] * F)
    B = np.linalg.solve(ATA, ATF)

    # Update A
    b1 = B[0,:]
    BBT = np.dot(B[1:], B[1:].T)
    BFT = np.dot(B[1:], (F / S[:,None] - b1[None,:]).T)
    A = np.hstack((np.ones(nt).reshape(-1,1), np.linalg.solve(BBT, BFT).T))

    # Update S
    M = np.dot(A, B)
    S = np.sum(M * F, axis = 1) / np.sum(M * M, axis = 1)

    return F, S, A

import h5py
import matplotlib.pyplot as plt
import numpy as np


f = h5py.File('archive.hdf5', 'a')
dset = f['images']
times = np.loadtxt('times.txt')

target = dset[150:190, 150:190, :]
comparison1 = dset[170:200, 185:220, :]
fractional_target = (target/np.sum(target, axis=(0, 1))).reshape((40*40, -1))
fractional_comp = (comparison1/np.sum(comparison1, axis=(0, 1))).reshape((comparison1.shape[0] * comparison1.shape[1]), -1)

F, S, A = Init(fractional_target.T)
for n in range(20):
    F, S, A = Step(F, S, A)

F, S_c, A = Init(fractional_comp.T)
for n in range(20):
    F, S_c, A = Step(F, S_c, A)

plt.plot(times, S/S_c, '.')
# plt.plot(S_c)
plt.show()