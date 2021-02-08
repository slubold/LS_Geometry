import numpy as np
import numpy.linalg
import scipy.spatial
import matplotlib.pyplot as plt


def Make_Euclidean_D(K, n, sigma, p):
    muVec = np.zeros((K, p)) # group centers
    V = np.zeros((n, p))     # node locations
    c = np.array([0 for i in range(n)]) # group memberships

    for i in range(K):
        muVec[i,:] = np.random.multivariate_normal(np.zeros(p),  sigma * np.identity(p), 1)
    DGroups = scipy.spatial.distance_matrix(muVec, muVec)

    for i in range(n):
        c[i] = np.floor(np.float(i)*K/n)
        V[i,:] = np.random.multivariate_normal(muVec[c[i],:], sigma/K * np.identity(p), 1)



    D = scipy.spatial.distance_matrix(V, V) # Compute distance matrix
    return([DGroups, D])
