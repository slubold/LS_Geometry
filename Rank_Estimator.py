import numpy as np
import numpy.linalg
import scipy.spatial
import time
from scipy.stats import gamma
from matplotlib import pyplot as plt
import pickle

def RankEstimate(FHat, K, B, Y, clique, clique_size, E_nu, J, geometry, kappaHat):

    # First compute eigenvalues, eigenvectors of FHat
    eigvalFHat, eigvecFHat = np.linalg.eig(FHat @ FHat)
    idx = eigvalFHat.argsort()[::-1]
    eigvalFHat = eigvalFHat[idx] # sort the eigenvalues in decreasing order.
    eigvecFHat = eigvecFHat[:,idx] # sort the eigenvectors in decreasing order.

    # Bootstrap
    BStar = np.zeros((K, K, B))
    for indexB in range(B):
        PB = np.zeros((K, K))
        indices = np.random.randint(low = 0, high = clique_size, size = clique_size)
        for i1 in range(K):
            for i2 in range(K):
                if i1 < i2:
                    indices1 = clique[i1,indices]
                    indices2 = clique[i2,indices]
                    indices1 = np.array([np.int(j) for j in indices1])
                    indices2 = np.array([np.int(j) for j in indices2])
                    PB[i1, i2] = max(1/clique_size**2, np.sum(Y[np.ix_(indices1, indices2)]))
        PB = (PB + np.transpose(PB))/clique_size**2
        np.fill_diagonal(PB, E_nu)
        DB = -np.log(PB/E_nu)
        np.fill_diagonal(DB, 0)
        if geometry == 'Euclidean':
            FB = -1/2 * J @ np.square(DB) @ J
            eigval, eigvec = np.linalg.eig(FB)
        if geometry == 'Spherical':
            CB = np.cos(DB * np.sqrt(kappaHat))
            eigval, eigvec = np.linalg.eig(CB)
        if geometry == 'Hyperbolic':
            CB = np.cosh(np.sqrt(-kappaHat) * DB)
            eigval, eigvec = np.linalg.eig(CB @ CB)
        idx = eigval.argsort()[::-1]
        eigval = eigval[idx] # sort the eigenvalues in decreasing order.
        eigvec = eigvec[:,idx]
        BStar[:,:,indexB] = eigvec

    def phi(k):
        return(eigvalFHat[k]/np.sum(eigvalFHat))

    def f0(k):
        if k == 0:
            return(0)
        if k > 0:
            temp = np.zeros(B)
            for i in range(B):
                temp[i] = 1.0 - np.abs(np.linalg.det(np.transpose(eigvecFHat[:,0:k]) @ BStar[:,0:k,i]))
            return(np.mean(temp))

    def fn(j):
        return(f0(j)/(np.sum([f0(i) for i in range(K-1)])))

    def g(k):
        return(fn(k) + phi(k))
    valueC = [g(j) for j in range(K-1)] 
    return(np.argmin(valueC))
