import numpy as np
import random
from matplotlib import pyplot as plt
import time
from scipy.stats.mstats import mquantiles

def Bootstrap_Romano(Y, cliques, clique_size, K, E_nu, J, lambdaHat, alpha, m, geometry, kappa, rate, zeta):
    s = 1000  # number of bootstrap simulations
    temp = np.zeros(s)
    lambdaStar = np.zeros(s)
    print("these are the cliques")
    print(cliques)
    print("This is mstar")
    print(m)
    print(Y)
    print(clique_size)

    for i in range(s):
        pStar = np.zeros((K, K))
        indices_1 = np.random.randint(1, clique_size, size = m)
        indices_2 = np.random.randint(1, clique_size, size = m)
        for k in range(K):
             for kPrime in range(K):
                 if k < kPrime:
                     Y_sub = Y[np.ix_([np.int(j) for j in cliques[k,:]], [np.int(j) for j in cliques[kPrime,:]])]
                     pStar[k,kPrime] = np.maximum(1/clique_size**2, np.mean(Y_sub[np.ix_(indices_1, indices_2)]))
        pStar = pStar + np.transpose(pStar)
        if E_nu == 0:
            E_nu = 1
        np.fill_diagonal(pStar, E_nu)
        dStar = -np.log(pStar/E_nu)
        dStar = dStar/zeta

        if geometry == "E":
            fStar = -0.5 * J @ np.square(dStar) @ J
            lambdaStar[i] = np.linalg.eigvalsh(fStar)[0] - lambdaHat

        if geometry == "S":
            cStar = np.cos(dStar * np.sqrt(kappa))
            lambdaStar[i] = np.linalg.eigvalsh(cStar)[0] - lambdaHat

        if geometry == "H":
            cStar = np.cosh(np.sqrt(-kappa) * dStar)
            lambdaStar[i] = np.linalg.eigvalsh(cStar)[-2] - lambdaHat

    if geometry == "E":
        c_alpha = mquantiles(m**(2*rate) * lambdaStar, prob = alpha) # compute quantile of bootstrapped distribution.
        pvalue = len([i for i in range(s) if m**(2*rate) * lambdaStar[i] < clique_size ** (2 * rate) * lambdaHat])/float(s)
        return([c_alpha/(clique_size**(2*rate)) > lambdaHat, pvalue])
    if geometry == "S":
        c_alpha = mquantiles(m**(2*rate) * lambdaStar, prob = alpha) # compute quantile of bootstrapped distribution.
        pvalue = len([i for i in range(s) if m**(2*rate) * lambdaStar[i] < clique_size ** (2 * rate) * lambdaHat])/float(s)
        return([c_alpha/(clique_size**(2*rate)) > lambdaHat, pvalue])
    if geometry == "H":
        c_alpha = mquantiles(m**(2*rate) * lambdaStar, prob = 1 - alpha) # compute quantile of bootstrapped distribution.
        pvalue = len([i for i in range(s) if m**(2*rate) * lambdaStar[i] > clique_size ** (2 * rate) * lambdaHat])/float(s)
        return([lambdaHat > c_alpha/clique_size**(2*rate), pvalue])
