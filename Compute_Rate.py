import numpy as np
import numpy.linalg
import scipy.spatial
from scipy.stats.mstats import mquantiles

def Estimate_Rate(Y, K, clique_size, E_nu, cliques, lambdaHat, geometry, kappa = 0):
    J = np.identity(K) - 1/float(K) * np.outer(np.ones(K), np.ones(K))
    b = [j**2 for j in [3, 5, 7]]
    s = np.linspace(0.01, 0.49, 10)
    t = np.linspace(0.51, 0.99, 10)

    def Return_Inv_CDF(Y, b, s_i, t_i):
        s = 1000 # number of bootstrap samples
        lambdaStar = np.zeros(s)
        for i in range(s):
            pStar = np.zeros((K, K))
            indices = np.random.randint(1, clique_size**2, size = b)
            for k in range(K):
                 for kPrime in range(K):
                     if k < kPrime:
                         temp = np.matrix.flatten(Y[np.ix_([np.int(j) for j in cliques[k,:]], [np.int(j) for j in cliques[kPrime,:]])])
                         pStar[k,kPrime] = np.maximum(1/float(clique_size**2), np.mean(temp[indices]))
            pStar = pStar + np.transpose(pStar)
            np.fill_diagonal(pStar, E_nu)
            dStar = -np.log(pStar/E_nu)
            if geometry == 'Euclidean':
                wStar = -1/2 * J @ np.square(dStar) @ J
                lambdaStar[i] = np.linalg.eigvalsh(wStar)[0] - lambdaHat
            if geometry == 'Spherical':
                wStar = np.cos(np.sqrt(kappa) * dStar)
                lambdaStar[i] = np.linalg.eigvalsh(wStar)[0] - lambdaHat
            if geometry == 'Hyperbolic':
                wStar = np.cosh(np.sqrt(-kappa) * dStar)
                lambdaStar[i] = np.linalg.eigvalsh(wStar)[-2] - lambdaHat
        c_alpha_s = mquantiles(lambdaStar, prob = s_i)
        c_alpha_t = mquantiles(lambdaStar, prob = t_i)
        return(np.log(c_alpha_t - c_alpha_s))

    Y_Matrix = np.zeros((len(b), len(s)))
    for index1 in range(len(b)):
        for index2 in range(len(s)):
            Y_Matrix[index1,index2] = Return_Inv_CDF(Y, b[index1], s[index2], t[index2])
    Y_Dot = np.mean(Y_Matrix, axis = 1)
    Y_bar = np.mean(Y_Matrix)
    log_bar = np.mean([np.log(b[i]) for i in range(len(b))])
    num = np.sum((Y_Dot - Y_bar)*(np.log(b) - log_bar))
    denom = np.sum((np.log(b) - log_bar)**2)
    rate = -num/denom #compute rate
    return(rate)
