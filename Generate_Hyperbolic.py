import numpy as np
import matplotlib.pyplot as plt

def Generate_Hyperbolic(n, dim, scale, kappa):
    def genHyperbolic(x):
        temp = np.zeros(dim)
        temp[0:(dim-1)] = [np.random.uniform(low = -scale, high = scale, size = 1), np.random.uniform(low = -scale, high = scale, size = 1)]
        temp[-1] = np.sqrt(1/kappa + np.sum(temp[0:(dim-1)]**2))
        return(temp)

    # Generate points
    pos = np.zeros((n,dim))
    for i in range(n):
        pos[i,:] = genHyperbolic(n)

    temp = pos[0,:]
    def B(x,y):
        return( x[-1]*y[-1] - np.inner(x[0:(dim-1)], y[0:(dim-1)]) )

    D = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if i < j:
                #print(B(pos[i,:],pos[j,:])*kappa)
                D[i,j] = np.arccosh(B(pos[i,:],pos[j,:])*kappa)/np.sqrt(kappa)
    D = D + np.transpose(D)
    np.fill_diagonal(D, 0)
    return([pos, D])


def BForm(x,y, p):
    return( x[-1]*y[-1] - np.inner(x[0:(p-1)], y[0:(p-1)]))

def Make_D_FromMu_Hyperbolic(n, p, K, muVec, kappa, delta):

    V = np.zeros((n, p))
    c = np.array([0 for alphaIndex in range(n)])
    for i in range(n):
        c[i] = np.floor(np.float(i)*K/n)
        for j in range(p-1):
            V[i,j] = np.random.uniform(muVec[c[i], j]-delta, muVec[c[i],j]+delta, size = 1)
        V[i,-1] = np.sqrt(1/kappa +np.sum(V[i,0:(p-1)]**2))

    D = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if i!=j:
                D[i,j] = np.arccosh(np.maximum(1, BForm(V[i,:],V[j,:], p)*kappa))/np.sqrt(kappa)
    return(D)
