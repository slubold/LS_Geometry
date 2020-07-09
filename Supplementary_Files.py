import numpy as np
import random
np.seterr(divide='ignore')


def EstimatePhat(k, kPrime, K, n,c,Y):
      G = {}
      term = 0
      for i in range(K):
          G[i] = []
      for i in range(n):
          G[c[i]].append(i)
          if c[i] == k:
              term = term + np.sum(Y[i,]*(c == kPrime) )
      if k != kPrime:
          return([term,len(G[k])*len(G[kPrime])])
      if k == kPrime:
          return([term/2,n*(n-1)/2])

def approximateEnu(n, K, degCutoff, Y, clique, clique_size):
    nodesPerClique = np.int(n/float(K))
    approx = np.zeros(K)
    for indexClique in range(K):
        tempIndices = nodesPerClique * indexClique + np.arange(nodesPerClique)
        YtempClique = Y[np.ix_([np.int(j) for j in clique[indexClique,:]], tempIndices)]
        nodes = [nodesPerClique * indexClique + j for j in range(nodesPerClique) if np.sum(YtempClique[:,j]) == degCutoff if np.sum(YtempClique[:,j]) < clique_size if j not in clique[indexClique,:]]
        if len(nodes) < 3:
            approx[indexClique] = 0
        if len(nodes) >= 3:
            approx[indexClique] = np.sum(Y[np.ix_(nodes, nodes)])/(len(nodes)*(len(nodes)-1))
    if np.sum(approx) == 0:
        return(0)
    else:
        return(np.sum(approx)/float(len([j for j in range(K) if approx[j] > 0])))


def PickCliquesFeasible(n, K, Y, clique, numSamples, clique_size):
    objFun = np.zeros(numSamples)
    indices = np.zeros((numSamples, K))
    cliques_saved = [[] for i in range(numSamples)]
    dHat_saved = [[] for i in range(numSamples)]
    Enu_saved = np.zeros(numSamples)
    for index in range(numSamples):
        pHat = np.zeros((K, K))
        tempSum = 0
        a = np.ones(K)
        while len(a) > len(np.unique(a)):                                       # this while loop ensures that all elements of "a" are distinct.
            a = [random.randint(1, len(clique) - 1) for j in range(K)]          # Sample a subset of the cliques with indices in "a".

        for alpha in range(K):
            for beta in range(K):
                if alpha < beta:
                    tempSum = tempSum + len([1 for j in clique[a[alpha]] if j in clique[a[beta]]])
        cliques_sampled = np.array([clique[np.int(a[j])] for j in range(K)])
        indices[index,:] = a                                                    # save indices of sampled cliques.
        cliques_saved[index] = cliques_sampled                                  # save sampled cliques
        tempIndices = [int(b) for b in cliques_sampled.flatten()]
        A_c = Y[np.ix_(tempIndices, tempIndices)]                               # adjacency matrix for subgraph induced by cliques.
        c_clique = np.array([0 for a in range(K*clique_size)])
        for i in range(K*clique_size):
            c_clique[i] = np.floor(np.float(i)/clique_size)
        pHatVec = [EstimatePhat(i,j,K,clique_size * K,c_clique,A_c)[0]/EstimatePhat(i,j,K,clique_size*K,c_clique,A_c)[1] for i in range(K) for j in range(K) if i < j]
        pHat[np.triu_indices(K, 1)] = pHatVec
        pHat = pHat + np.transpose(pHat)
        E_nu = np.mean([j for j in [approximateEnu(n, K, i, Y, cliques_sampled, clique_size) for i in [clique_size - 1]] if j > 0])
        if np.isnan(E_nu):                                                      # If E_nu is nan, set E_nu = 1 (so no FE in model)
            E_nu = 1
        Enu_saved[index] = E_nu
        np.fill_diagonal(pHat, E_nu)
        dHat = np.zeros((K, K))
        for i in range(K):
            for j in range(K):
                if i != j:
                    dHat[i,j] = -np.log(pHat[i,j]/float(E_nu))
        dHat_saved[index] = dHat                                                # save dHat
        objFun[index] = tempSum + 100*(0 in pHat)
    index_optim = np.argmin(objFun)                                             # find minimizer of objFun
    return([indices[index_optim,:], np.min(objFun) < 100, dHat_saved[index_optim], cliques_saved[index_optim], Enu_saved[index_optim]])
