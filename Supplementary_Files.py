import numpy as np
import random
#from Bootstrap import *
import networkx as nx
from Estimate_Curvature import *
np.seterr(divide='ignore')
import time

def GenerateGraph(n, probs):
    Y = np.zeros((n,n))
    Y[np.triu_indices(n, 1)] = 1*np.random.uniform(size = np.int(n*(n-1)/2)) < probs # convert boolean to 1 for edge, 0 for no edge.
    return(Y + np.transpose(Y))

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

# remove one of the nodes
def removeNodes(Y, finalCliques, t, clique_size, K_vec, K_current, n, max_ell):
    G = nx.Graph(Y)
    finalCliques = finalCliques.tolist() # convert to list of lists
    print(finalCliques)

    for index in range(K_current):
        for indexClique in range(max_ell - clique_size):
            G = nx.contracted_nodes(G, finalCliques[index][indexClique], finalCliques[index][indexClique + 1])
            del finalCliques[index][indexClique + 1]

    current_cliques = np.array(finalCliques)
    pHat = np.zeros((K_current, K_current))
    tempIndices = [int(b) for b in np.array(current_cliques).flatten()]
    A_c = Y[np.ix_(tempIndices, tempIndices)] # adjacency matrix for subgraph induced by cliques.
    c_clique = np.array([0 for a in range(K_current*clique_size)])
    for i in range(K_current*clique_size):
        c_clique[i] = np.floor(np.float(i)/clique_size)
    print(c_clique)
    print([EstimatePhat(i, j, K_current, clique_size * K_current,c_clique,A_c)[0] for i in range(K_current) for j in range(K_current) if i < j])
    assert 0
    pHatVec = [EstimatePhat(i, j, K_current, clique_size * K_current,c_clique,A_c)[0]/EstimatePhat(i,j,K_current,clique_size*K_current,c_clique,A_c)[1] for i in range(K_current) for j in range(K_current) if i < j]
    pHat[np.triu_indices(K_current, 1)] = pHatVec
    pHat = pHat + np.transpose(pHat)
    print(pHat)
    E_nu = np.mean([approximateEnu(n, K_current, j, Y, current_cliques, clique_size) for j in [t]])
    if E_nu == 0:
        E_nu = 1
    np.fill_diagonal(pHat, E_nu)
    dHat = -np.log(pHat/E_nu)
    return([dHat])



def Compute_ReducedY_timed(Y, finalCliques, t, clique_size, K_vec, K_current, n, max_ell):
    time_elapsed = 0
    while time_elapsed < 100:
        time1 = time.time()
        pHat = np.zeros((K_current, K_current))
        current_cliques = np.zeros((K_current, clique_size))
        for index in range(K_current):
            random_indices = np.random.choice(max_ell, clique_size, replace = False)
            current_cliques[index, ] = finalCliques[index, random_indices]
        tempIndices = [int(b) for b in current_cliques.flatten()]
        A_c = Y[np.ix_(tempIndices, tempIndices)] # adjacency matrix for subgraph induced by cliques.
        c_clique = np.array([0 for a in range(K_current*clique_size)])
        for i in range(K_current*clique_size):
            c_clique[i] = np.floor(np.float(i)/clique_size)
        pHatVec = [EstimatePhat(i,j,K_current,clique_size* K_current,c_clique,A_c)[0]/EstimatePhat(i,j,K_current,clique_size*K_current,c_clique,A_c)[1] for i in range(K_current) for j in range(K_current) if i < j]
        pHat[np.triu_indices(K_current, 1)] = pHatVec
        pHat = pHat + np.transpose(pHat)
        E_nu = np.mean([approximateEnu(n, K_current, j, Y, current_cliques, clique_size) for j in [t]])
        if E_nu == 0:
            E_nu = 1
        np.fill_diagonal(pHat, E_nu)
        print(np.sum(pHat == 0))
        if 0 not in pHat:
            print("found this great thing")
            dHat = -np.log(pHat/E_nu)
            return([1, dHat])
        time_elapsed = time_elapsed + (time.time() - time1)
        print("this is time elapsed: {}".format(time_elapsed))
    return([0, pHat])


def Compute_ReducedY(Y, finalCliques, t, clique_size, K_vec, K_current, n, max_ell):
    while 1:
        pHat = np.zeros((K_current, K_current))
        current_cliques = np.zeros((K_current, clique_size))
        for index in range(K_current):
            random_indices = np.random.choice(max_ell, clique_size, replace = False)
            current_cliques[index, ] = finalCliques[index, random_indices]
        tempIndices = [int(b) for b in current_cliques.flatten()]
        A_c = Y[np.ix_(tempIndices, tempIndices)] # adjacency matrix for subgraph induced by cliques.
        c_clique = np.array([0 for a in range(K_current*clique_size)])
        for i in range(K_current*clique_size):
            c_clique[i] = np.floor(np.float(i)/clique_size)
        pHatVec = [EstimatePhat(i,j,K_current,clique_size* K_current,c_clique,A_c)[0]/EstimatePhat(i,j,K_current,clique_size*K_current,c_clique,A_c)[1] for i in range(K_current) for j in range(K_current) if i < j]
        pHat[np.triu_indices(K_current, 1)] = pHatVec
        pHat = pHat + np.transpose(pHat)
        E_nu = np.mean([approximateEnu(n, K_current, j, Y, current_cliques, clique_size) for j in [t]])
        if E_nu == 0:
            E_nu = 1
        np.fill_diagonal(pHat, E_nu)
        if 0 not in pHat:
            dHat = -np.log(pHat/E_nu)
            return([dHat])

def findCliques_GivenY(Y, K, n, clique_size, E_nu, numTaken, t):
    nodesPerClique = np.int(n/float(K))
    tempCliques = [[] for j in range(K)]
    finalCliques = np.zeros((K, clique_size))
    print("trying this")
    while 1:
        pHat = np.zeros((K, K))
        found_empty = False
        for indexClique in range(K):
            tempIndices = nodesPerClique*indexClique + np.arange(numTaken)
            Ytemp = Y[np.ix_(tempIndices, tempIndices)]
            cliques = list(nx.enumerate_all_cliques(nx.from_numpy_matrix(Ytemp)))
            tempCliques[indexClique] = [k for k in cliques if len(k) == clique_size]
            print(len(tempCliques[indexClique]))
            if tempCliques[indexClique] == []:
                found_empty = True
                break
        if found_empty:
            continue
        for indexClique in range(K):
            index_random = np.random.choice(len(tempCliques[indexClique]))
            finalCliques[indexClique] = [j + nodesPerClique*indexClique for j in tempCliques[indexClique][index_random]]
        tempIndices = [int(b) for b in finalCliques.flatten()]
        A_c = Y[np.ix_(tempIndices, tempIndices)] # adjacency matrix for subgraph induced by cliques.
        c_clique = np.array([0 for a in range(K*clique_size)])
        for i in range(K*clique_size):
            c_clique[i] = np.floor(np.float(i)/clique_size)
        pHatVec = [EstimatePhat(i,j,K,clique_size* K,c_clique,A_c)[0]/EstimatePhat(i,j,K,clique_size*K,c_clique,A_c)[1] for i in range(K) for j in range(K) if i < j]
        pHat[np.triu_indices(K, 1)] = pHatVec
        pHat = pHat + np.transpose(pHat)
        E_nu = np.mean([approximateEnu(n, K, j, Y, finalCliques, clique_size) for j in [t]])
        np.fill_diagonal(pHat, E_nu)
        print("this is how many pHat == 0")
        print(np.sum(pHat == 0))
        if not 0 in pHat:
            dHat = -np.log(pHat/E_nu)
            return([dHat, finalCliques]) # return adjacency matrix and pHat.

def GenerateY(K, n, probs, clique_size, E_nu, numTaken, t):
    nodesPerClique = np.int(n/float(K))
    tempCliques = [[] for j in range(K)]
    finalCliques = np.zeros((K, clique_size))
    while 1:
        Y = GenerateGraph(n, probs)
        pHat = np.zeros((K, K))
        tempCliques = [[] for j in range(K)]
        found_empty = False
        for indexClique in range(K):
            tempIndices = nodesPerClique*indexClique + np.arange(numTaken)
            Ytemp = Y[np.ix_(tempIndices, tempIndices)]
            cliques = list(nx.enumerate_all_cliques(nx.from_numpy_matrix(Ytemp)))
            tempCliques[indexClique] = [k for k in cliques if len(k) == clique_size]
            print(len(tempCliques[indexClique]))
            if tempCliques[indexClique] == []:
                found_empty = True
                break
        if found_empty:
            continue
        for indexClique in range(K):
                finalCliques[indexClique] = [j + nodesPerClique*indexClique for j in tempCliques[indexClique][0]]
        tempIndices = [int(b) for b in finalCliques.flatten()]
        A_c = Y[np.ix_(tempIndices, tempIndices)] # adjacency matrix for subgraph induced by cliques.
        c_clique = np.array([0 for a in range(K*clique_size)])
        for i in range(K*clique_size):
            c_clique[i] = np.floor(np.float(i)/clique_size)
        pHatVec = [EstimatePhat(i,j,K,clique_size* K,c_clique,A_c)[0]/EstimatePhat(i,j,K,clique_size*K,c_clique,A_c)[1] for i in range(K) for j in range(K) if i < j]
        pHat[np.triu_indices(K, 1)] = pHatVec
        pHat = pHat + np.transpose(pHat)
        E_nu = np.mean([approximateEnu(n, K, j, Y, finalCliques, clique_size) for j in [t]])
        np.fill_diagonal(pHat, E_nu)
        print("this is how many pHat == 0")
        print(np.sum(pHat == 0))
        if not 0 in pHat:
            dHat = -np.log(pHat/E_nu)
            return([Y, pHat, dHat, finalCliques, E_nu]) # return adjacency matrix and pHat.


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

def ComputeDhat(Y, K, clique_size, t, cliques, n):
    pHat = np.zeros((K, K))

    tempIndices = [int(b) for b in cliques.flatten()]
    A_c = Y[np.ix_(tempIndices, tempIndices)]                               # adjacency matrix for subgraph induced by cliques.
    c_clique = np.array([0 for a in range(K*clique_size)])
    for i in range(K*clique_size):
        c_clique[i] = np.floor(np.float(i)/clique_size)
    pHatVec = [EstimatePhat(i,j,K,clique_size * K,c_clique,A_c)[0]/EstimatePhat(i,j,K,clique_size*K,c_clique,A_c)[1] for i in range(K) for j in range(K) if i < j]
    pHat[np.triu_indices(K, 1)] = pHatVec
    pHat = pHat + np.transpose(pHat)
    E_nu = np.mean([j for j in [approximateEnu(n, K, i, Y, cliques, clique_size) for i in [t]] if j > 0])
    if np.isnan(E_nu):                                                      # If E_nu is nan, set E_nu = 1 (so no FE in model)
        E_nu = 1
    np.fill_diagonal(pHat, E_nu)
    dHat = np.zeros((K, K))
    print(pHat)
    for i in range(K):
        for j in range(K):
            if i != j:
                dHat[i,j] = -np.log(pHat[i,j]/float(E_nu))
    return([dHat, E_nu])


def PickCliquesFeasible(n, K, Y, clique, numSamples, clique_size, t):
    objFun = np.zeros(numSamples)
    indices = np.zeros((numSamples, K))
    cliques_saved = [[] for i in range(numSamples)]
    dHat_saved = [[] for i in range(numSamples)]
    Enu_saved = np.zeros(numSamples)
    for index in range(numSamples):
        print(index)
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
        E_nu = np.mean([j for j in [approximateEnu(n, K, i, Y, cliques_sampled, clique_size) for i in [t]] if j > 0])
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
    return([indices[index_optim,:], np.min(objFun), np.min(objFun) < 100, dHat_saved[index_optim], cliques_saved[index_optim], Enu_saved[index_optim]])



def ReturnClassification(dHat, J, K, Y, finalCliques, clique_size, E_nu, zeta):
    fHat = -0.5 * J @ np.square(dHat) @ J
    rejectEuclidean, p_E = Bootstrap_Romano(Y, finalCliques, clique_size, K, E_nu, J, np.linalg.eigvalsh(fHat)[0], 0.05, clique_size -1, 'Euclidean', 0, 0, 1, zeta)
    a = np.max(dHat)/np.pi
    b = 3 * a

    kappaHat_S = np.mean([Estimate_Curvature_Sphere(dHat, K, a, b, j) for j in [0]])
    lambdaHat = np.linalg.eigvalsh(np.cos(np.sqrt(kappaHat_S) * dHat))[0]
    rejectSpherical, p_S = Bootstrap_Romano(Y, finalCliques, clique_size, K, E_nu, J, lambdaHat, 0.05, clique_size - 1, 'Spherical', kappaHat_S, 0, 0.33, zeta)

    kappaHat_H = np.mean([Estimate_Curvature_Hyperbolic(dHat, K, -a, -b, j) for j in [-2]])
    lambdaHat = np.linalg.eigvalsh(np.cosh(np.sqrt(-kappaHat_H) * dHat))[-2]
    rejectHyperbolic, p_H = Bootstrap_Romano(Y, finalCliques, clique_size, K, E_nu, J, lambdaHat, 0.05, clique_size - 1, 'Hyperbolic', -kappaHat_H, -2,  0.66, zeta)

    # Classify using the ordered test (the order is E -> S -> H).
    if rejectEuclidean == 0:
        classification = 0
    if rejectEuclidean == 1 and rejectSpherical == 0:
        classification = 1
    if rejectEuclidean == 1 and rejectSpherical == 1 and rejectHyperbolic == 0:
        classification = 2
    if rejectEuclidean == 1 and rejectSpherical == 1 and rejectHyperbolic == 1:
        classification = 3
    return([classification, p_E, p_S, p_H, rejectEuclidean, rejectSpherical, rejectHyperbolic])
