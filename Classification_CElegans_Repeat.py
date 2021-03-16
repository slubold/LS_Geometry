import numpy as np
import networkx as nx
#import seaborn as sns; sns.set(color_codes=True)
#from Return_Classification import *
import scipy.io as sio
from Supplementary_Files import *
from Bootstrap import *
import os
import pickle 
from datetime import datetime
import time


JOBID = int(os.getenv('SLURM_ARRAY_TASK_ID', 0)) # saves the job ID when running on the cluster

print(JOBID)


def Return_Classification(Y, clique_size, K, m, mode, computeRate, finalCliques = [], dHat = [], E_nu = 1):
    G = nx.from_numpy_matrix(Y)
    n = len(Y[0,:])
    J = np.identity(K) - 1/float(K) * np.outer(np.ones(K), np.ones(K))

    fHat = -0.5 * J @ np.square(dHat) @ J
    lambda_F = np.linalg.eigvalsh(fHat)[0]

   # a = 0.5
   # b = 1.5
    print(dHat)
    b =  (np.pi/np.max([dHat[i,j] for i in range(K) for j in range(K) if i < j]))**2
    a =  (3 * np.min([dHat[i,j] for i in range(K) for j in range(K) if i < j]))**(-2)
    

    kappaHat_S = np.mean([Estimate_Curvature_Sphere(dHat, K, a, b, j) for j in [0, 1, 2]])
    lambda_CS = np.linalg.eigvalsh(np.cos(np.sqrt(kappaHat_S) * dHat))[0]

    kappaHat_H = np.mean([Estimate_Curvature_Hyperbolic(dHat, K, -a, -b, j) for j in [-2, -3, -4]])
    lambda_CH = np.linalg.eigvalsh(np.cosh(np.sqrt(-kappaHat_H) * dHat))[-2]


    rate_E = 1/3
    rate_S = 1/3
    rate_H = 1/3


    rejectEuclidean, p_E = Bootstrap_Romano(Y, finalCliques, clique_size, K, E_nu, J, lambda_F, 0.05, m, "E", 0, rate = rate_E, zeta = 1)
    rejectSpherical, p_S = Bootstrap_Romano(Y, finalCliques, clique_size, K, E_nu, J, lambda_CS, 0.05, m, "S", kappaHat_S,  rate = rate_S, zeta = 1)
    rejectHyperbolic, p_H = Bootstrap_Romano(Y, finalCliques, clique_size, K, E_nu, J, lambda_CH, 0.05, m, "H", kappaHat_H, rate = rate_H, zeta = 1)


    if rejectEuclidean == 0:
        classification_ordered = 0
    if rejectEuclidean == 1 and rejectSpherical == 0:
        classification_ordered = 1
    if rejectEuclidean == 1 and rejectSpherical == 1 and rejectHyperbolic == 0:
        classification_ordered = 2
    if rejectEuclidean == 1 and rejectSpherical == 1 and rejectHyperbolic == 1:
        classification_ordered = 3

    pMax = np.max([p_E, p_S, p_H])
    if p_E == pMax:
        classification_max = 0
    if p_S == pMax:
        classification_max = 1
    if p_H == pMax:
        classification_max = 2


    if classification_max == 0:
        W_Hat = fHat
        geometry = "Euclidean"
        kappaHat = 0
    if classification_max == 1:
        W_Hat = np.cos(np.sqrt(kappaHat_S) * dHat)
        geometry = "Spherical"
        kappaHat = kappaHat_S
    if classification_max == 2:
        W_Hat = np.cosh(np.sqrt(-kappaHat_H) * dHat)
        geometry = "Hyperbolic"
        kappaHat = kappaHat_H

    return([classification_max, p_E, p_S, p_H, kappaHat_S, kappaHat_H, a, b])

#sim_number = JOBID % 10 
time_simulation = time.strftime("%Y%m%d-%H%M%S")

K = 12


Y = np.loadtxt(open("Network_Data/celegans131matrix.csv", "rb"), delimiter=",") 
G = nx.from_numpy_matrix(Y)
n = len(Y[0,])
clique_number = np.int(nx.graph_clique_number(G))                # find clique number in graph

clique_size = 5 # the clique number is 6, so we take clique number - 1
print("this is clique size")
print(clique_size)
cliques = np.array([x for x in list(nx.enumerate_all_cliques(G)) if len(x) == clique_size])
numSamples = 10**6
m = 3 #clique_size - 1


t = clique_number - 2

while 1:
    print("doing it again")
    indices, min_obj, found_min, dHat, temp, E_nu  = PickCliquesFeasible(n, K, Y, cliques, numSamples, int(clique_size), t)
    print(found_min)
    if found_min == True:
        break
    if found_min != True:
        continue
print("out of while loop")

results = Return_Classification(Y, clique_size, K, m, "Not Data", "False", cliques, dHat)

data = {'results': results, 'clique_size': clique_size, "K": K, "m": m, "indices": indices, "min": min_obj, "numSamples": numSamples}

file_name = 'CElegans_Classification/' + 'K_12_1000k_ab_WilsonBounds_m3' + 'sim_number' + str(JOBID) + '.pickle'

with open(file_name, 'wb') as handle:
    pickle.dump(data, handle, protocol = pickle.HIGHEST_PROTOCOL)




