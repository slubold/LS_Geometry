# Author: Shane Lubold (sl223@uw.edu)
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------

# Import files
import numpy as np
import pickle
import numpy.linalg
import scipy.spatial
from Make_Euclidean_D import *
from Make_D_Spherical import *
from Generate_Hyperbolic import *
from Supplementary_Files import *
from Bootstrap import *
from datetime import datetime
import time
import os

# Uncomment this line of code if running on a cluster.
#JOBID = int(os.getenv('SLURM_ARRAY_TASK_ID', 0)) # saves the job ID when running on the cluster


# Set parameters P
P = {
    'n': 1200,                      # size of graph.
    'K_vec': [5, 7, 10],           # values of K (number of cliques).
    'dim': 3,                       # dimension of latent space.
    'kappa': 0.75,                   # curvature of latent space.
    'numSim': 100,                   # number of sets of cliques that we simulate.
    'beta': -0.01,                  # lower bound for distribution of fixed effects.
    'ell_vec': [7],           # values of ell (clique size).
    'alpha': 0.05,                  # significance level of test.
    'zeta': 1,                      # scaling in front of distances (default = 1).
    'scale': 2.5,                   # scale used when generating points in hyperbolic space.
    'ub_theta': np.pi,              # upper bound for distribution of theta angle for points in  spherical LS.
    'ub_phi': 2 * np.pi,            # upper bound for distribution of phi angle for points in  spherical LS.
    'delta': 10**(-2),              # controls the spread of the nodes around their group centers in spherical LS.
    'rate': 1/3,                   # rate used when sub-sampling. This is the exponent of tau_n = n^rate.
    'sigma': 0.8,                  # controls the spread of the points in Euclidean space.
    'geometry_true': 'E',           # we simulate graphs from geometry_true.
    'geometry_test': 'S',           # we test if the latent space is geometry_test.
    'numSamples': 45,               # determines how many nodes to take in one group to find cliques. (larger values slow down the code).
    'a': 1/3,                       # lower bound used to search for kappa.
    'b': 1.5,                       # upper bound used to search for kappa.
    'q': 2,                          # number of eigenvalues we minimize to compute our estimate of kappa.
    'm': [6]                  # sub-sample rate. We take m = clique_size - 1.
}

def ComputePerformance(P):

    # Pre-define variables.
    # --------------------------------------------------------------------------------
    wHat_array = [[[] for x in range(len(P['ell_vec']))] for y in range(len(P['K_vec']))]
    dHat_array = [[[] for x in range(len(P['ell_vec']))] for y in range(len(P['K_vec']))]
    reject = np.zeros((len(P['K_vec']), len(P['ell_vec']),  P['numSim']))
    pvalue = np.zeros((len(P['K_vec']), len(P['ell_vec']), P['numSim']))
    kappaHat = np.zeros((len(P['K_vec']), len(P['ell_vec']), P['numSim']))
    HatE_nu_vec = np.zeros(P['numSim'])

    # Generate latent space positions.
    # --------------------------------------------------------------------------------
    if P['geometry_true']  == 'E':
        kappa = 0
        DGroups, D = Make_Euclidean_D(P['K_vec'][-1], P['n'], P['sigma'], P['dim'])
    if P['geometry_true'] == "S":
        DGroups, D = Make_Separated_D_Spherical_3D(P['K_vec'][-1], P['n'], P['kappa'], P['ub_theta'], P['ub_phi'], P['delta'])
    if P['geometry_true'] == "H":
        muVec, DGroups = Generate_Hyperbolic(P['K_vec'][-1], P['dim'], P['scale'], P['kappa'])
        D = Make_D_FromMu_Hyperbolic(P['n'], P['dim'], P['K_vec'][-1], muVec, P['kappa'], P['delta'])
    # --------------------------------------------------------------------------------

    # Generate Graph.
    # --------------------------------------------------------------------------------
    nu = [np.random.uniform(P['beta'],0) for i in range(P['n'])]                                                  # compute fixed effects
    E_nu = np.mean(np.exp(nu))**2
    edge_probabilities = [np.exp(nu[i] + nu[j] - P['zeta'] * D[i,j]) for i in range(P['n']) for j in range(P['n']) if i < j]  # compute probability of edges between nodes
    # --------------------------------------------------------------------------------

    dHat_array_changing_ell = [[] for i in range(len(P['ell_vec']))]
    finalCliques = [[] for i in range(len(P['ell_vec']))]

    for indexSim in range(P['numSim']):
        print("Simulation: {}".format(indexSim))
        t = P['ell_vec'][-1] - 1                                                # number of edges used in the "almost clique" to compute FE expectation.
        Y, pHat, dHat, cliques, HatE_nu_vec[indexSim] = GenerateY(P['K_vec'][-1], P['n'], edge_probabilities, P['ell_vec'][-1], E_nu, P['numSamples'], t)
        dHat = dHat/P['zeta']

        for indexTemp in range(len(P['ell_vec']) - 1):
            print("this is index: {}".format(indexTemp))
            t = P['ell_vec'][indexTemp] - 1                 # number of edges used in the "almost clique" to compute FE expectation.
            dHat_array_changing_ell[indexTemp], finalCliques[indexTemp] = findCliques_GivenY(Y, P['K_vec'][-1], P['n'], P['ell_vec'][indexTemp], E_nu, P['numSamples'], t) # removeNodes(Y, cliques, t, P['ell_vec'][index], P['K_vec'], P['K_vec'][-1], P['n'], P['ell_vec'][-1]) #
        dHat_array_changing_ell[-1] = dHat
        finalCliques[-1] = cliques

        for indexEll in range(len(P['ell_vec'])):

            temp_dHat = np.squeeze(dHat_array_changing_ell[indexEll]) # return 2D array, not list
            mStar =  P['m'][indexEll]

            for indexK in range(len(P['K_vec'])):
                print("indexEll: {}".format(indexEll))
                print("indexK: {}".format(indexK))
                print("this is d current")
                print(temp_dHat)

                J = np.identity(P['K_vec'][indexK]) - 1/P['K_vec'][indexK] * np.outer(np.ones(P['K_vec'][indexK]), np.ones(P['K_vec'][indexK]))
                indices_sampled = np.random.choice(P['K_vec'][indexK], P['K_vec'][indexK], replace = False)
                dHat_array[indexK][indexEll] = temp_dHat[np.ix_(indices_sampled, indices_sampled)]
                print("this is J")
                print(J)
                print(dHat_array[indexK][indexEll])

                if P['geometry_test'] == 'E':
                    wHat_array[indexK][indexEll] = -1/2 * J @ np.square(dHat_array[indexK][indexEll]) @ J
                    kStar = 0
                    kappaHat[indexK, indexEll, indexSim] = 0

                if P['geometry_test'] == "S":
                    kappaHat[indexK, indexEll, indexSim] = np.mean([Estimate_Curvature_Sphere(dHat_array[indexK][indexEll],
                                P['K_vec'][indexK], P['a'], P['b'], index) for index in range(P['q'])])
                    wHat_array[indexK][indexEll] = np.cos(np.sqrt(kappaHat[indexK, indexEll, indexSim]) * dHat_array[indexK][indexEll])
                    kStar = 0

                if P['geometry_test'] == "H":
                    indices_eig = [-2 - temp for temp in range(P['q'])]
                    kappaHat[indexK, indexEll, indexSim] = np.mean([Estimate_Curvature_Hyperbolic(dHat_array[indexK][indexEll],
                                    P['K_vec'][indexK], -P['a'], -P['b'], index) for index in indices_eig])
                    wHat_array[indexK][indexEll] = np.cosh(np.sqrt(-kappaHat[indexK, indexEll, indexSim]) * dHat_array[indexK][indexEll])
                    kStar = -2

                eigWHat = np.linalg.eigvalsh(wHat_array[indexK][indexEll])
                reject[indexK, indexEll, indexSim], pvalue[indexK, indexEll, indexSim] = Bootstrap_Romano(Y, finalCliques[indexEll], P['ell_vec'][indexEll], P['K_vec'][indexK],
                        HatE_nu_vec[indexSim], J, eigWHat[kStar], P['alpha'], mStar, P['geometry_test'], kappaHat[indexK, indexEll, indexSim], P['rate'], P['zeta'])
    LSreject = np.squeeze(np.mean(reject, axis = 2))
    return({'reject': reject, 'pvalue': pvalue, 'kappaHat': kappaHat, 'D': D, 'DGroups': DGroups, 'HatE_nu_vec': HatE_nu_vec})

results = ComputePerformance(P)

time_simulation = time.strftime("%Y%m%d-%H%M%S")

results_dictionary = {'parameters': P, 'results': results}

# enter file name here
file_name = 'file_name' #'Power_Using_M468/final_ES/geometry_true_' + P['geometry_true'] +'_geometry_test' + P['geometry_test'] + str(JOBID) + time_simulation + '.pickle'

with open(file_name, 'wb') as handle:
    pickle.dump(results_dictionary, handle, protocol = pickle.HIGHEST_PROTOCOL)


