import numpy as np
from matplotlib import pyplot as plt

def Make_Separated_D_Spherical_3D(K, n, kappa, ub_theta, ub_phi, delta):
    DGroups = np.zeros((K, K))
    theta = np.random.uniform(0, ub_theta, K)
    phi = np.random.uniform(0, ub_phi, K)
    Z = np.zeros((K, 3))
    for i in range(K):
        Z[i,:] = [1/np.sqrt(kappa) *  np.sin(theta[i])*np.cos(phi[i]),  1/np.sqrt(kappa) * np.sin(theta[i]) * np.sin(phi[i]), 1/np.sqrt(kappa) *  np.cos(theta[i])]

    for i in range(K):
        for j in range(K):
            if i != j:
                DGroups[i,j] = 1/np.sqrt(kappa) *  np.arccos(kappa * Z[i,:].dot(Z[j,:]))

    theta_Pos = np.zeros(n)
    phi_Pos = np.zeros(n)
    Z = np.zeros((n, 3))
    for i in range(n):
        theta_Pos[i] = np.random.uniform(theta[np.int(np.floor(i*K/n))] - delta, theta[np.int(np.floor(i*K/n))] + delta, size = 1)
        phi_Pos[i] = np.random.uniform(phi[np.int(np.floor(i*K/n))] - delta, phi[np.int(np.floor(i*K/n))] + delta, size = 1)
        Z[i,:] = [1/np.sqrt(kappa) * np.sin(theta_Pos[i])*np.cos(phi_Pos[i]),  1/np.sqrt(kappa) * np.sin(theta_Pos[i]) * np.sin(phi_Pos[i]), 1/np.sqrt(kappa) * np.cos(theta_Pos[i])]

    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                D[i,j] = 1/np.sqrt(kappa) * np.arccos(Z[i,:].dot(Z[j,:]) * kappa)
    np.fill_diagonal(D, 0)
    return([DGroups, D])
