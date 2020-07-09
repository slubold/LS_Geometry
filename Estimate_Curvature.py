import numpy as np
import matplotlib.pyplot as plt

def Estimate_Curvature_Sphere(dHat, K, a, b, index):
        kappaVec = np.linspace(a, b, 10000)
        objFun = np.zeros(len(kappaVec))
        for i in range(len(kappaVec)):
            C = np.cos(dHat * np.sqrt(kappaVec[i]))
            objFun[i] = np.abs(np.linalg.eigvalsh(C)[index])
        index_min = np.argmin(objFun)
        return(kappaVec[index_min])

def Estimate_Curvature_Hyperbolic(dHat, K, a, b,index):
        kappaVec = np.linspace(a, b, 10000)
        objFun = np.zeros(len(kappaVec))
        for i in range(len(kappaVec)):
            C = np.cosh(dHat * np.sqrt(-kappaVec[i]))
            objFun[i] = np.abs(np.linalg.eigvalsh(C)[index])
        index_min = np.argmin(objFun)
        return(kappaVec[index_min])
