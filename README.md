# LS_Geometry

The code in this repository implements the geometry classification methods from *Identifying Latent Space Geometry in Network Models using Analysis of Curvature* by Lubold et al. The list of files is below. Please send suggestions or comments to Shane Lubold at sl223@uw.edu!

Files:
1) Classification.py: This file takes as input the adjacency matrix of the graph and outputs the classification of the LS (Euclidean, spherical, or hyperbolic). It also outputs the LS dimension and the LS curvature.
2) Bootstrap.py: This file implements the bootstrapping method from Romano et al.
3) Compute_Rate.py: This file estimatates the parameter $'\beta'$



![\Large x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}](https://latex.codecogs.com/svg.latex?x%3D%5Cfrac%7B-b%5Cpm%5Csqrt%7Bb%5E2-4ac%7D%7D%7B2a%7D)
