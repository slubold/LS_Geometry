# LS_Geometry

The code in this repository implements the geometry classification methods from *Identifying Latent Space Geometry in Network Models using Analysis of Curvature* by Lubold et al. The list of files is below. Please send suggestions or comments to Shane Lubold at sl223@uw.edu!

Files:
1) Classification.py: This file takes as input the adjacency matrix of the graph and outputs the classification of the LS (Euclidean, spherical, or hyperbolic). It also outputs the LS dimension and the LS curvature.
2) Bootstrap.py: This file implements the bootstrapping method from Romano et al.
3) Compute_Rate.py: This file estimatates the parameter \beta that appears in the Assumption 1 of Romano et al. 
4) Estimate_Curvature.py: This file estimates the curvature of the LS from the estimated distance matrix.
5) Rank_Estimator.py: This file estimates the rank of the LS from the estimated distance matrix. 
6) Supplementary_Files.py: This file contains supplementary files needed to run the above files. For example, it contains code to estimate 
the matrix that contains the probabilities of connections between LS locations, denoted by P, from the graph.
7) celeagsn131matrix.csv: This file contains the adjacency matrix of a neuron network in a C. Elegans. This data can be accessed here: https://www.dynamic-connectome.org/?page_id=25
