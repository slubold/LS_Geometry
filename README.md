# LS Geometry

The code in this repository implements the geometry classification methods from *Identifying Latent Space Geometry in Network Models using Analysis of Curvature* by Lubold et al (https://arxiv.org/abs/2012.10559). The list of files is below. Send questions or comments to Shane Lubold at sl223@uw.edu.
The Indian Villages data used in this paper is not available publicly, so we only provide the code to reproduce the C Elegans example.

Files:
- Bootstrap.py: Implements the bootstrapping method used to test the three geometry hypotheses (Euclidean, spherical, and hyperbolic). 
- celegans131matrix.csv: Contains the adjacency matrix of a neuron network in a C. Elegans worm. This data can be accessed here: https://www.dynamic-connectome.org/?page_id=25
- Classification_CElegans_Repeat.py: Classifies the C Elegans neural network available in celegans131matrix.csv. 
- Compute_Rate: Implements an estimator of the rate of the sequence \tau_n that appears in Section 2 of https://projecteuclid.org/euclid.aos/1176325770. Currently, our method does not use this code. We simply use \tau_n = 1/3. However, future work could use this code to construct use better rate estimates. 
- Generate_Hyperbolic.py: Simulates positions and distances on hyperbolic space.
- Make_D_Spherical.py: Simulates positions and distances on spherical space. 
- Make_Euclidean_D.py: Simulates positions and distances on Euclidean space. 
- Rank_Estimator.py: Estimates the rank of a matrix from a noisy estimate of the matrix. We implement the "ladle" estimator from https://academic.oup.com/biomet/article-abstract/103/4/875/2659039. 
- Supplementary_Files: Contains various pieces of code that we need, such as estimating the distance matrix from the adjacency matrix.
