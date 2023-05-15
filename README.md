# Analysis of Calcium Imaging data underlying polymodal sensory perception in Ciona intestinalis

This repository contains the code for :
- generating low dimensional traces of neural activity in Ciona larvae using PCA (using whole brain Ca Imaging data)
- processing Calcium imaging data obtained from Mesmerise software and plotting stimulus response traces [[1]](#1)   

### Dependencies
Requires numpy, scipy, sklearn installed. Required matplotlib and seaborn for plotting.

This code uses the Python version of Rick Chartrand's algorithm for numerical differentiation of noisy data impleneted in this [github repo](https://github.com/stur86/tvregdiff).

### References
<a id="1">[1]</a> 
[Kolar, K., Dondorp, D., Zwiggelaar, J.C. et al. Mesmerize is a dynamically adaptable user-friendly analysis platform for 2D and 3D calcium imaging data. Nat Commun 12, 6569 (2021)](https://doi.org/10.1038/s41467-021-26550-y_

