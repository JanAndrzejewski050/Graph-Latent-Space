# GCN model 
## First version (first_full_train+eval_gcn1.ipynb)
The first version contained only atom and bonds as node features in graphs. 
It had a better validity score and uniqueness but the tsne showed that the space is less coherent
## Second version (full_train+eval_gcn2.ipynb)
We added much more information from the dataset, now node features are of length 30. 
It had worse performance on latent analysis but did better when it comes to exploring the latent space.
