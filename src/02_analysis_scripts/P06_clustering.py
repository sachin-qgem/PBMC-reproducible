import scanpy as sc
import matplotlib.pyplot as plt
import anndata as ad
from sklearn.model_selection import train_test_split
ad.settings.allow_write_nullable_strings = True
sc.settings.figdir = "./results/figures/phase5_B_clustered_geometry"
sc.settings.verbosity = 3

# Loading Evidence
def load_evidence(h5ad_path):

    # Loads the PCA artifact from P05 file
    print(f"   -> Loading PCA artifact from: {h5ad_path}")
    adata = sc.read_h5ad(h5ad_path)
    print(f"   -> Loaded dimensions: {adata.n_obs} cells x {adata.n_vars} genes")
    return adata

'''
We will be splitting the dataset into two(A, B), to sove the Double Dipping problem
Then perform KNN, leiden, and Umap, on A, train a classifier on it and project
a untouched B on A, with obs columns 'leiden'. So, that B only has labelled leiden
based on pattern learned from A and not actual clusters. And then Visualizing and comparing
the splits UMAPs to check if it got projected correctly
'''
# Split Dataset
def split_data(adata):
    A_data_indices, B_data_indices = train_test_split(adata.obs_names,test_size=0.5,
                                                     random_state=42,shuffle=True)
    adata_A = adata[A_data_indices].copy()
    adata_B = adata[B_data_indices].copy()
    return adata_A,adata_B

# Run clustering and KNN , Umap on A_Data
def knn_clustering_umap(adata_A, n_neighbors:int,n_pcs:int,
                        leiden_resolution:float):
    sc.pp.neighbors(adata_A,n_neighbors = n_neighbors,n_pcs=n_pcs,method='umap',
                    knn=True,metric='euclidean',random_state=42)
    sc.tl.leiden(adata_A,resolution=leiden_resolution,n_iterations= -1,
                 flavor='leidenalg',random_state=42)
    sc.tl.umap(adata_A,min_dist=0.3,maxiter=500,random_state=42)
    
    return adata_A

# Train classifier on A and project B onto it
def project_B_on_A(adata_B,adata_A):
    sc.tl.ingest(adata_B,adata_A,obs='leiden',embedding_method='umap')
    return adata_B

# Visualizing and comparing the splits
def visualize_split(adata_A,adata_B):
    
    sc.pl.umap(adata_A,color='leiden',components ='all',
               size= 10.0,show=True,title = 'Training Manifold',
                save = f"_Training Manifold.png" )
    sc.pl.umap(adata_B,color='leiden',components ='all',
               size= 10.0,show=True,title = 'Projected Validation',
               save = f"_Projected Validation.png")
    

if __name__ == '__main__':
    h5ad_path = "data/objects/pbmc3k_pca.h5ad"
    output_path = "data/objects/pbmc3k_clustered_B.h5ad"
    adata = load_evidence(h5ad_path)
    adata_A, adata_B = split_data(adata)
    adata_A = knn_clustering_umap(adata_A,15,10,0.5)
    adata_B = project_B_on_A(adata_B,adata_A)
    print(f"Saving and Visualizing Both UMAPs from A and B")
    visualize_split(adata_A,adata_B)
    print(f"Saving the adata_B to {output_path} for Downstream analaysis")
    adata_B.write_h5ad(output_path)

    