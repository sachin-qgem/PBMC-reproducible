import scanpy as sc
import matplotlib.pyplot as plt
import anndata as ad

ad.settings.allow_write_nullable_strings = True
sc.settings.figdir = "./results/figures"
sc.settings.verbosity = 3

# Load Evidence
def load_evidence(h5ad_path):

    # Loads the Analytically Pearson residuals and HVGs artifact from P04 file
    print(f"   -> Loading NPR and HVGs artifact from: {h5ad_path}")
    adata = sc.read_h5ad(h5ad_path)
    print(f"   -> Loaded dimensions: {adata.n_obs} cells x {adata.n_vars} genes")
    return adata

#perform PCA,
def pca(adata):
    sc.pp.pca(adata,n_comps=100,zero_center=True,svd_solver='arpack',
              mask_var='highly_variable')
    return adata

#screeplot and save in figures folder
def screeplot_and_pca(adata,stagename:str):
    
    sc.pl.pca_variance_ratio(adata,n_pcs=50,show=True,save=f"_{stagename}.png")
    sc.pl.pca(adata,color_map = 'Blues',save=f"_components.png"
              ,projection = '3d',annotate_var_explained=True)

if __name__ == '__main__':
    h5ad_path = "data/objects/pbmc3k_norm.h5ad"
    output_path = "data/objects/pbmc3k_pca.h5ad"
    adata = load_evidence(h5ad_path)
    adata = pca(adata)
    screeplot_and_pca(adata,'P05_pca_elbow_plot')
    adata.write_h5ad(output_path)
    


    

