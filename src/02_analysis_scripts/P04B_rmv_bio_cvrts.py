import scanpy as sc
import matplotlib.pyplot as plt
import anndata as ad
import os
ad.settings.allow_write_nullable_strings = True
sc.settings.figdir = "./results/figures"
sc.settings.verbosity = 3

def load_evidence(h5ad_path):
    '''
    Loads the QC artifact from P03 file
    '''
    print(f"   -> Loading QC artifact from: {h5ad_path}")
    adata = sc.read_h5ad(h5ad_path)
    print(f"   -> Loaded dimensions: {adata.n_obs} cells x {adata.n_vars} genes")
    return adata

