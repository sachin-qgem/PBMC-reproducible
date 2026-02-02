import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy import sparse, special, optimize
import numpy as np
from typing import Literal
import anndata as ad
import statsmodels.api as sm

ad.settings.allow_write_nullable_strings = True
sc.settings.figdir = "./results/figures"
#sc.settings.verbosity = 3


'''
Loading the h5ad file from the P03_qc_filtering step.
We will have the following structure:
  - `adata.layer['counts']` to preserve the raw counts to be later used in further
  steps downstream. 
  - `adata.layer['lop1p_norm']` to have the normalized total counts of cells and the
  log1p on that, it is easier # !for visualization ONLY.
  - Tag Only (`adata.var['highly_variable']`), highly variable genes using analytic Pearson residuals HVG on adata.X raw
  - Estimate_global_theta ($\theta = {1/{\alpha}}$) instead of standard 100 for the next step of Pearson residual normalization 
  - Apply analytic Pearson residual normalization on adata.X raw (all genes)
  - Save the modified adata into h5ad

We will use theta value as 100.0 as reviewed from (Lause et al., 2021): 
"Of course, any given dataset would be better fit using gene-specific values \theta.
 However, our goal is not the best possible fit: We want the model to account only 
 for technical variability, but not biological variability..."

"Rather than estimating the \theta value from a biologically heterogeneous dataset
 such as PBMC, we think it is more appropriate to estimate the technical overdispersion
   using negative control datasets..."

"We found that across different protocols 10x, inDrop, etc., negative control data
 were consistent with overdispersion \theta \approx 100 or larger."
    
'''

def load_evidence(h5ad_path):

    # Loads the QC-cleaned artifact from P03 file
    print(f"   -> Loading QC artifact from: {h5ad_path}")
    adata = sc.read_h5ad(h5ad_path)
    print(f"   -> Loaded dimensions: {adata.n_obs} cells x {adata.n_vars} genes")
    return adata

    
def data_layering(adata):
    '''
    create the adata.layer['counts'], adata.layer['lop1p_norm'],and keeping the
    adata.X clean and raw from the P03 file
    '''
    
    adata.layers['counts'] = adata.X.copy()
    adata_temp = adata.copy()
    sc.pp.normalize_total(adata_temp, target_sum = 1e4)
    sc.pp.log1p(adata_temp)
    adata.layers['log1p_norm'] = adata_temp.X.copy()
    del adata_temp
    return adata

def npr_and_hvg(adata):
    
    # we will tag the genes as HVG or not
    # Taking theta Value to be 100.0 as from experiments in the 
    sc.experimental.pp.highly_variable_genes(adata, theta= 100.0, flavor="pearson_residuals",
                                              n_top_genes=3000,clip=True,subset=False)
    # Pearson residuals calculation
    sc.experimental.pp.normalize_pearson_residuals(adata, theta = 100.0,clip=True)
    return adata


if __name__ == "__main__":
    h5ad_path = "data/objects/pbmc3k_qc.h5ad"
    output_path = "data/objects/pbmc3k_norm.h5ad"
    adata = load_evidence(h5ad_path)
    adata = data_layering(adata)
    adata = npr_and_hvg(adata)
    adata.write_h5ad(output_path,compression='gzip')
    print (f"::: P04 Output {output_path}  with pearson residuals and HVG saved")
