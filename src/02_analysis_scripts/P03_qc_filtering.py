import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import os
import scipy.stats as sps
import numpy as np
from typing import Literal
import anndata as ad
ad.settings.allow_write_nullable_strings = True
sc.settings.figdir = "./results/figures"
sc.settings.verbosity = 3
def load_evidence(path):

    #Load and return the adata object from the mtx files
    adata = sc.read_10x_mtx(path,var_names= "gene_symbols",make_unique=True)
    
    print(f" -> Loaded Successfully. Shape : {adata.shape}")
    return adata

# Calculating all the metrics we would need \
# in next function to filter and clean the data.
def calculate_vital_signs(adata):
  # 1. IDENTIFY MITOCHONDRIA (The Label)
  # We must flag the genes first so the machine knows what to count for Step 2.
  # Logic: Look at 'gene_symbols' (or index) for names starting with "MT-"
    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(adata, expr_type="counts", 
                                      qc_vars=["mt"],
                                         inplace=True, log1p=False)
    return adata

'''
After calculating and storing the required metrics
 before we proceed to cut or filter any genes or cells, trimming, we need to 
 find the thresholds for our case, because blindly doing <5% mito cut or
 <200/>2500 gene counts threshold for cells wouldn't be wise.
To come around this we can either plot our data, and check a suitable thresholds
 or **also employ some statistics (Like 3 or 5 MADs) to find these threshholds**
For this we deploy two functions (is_outlier) and (audit_distribution)
'''
# is_outlier
def is_outlier(adata, metric:str, nmads:int,
                side: Literal["both", "upper", "lower"] = "both"):
    M = adata.obs[metric]
    m_mad = sps.median_abs_deviation(M)
    m_median = np.median(M)
    lower_bound = m_median - nmads * m_mad
    upper_bound = m_median + nmads * m_mad
    if side == "upper":
        return M > upper_bound
    elif side == "lower":
        return M < lower_bound
    return (M < lower_bound) | (M > upper_bound)
    

# audit_distribution
def audit_distribution(adata,voilin_keys,scatter_x:str,scatter_y:str,
                       scatter_color:str,stagename:str):
# plotting three obs columns : n_genes_by_counts,pct_counts_mt, 
# total_counts (for each cell) 
    sc.pl.violin(adata,voilin_keys,jitter=0.4,
                 multi_panel=True,save= f"_{stagename}.png")
    sc.pl.scatter(adata,x=scatter_x,y=scatter_y,color=scatter_color,
                  size=5,save=f"_{stagename}.png")


''' 
After checking the plots, we can verify if the number of MADs (nmads)
can be 5 or 3 or adjusted accordingly.
Here we chekced for our use case, and nmads=5 is safe
'''
# applying the filters and cleaning the adata
def apply_filter(adata):
    
    # creating the outlier columns
    adata.obs["outlier_n_genes_by_counts"] = is_outlier(adata,
                                                        "n_genes_by_counts",5,"both")
    adata.obs["outlier_pct_counts_mt"] = is_outlier(adata,"pct_counts_mt",5,"upper")

    # Keeping cells excluding which are both outliers in Mito and gene counts threshold
    adata.obs["keep_cells"] = ((~adata.obs["outlier_n_genes_by_counts"]) & 
        (~adata.obs["outlier_pct_counts_mt"]))
    
    # cleaned adata
    adata_filtered = adata[adata.obs["keep_cells"], :].copy()

    # THE GENE PURGE (Remove Empty Inventory)
    # Filter genes that are not present in at least 3 cells of the remaining population
    sc.pp.filter_genes(adata_filtered, min_cells=3)
    
    # 5. REPORT & RETURN
    cells_removed = adata.n_obs - adata_filtered.n_obs
    genes_removed = adata.n_vars - adata_filtered.n_vars
    
    print(f"   -> Final Dimensions: {adata_filtered.n_obs} \
    cells x {adata_filtered.n_vars} genes")
    print(f"   -> Removed {cells_removed} cells and {genes_removed} genes.")
    return adata_filtered


if __name__ == "__main__":

    #define the Filtered mtx folder downloaded from 10x genomics (Ground Truth)
    mtx_path = "data/raw/pbmc3k_filtered_gene_bc_matrices/hg19"
    pbmc3k_qc_h5ad_path = "data/objects/pbmc3k_qc.h5ad"
    adata = load_evidence(mtx_path)
    adata = calculate_vital_signs(adata)
    v_keys = ['n_genes_by_counts', 'total_counts', 'pct_counts_mt']
    print("   -> Running Pre-Filter Audit...")
    audit_distribution(adata, v_keys, "total_counts", "n_genes_by_counts",
                    "pct_counts_mt", "pre_filter")
    
    adata_filtered = apply_filter(adata)
    print("   -> Running Post-Filter Audit...")
    audit_distribution(adata_filtered, v_keys, "total_counts", "n_genes_by_counts",
                        "pct_counts_mt", "post_filter")

    # Save the clean state for P04
    print(f"   -> Saving Golden Copy to {pbmc3k_qc_h5ad_path}...")
    adata.write_h5ad(pbmc3k_qc_h5ad_path, compression='gzip')
    
    print("::: P03 COMPLETE. READY FOR PHASE IV. :::")
