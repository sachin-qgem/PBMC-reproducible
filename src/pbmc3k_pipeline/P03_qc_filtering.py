import gc
import os
from typing import Literal

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.stats as sps
import skimage

# Global environment settings
ad.settings.allow_write_nullable_strings = True

sc.settings.verbosity = 3



def load_matrix(path: str) -> ad.AnnData:
    """
        Loads raw 10x Genomics Matrix Market data into an AnnData object.

        Parameters
        ----------
        path : str
            The physical directory path containing the 10x '.mtx' files.

        Returns
        -------
        ad.AnnData
            The raw, unadulterated single-cell expression matrix.
        """
    
    adata = sc.read_10x_mtx(path,var_names= "gene_symbols",make_unique=True,cache=True)
    print(f"[LOG] Matrix loaded. Dimensions: {adata.shape}")

    return adata


def compute_quality_metrics(adata: ad.AnnData) -> ad.AnnData:
    """
    Calculates primary quality control metrics including mitochondrial and ribosomal proportions.

    Parameters
    ----------
    adata : ad.AnnData
        The raw expression matrix.

    Returns
    -------
    ad.AnnData
        The matrix annotated with 'n_genes_by_counts', 'total_counts', 
        and 'pct_counts_mt'.
    """
    print("[LOG] Calculating QC metrics.")
    
    adata.var['mt'] = adata.var_names.str.startswith('MT-')

    ribo_url = "http://software.broadinstitute.org/gsea/msigdb/download_geneset.jsp?geneSetName=KEGG_RIBOSOME&fileType=txt"
    ribo_ledger = pd.read_table(ribo_url, skiprows=2, header=None)
    ribo_genes = ribo_ledger[0].values.tolist()
    
    adata.var['ribo'] = adata.var_names.isin(ribo_genes)

    sc.pp.calculate_qc_metrics(
        adata, 
        qc_vars=["mt","ribo"], 
        expr_type="counts",
        inplace=True, 
        log1p=False
    )
    
    return adata


def calculate_mad_outlier(
    adata: ad.AnnData, 
    metric: str, 
    nmads: int, 
    side: Literal["both", "upper", "lower"] = "both"
) -> np.ndarray:
    """
    Calculates Median Absolute Deviation (MAD) thresholds to identify outlier cells.

    Parameters
    ----------
    adata : ad.AnnData
        The annotated expression matrix.
    metric : str
        The specific observation column to evaluate (e.g., 'pct_counts_mt').
    nmads : int
        The multiplier for the MAD boundary (e.g., 5).
    side : Literal["both", "upper", "lower"], default "both"
        The direction of the threshold cutoff.

    Returns
    -------
    np.ndarray
        A boolean mask where True indicates an outlier violating the threshold.
    """
    M = adata.obs[metric]
    m_mad = sps.median_abs_deviation(M)
    m_median = np.median(M)
    
    lower_bound = m_median - (nmads * m_mad)
    upper_bound = m_median + (nmads * m_mad)
    
    if side == "upper":
        return M > upper_bound
    elif side == "lower":
        return M < lower_bound
    
    return (M < lower_bound) | (M > upper_bound)



def plot_qc_distributions(
    adata: ad.AnnData, 
    violin_keys: list, 
    stagename: Literal["pre_filter", "post_filter"],
    scatter_x: str = "total_counts", 
    scatter_y: str = "n_genes_by_counts", 
    scatter_color: str = "pct_counts_mt"
) -> None:
    """
    Generates violin and scatter plots for QC metric distributions.

    Parameters
    ----------
    adata : ad.AnnData
        The expression matrix.
    violin_keys : list
        The list of metrics to render in the violin plot.
    stagename : Literal["pre_filter", "post_filter"]
        The suffix appended to the saved plot filenames.
    scatter_x : str, default "total_counts"
        The metric mapped to the scatter plot X-axis.
    scatter_y : str, default "n_genes_by_counts"
        The metric mapped to the scatter plot Y-axis.
    scatter_color : str, default "pct_counts_mt"
        The metric governing the scatter plot color gradient.

    Returns
    -------
    None
    """
    print(f"[LOG] Generating QC distribution plots for: {stagename}")
    
    sc.pl.violin(
        adata, 
        violin_keys, 
        jitter=0.4, 
        multi_panel=True,
        show=False, 
        save=f"_{stagename}.svg"
    )
    sc.pl.scatter(
        adata, 
        x=scatter_x, 
        y=scatter_y, 
        color=scatter_color,
        size=5,
        show=False,
        save=f"_{stagename}.svg"
    )



def apply_quality_filters(adata: ad.AnnData) -> ad.AnnData:
    """
    Removes outlier cells based on MAD thresholds and filters low-expression genes.

    Parameters
    ----------
    adata : ad.AnnData
        The annotated expression matrix containing unmapped distributions.

    Returns
    -------
    ad.AnnData
        The filtered, structurally sound expression matrix.
    """
    print("[LOG] Applying 5-MAD threshold filters and removing genes expressed in <3 cells.")
    
    adata.obs["outlier_n_genes_by_counts"] = calculate_mad_outlier(
        adata, "n_genes_by_counts", 5, "both"
    )
    adata.obs["outlier_pct_counts_mt"] = calculate_mad_outlier(
        adata, "pct_counts_mt", 5, "upper"
    )
    
    adata.obs["keep_cells"] = (
        (~adata.obs["outlier_n_genes_by_counts"]) & 
        (~adata.obs["outlier_pct_counts_mt"])
    )
    
    adata_filtered = adata[adata.obs["keep_cells"], :].copy()
    sc.pp.filter_genes(adata_filtered, min_cells=3)
    
    cells_removed = adata.n_obs - adata_filtered.n_obs
    genes_removed = adata.n_vars - adata_filtered.n_vars
    
    print(f"[LOG] Removed {cells_removed} cells and {genes_removed} genes.")
    print(f"[LOG] Filtered dimensions: {adata_filtered.n_obs} cells x {adata_filtered.n_vars} genes.")
    
    return adata_filtered

def remove_multiplets(adata: ad.AnnData) -> ad.AnnData:
    """
    Identifies and removes multiplet droplets using the Scrublet algorithm.

    Parameters
    ----------
    adata : ad.AnnData
        The structurally filtered expression matrix containing viable cells.

    Returns
    -------
    ad.AnnData
        The matrix strictly cleansed of multi-cell droplets.
    """
    print("\n[LOG] Initializing Scrublet for multiplet detection.")
    
    sc.pp.scrublet(adata)
    
    doublet_mask = adata.obs['predicted_doublet']
    doublet_count = doublet_mask.sum()
    
    print(f"[LOG] Scrublet identified {doublet_mask.sum()} potential multiplets.")
    
    sc.pl.scrublet_score_distribution(
        adata, 
        show=False, 
        save="_doublet_distribution.svg"
    )
    
    adata_filtered = adata[~doublet_mask].copy()
    
    print(f"[LOG] Multiplets removed. New dimensions: {adata_filtered.n_obs} cells x {adata_filtered.n_vars} genes.")
    
    return adata_filtered

def execute_qc_pipeline(
    mtx_path: str, 
    pbmc3k_qc_h5ad_path: str, 
    v_keys: list = ['n_genes_by_counts', 'total_counts', 'pct_counts_mt']
) -> None:
    """
    Executes the complete quality control, filtering, and normalization pipeline.

    Parameters
    ----------
    mtx_path : str
        The path to the raw 10x Genomics matrix folder.
    pbmc3k_qc_h5ad_path : str
        The target save path for the finalized '.h5ad' artifact.
    v_keys : list, default ['n_genes_by_counts', 'total_counts', 'pct_counts_mt']
        The standard list of metrics to audit.

    Returns
    -------
    None
    """
    print("\n[LOG] Initiating Phase I QC pipeline.")
    
    adata = load_matrix(mtx_path)
    adata = compute_quality_metrics(adata)
    
    plot_qc_distributions(
        adata, v_keys, "pre_filter", 
        "total_counts", "n_genes_by_counts", "pct_counts_mt"
    )
    
    adata_filtered = apply_quality_filters(adata)
    adata_filtered = remove_multiplets(adata_filtered)
    
    adata_filtered.layers['counts'] = adata_filtered.X.copy()
    
    # Generate the normalized log1p layer
    adata_temp = adata_filtered.copy()
    sc.pp.normalize_total(adata_temp, target_sum=1e4)
    sc.pp.log1p(adata_temp)
    adata_filtered.layers['log1p_norm'] = adata_temp.X.copy()
    
    plot_qc_distributions(
        adata_filtered, v_keys, "post_filter", 
        "total_counts", "n_genes_by_counts", "pct_counts_mt"
    )
    
    print(f"[LOG] Saving processed matrix to {pbmc3k_qc_h5ad_path}")
    adata_filtered.write_h5ad(pbmc3k_qc_h5ad_path, compression='gzip')
    
    del adata, adata_filtered, adata_temp
    gc.collect()



def main():
    
    sc.settings.figdir = "./results/figures/p03_qc_filtering"
    os.makedirs(sc.settings.figdir, exist_ok=True)
    mtx_path = "data/raw/pbmc3k_filtered_gene_bc_matrices/hg19"
    pbmc3k_qc_h5ad_path = "data/objects/pbmc3k_qc.h5ad"

    if not os.path.exists(mtx_path):
        raise FileNotFoundError(f"[ERROR] Input matrix missing at {mtx_path}.")
    
    os.makedirs(os.path.dirname(pbmc3k_qc_h5ad_path), exist_ok=True)

    execute_qc_pipeline(
        mtx_path=mtx_path, 
        pbmc3k_qc_h5ad_path=pbmc3k_qc_h5ad_path
    )
    
    print("\n[LOG] Phase I complete.")

if __name__ == "__main__":
    main()