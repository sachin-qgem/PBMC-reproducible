import gc
import os
from typing import Literal

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.stats as sps

# Global environment settings
ad.settings.allow_write_nullable_strings = True
sc.settings.figdir = "./results/figures/p03_qc_filtering"
os.makedirs(sc.settings.figdir, exist_ok=True)
sc.settings.verbosity = 3



def load_evidence(path: str) -> ad.AnnData:
    """
        Ingests raw 10x Genomics Matrix Market data into an AnnData tensor.

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
    print(f"[INFO] Matrix loaded successfully. Dimensions: {adata.shape}")

    return adata


def calculate_vital_signs(adata: ad.AnnData) -> ad.AnnData:
    """
    Calculates primary quality control metrics, focusing on mitochondrial 
    expression to establish cell viability thresholds.

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
    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(
        adata, 
        qc_vars=["mt"], 
        expr_type="counts",
        inplace=True, 
        log1p=False
    )
    
    return adata


def is_outlier(
    adata: ad.AnnData, 
    metric: str, 
    nmads: int, 
    side: Literal["both", "upper", "lower"] = "both"
) -> np.ndarray:
    """
    Calculates absolute topological boundaries using Median Absolute 
    Deviation (MAD) to identify physical cell outliers.

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



def audit_distribution(
    adata: ad.AnnData, 
    violin_keys: list, 
    stagename: Literal["pre_filter", "post_filter"],
    scatter_x: str = "total_counts", 
    scatter_y: str = "n_genes_by_counts", 
    scatter_color: str = "pct_counts_mt"
) -> None:
    """
    Generates and saves visual telemetry distributions for QC metrics.

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
    print(f"[AUDIT] Generating visual telemetry for {stagename} state...")
    
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



def apply_filter(adata: ad.AnnData) -> ad.AnnData:
    """
    Executes the thermodynamic purge, removing cellular outliers and 
    empty gene vectors based on robust 5-MAD thresholds.

    Parameters
    ----------
    adata : ad.AnnData
        The annotated expression matrix containing unmapped distributions.

    Returns
    -------
    ad.AnnData
        The filtered, structurally sound expression matrix.
    """
    print("[INFO] Executing 5-MAD cell purge and dormant gene removal...")
    
    adata.obs["outlier_n_genes_by_counts"] = is_outlier(
        adata, "n_genes_by_counts", 5, "both"
    )
    adata.obs["outlier_pct_counts_mt"] = is_outlier(
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
    
    print(f"[INFO] Removed {cells_removed} cells and {genes_removed} genes.")
    print(f"[SUCCESS] Final Dimensions: {adata_filtered.n_obs} cells x {adata_filtered.n_vars} genes.")
    
    return adata_filtered



def orch_qc_filtering(
    mtx_path: str, 
    pbmc3k_qc_h5ad_path: str, 
    v_keys: list = ['n_genes_by_counts', 'total_counts', 'pct_counts_mt']
) -> None:
    """
    The master orchestrator for Phase I. Loads raw data, executes QC 
    audits, applies thermodynamic filters, generates standard layers, 
    and seals the golden artifact to disk.

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
    print("\n===========================================================")
    print(" PHASE I (QUALITY CONTROL INITIATION)")
    print("===========================================================")
    
    adata = load_evidence(mtx_path)
    adata = calculate_vital_signs(adata)
    
    audit_distribution(
        adata, v_keys, "pre_filter", 
        "total_counts", "n_genes_by_counts", "pct_counts_mt"
    )
    
    adata_filtered = apply_filter(adata)
    
    # Secure the raw counts layer before normalization
    adata_filtered.layers['counts'] = adata_filtered.X.copy()
    
    # Generate the normalized log1p layer
    adata_temp = adata_filtered.copy()
    sc.pp.normalize_total(adata_temp, target_sum=1e4)
    sc.pp.log1p(adata_temp)
    adata_filtered.layers['log1p_norm'] = adata_temp.X.copy()
    
    audit_distribution(
        adata_filtered, v_keys, "post_filter", 
        "total_counts", "n_genes_by_counts", "pct_counts_mt"
    )
    
    print(f"[SEALED] Writing Golden Copy to {pbmc3k_qc_h5ad_path}...")
    adata_filtered.write_h5ad(pbmc3k_qc_h5ad_path, compression='gzip')
    
    del adata, adata_filtered, adata_temp
    gc.collect()



if __name__ == "__main__":
    # Define absolute topological paths
    mtx_path = "data/raw/pbmc3k_filtered_gene_bc_matrices/hg19"
    pbmc3k_qc_h5ad_path = "data/objects/pbmc3k_qc.h5ad"
    if not os.path.exists(mtx_path):
        raise FileNotFoundError(
            f"[CRITICAL FAILURE] Genesis matrix missing at {mtx_path}. "
            "Ensure the 10x raw files are staged before execution."
        )
    os.makedirs(os.path.dirname(pbmc3k_qc_h5ad_path), exist_ok=True)
    orch_qc_filtering(
        mtx_path=mtx_path, 
        pbmc3k_qc_h5ad_path=pbmc3k_qc_h5ad_path
    )
    
    print("\nPHASE I COMPLETE. MATRIX READY FOR FRACTURE.")
