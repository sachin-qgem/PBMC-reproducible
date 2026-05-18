import gc
import json
import os
import warnings
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from scipy.stats import entropy
import igraph as ig
from math import log2
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import train_test_split


sc.settings.verbosity = 0
plt_fig_dir = Path('./results/figures/p04_clustering')
plt_fig_dir.mkdir(parents=True, exist_ok=True)



def load_h5ad_matrix(h5ad_path: str) -> ad.AnnData:
    """
    Loads an AnnData format matrix from disk.

    Parameters
    ----------
    h5ad_path : str
        The absolute or relative path to the .h5ad file.

    Returns
    -------
    ad.AnnData
        The loaded expression matrix.
    """
    print(f"[LOG] Loading matrix: {h5ad_path}")
    adata = sc.read_h5ad(h5ad_path)
    print(f"[LOG] Matrix loaded. Dimensions: {adata.shape}")
    
    return adata


def cell_cycle_check(
    h5ad_path: str, 
    cell_cycle_genes_path: str, 
    n_neighbors: int, 
    n_pcs: int, 
    leiden_res: float, 
    file_save_key: str
) -> None:
    """
    Evaluates cell cycle gene expression to verify clustering variance drivers.

    Parameters
    ----------
    h5ad_path : str
        Path to the target AnnData file.
    cell_cycle_genes_path : str
        Path to the text file containing cell cycle markers.
    n_neighbors : int
        Number of nearest neighbors for the UMAP graph.
    n_pcs : int
        Number of principal components to utilize.
    leiden_res : float
        Resolution for temporary clustering.
    file_save_key : str
        Prefix for the saved UMAP plot.

    Returns
    -------
    None
    """
    print(f"[LOG] Cell cycle evaluation initiated: {file_save_key}")
    adata = load_h5ad_matrix(h5ad_path)
    
    with open(cell_cycle_genes_path, 'r') as f:
        cell_cycle_genes = [x.strip() for x in f.readlines()]

    s_genes = [x for x in cell_cycle_genes[:43] if x in adata.var_names]
    g2m_genes = [x for x in cell_cycle_genes[43:] if x in adata.var_names]
    
    try:
        sc.pp.pca(adata)
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs, method='umap')
        sc.tl.umap(adata)
        sc.tl.leiden(adata, resolution=leiden_res)
        
        sc.tl.score_genes_cell_cycle(
            adata, s_genes=s_genes, g2m_genes=g2m_genes, layer='log1p_norm'
        )
        
        save_path = f"{file_save_key}_cell_cycle.svg"
        sc.pl.umap(
            adata, 
            color=['S_score', 'G2M_score', 'leiden'],
            layer='log1p_norm', 
            size=10.0, 
            legend_loc='on data',
            legend_fontsize='x-small', 
            legend_fontweight='bold',
            legend_fontoutline=3, 
            save=f"_{save_path}", 
            show=False
        )
        
        print("[LOG] Cell cycle check complete. Review plots for potential regression.")
    except:
        print("[ERROR] Cell cycle genes did not align with matrix variables. Evaluation bypassed. ")
    del adata
    gc.collect()

def random_split_data(h5ad_path: str, save_folder_path: str) -> dict:
    """
    Partitions the matrix into equal training and projection subsets.

    Parameters
    ----------
    h5ad_path : str
        Path to the master QC'd AnnData file.
    save_folder_path : str
        Directory to save the resulting split matrices.

    Returns
    -------
    dict
        Dictionary containing paths to the 'training_file' and 'projectable_file'.
    """
    print("[LOG] Data partitioning initiated (50/50).")
    adata = load_h5ad_matrix(h5ad_path)
    
    train_indices, project_indices = train_test_split(
        adata.obs_names, test_size=0.5, random_state=42, shuffle=True
    )
    
    adata_train = adata[train_indices].copy()
    adata_project = adata[project_indices].copy()
    
    train_path = f"{save_folder_path}/adata_train.h5ad"
    project_path = f"{save_folder_path}/adata_project.h5ad"
    
    adata_train.write_h5ad(train_path)
    adata_project.write_h5ad(project_path)
    
    del adata_project, adata_train, adata
    gc.collect()
    
    print("[LOG] Partitioning completed. Paths secured.")
    return {'training_file': train_path, 'projectable_file': project_path}

def calculate_jaccard_stability(
        adata: ad.AnnData,
        leiden_key: str,
        k: int,
        r: float,
        n_pcs: int
) :
    """
    Calculates cluster stability via iterative Jaccard index evaluation.
    """
    n_iterations = 20
    subsample_fraction = 0.8
    np.random.seed(42)
    
    original_labels = adata.obs[leiden_key].astype(str)
    unique_clusters = original_labels.unique()
    jaccard_ledger = {cluster: [] for cluster in unique_clusters}
    
    boot_k = max(2, int(k * subsample_fraction))
    
    for i in range(n_iterations):
        n_keep = int(adata.n_obs * subsample_fraction)
        surviving_indices = np.random.choice(
            adata.obs_names, size=n_keep, replace=False
        )
        
        adata_sub = adata[surviving_indices].copy()
        
        sc.pp.neighbors(
            adata_sub, n_neighbors=boot_k, n_pcs=n_pcs, method='umap',
            knn=True, metric='euclidean', random_state=42,
            use_rep='X_pca', key_added='boot_neighbors'
        )
        
        sc.tl.leiden(
            adata_sub, resolution=r, neighbors_key='boot_neighbors',
            key_added='boot_leiden', random_state=42
        )
        
        new_labels = adata_sub.obs['boot_leiden'].astype(str)
        
        for orig_cluster in unique_clusters:
            
            mask = (original_labels[surviving_indices] == orig_cluster).values
            orig_cells_in_sub = adata_sub.obs_names[mask]
            
            if len(orig_cells_in_sub) == 0:
                jaccard_ledger[orig_cluster].append(0.0)
                continue
                
            best_match_cluster = new_labels[orig_cells_in_sub].value_counts().index[0]
            
            set_A = set(orig_cells_in_sub)
            set_B = set(adata_sub.obs_names[new_labels == best_match_cluster])
            
            union_len = len(set_A.union(set_B))
            jaccard_score = len(set_A.intersection(set_B)) / union_len if union_len > 0 else 0.0
            
            jaccard_ledger[orig_cluster].append(jaccard_score)
            
        del adata_sub
        gc.collect()
    final_scores = {}
    for cluster, scores in jaccard_ledger.items():
        final_scores[str(cluster)] = float(np.mean(scores))
        
    return final_scores


def test_jaccard_stability(filepath: str, k: int, r: float) -> dict:
    """
    API endpoint for UI: Executes isolated parameter stability evaluation.
    """
    print(f"\n[LOG] Parameter evaluation initiated: k={k}, r={r}...")
    adata = load_h5ad_matrix(filepath)
    
    sc.pp.neighbors(adata, n_neighbors=k, n_pcs=10, method='umap', knn=True, random_state=42)
    sc.tl.leiden(adata, resolution=r, random_state=42, key_added='temp_test_leiden')
    
    
    final_scores = calculate_jaccard_stability(adata, leiden_key='temp_test_leiden', k=k, r=r,n_pcs=10)
    
    del adata
    import gc
    gc.collect()
    
    return final_scores


def compute_knn_umap_leiden(
    training_side_file_path: str, 
    n_neighbors: int, 
    n_pcs: int, 
    leiden_res: float, 
    key_name: str,
    embedding_dots_size: float
) -> tuple:
    """
    Computes neighborhood graphs, UMAP coordinates, and Leiden clusters.

    Parameters
    ----------
    training_side_file_path : str
        Path to the target AnnData file.
    n_neighbors : int
        Number of nearest neighbors.
    n_pcs : int
        Number of principal components.
    leiden_res : float
        Resolution for baseline clustering.
    key_name : str
        Prefix for generated observation columns (e.g., 'macro').

    Returns
    -------
    tuple
        (leiden_key, neighbors_key) generated during the operation.
    """
    adata = load_h5ad_matrix(training_side_file_path)
    
    leiden_key, neighbors_key, umap_key_added = None, None, None
    
    if adata.n_obs > n_neighbors +2:
        neighbors_key = f"{key_name}_neighbors"
        umap_key_added = f"{key_name}_umap"
        leiden_key = f"{key_name}_leiden"
        
        sc.pp.neighbors(
            adata, n_neighbors=n_neighbors, n_pcs=n_pcs, method='umap',
            knn=True, metric='euclidean', random_state=42, key_added=neighbors_key
        )
        sc.tl.umap(
            adata, maxiter=500, random_state=42, key_added=umap_key_added,
            neighbors_key=neighbors_key, min_dist=0.1, spread=1.0
        )
        sc.tl.leiden(
            adata, resolution=leiden_res, n_iterations=-1, flavor='leidenalg', 
            random_state=42, key_added=leiden_key, neighbors_key=neighbors_key
        )
        plot_basis = umap_key_added if umap_key_added is not None else 'X_umap'
        sc.pl.embedding(adata,basis=plot_basis,color=leiden_key,
                components ='all',size= embedding_dots_size,color_map = 'Blues',show=False,
                title = 'Training Manifold',legend_loc = 'on data',
                legend_fontsize = 'x-small',legend_fontweight = 'bold',
                legend_fontoutline = 3,save=f".svg")
        
        # Jaccard Test for Sampling and Uncertainity
        adata_su_check = adata.copy()
        print(f"\n[LOG] Subsampling stability evaluation for '{key_name}'...")
        
        
        jaccard_ledger = calculate_jaccard_stability(adata=adata_su_check,leiden_key=leiden_key,
                                                       k=n_neighbors,r=leiden_res,n_pcs=n_pcs)
        print(f" JACCARD UNCERTAINTY : {key_name} ---")
        
        
        for orig_cluster, mean_score in jaccard_ledger.items():
            if mean_score >= 0.85:
                grade = "[HIGH STABILITY]"
            elif mean_score >= 0.60:
                grade = "[MODERATE STABILITY]"
            else:
                grade = "[LOW STABILITY]"
                
            print(f"  Cluster {orig_cluster}: Jaccard = {mean_score:.3f} {grade}")
            
            
        
        adata.uns[f'{leiden_key}_SU_grades'] = jaccard_ledger
        adata.write_h5ad(training_side_file_path)
        
    del adata, adata_su_check
    gc.collect()
    
    return leiden_key, neighbors_key
           

def calculate_modularity_grid(
    filepath: str, 
    key_name: str, 
    k_grid: list, 
    r_grid: list,
    plt_fig_dir: str,
    n_pcs: int = 10
) -> tuple:
    """
    Calculates igraph modularity across parameter coordinates to identify cluster stability.
    """
    import numpy as np
    import pandas as pd
    import igraph as ig
    import scanpy as sc
    import matplotlib.pyplot as plt
    import seaborn as sns
    import gc
    
    label = key_name.upper()
    print(f"\n[LOG] Initiating grid search optimization for graph modularity ({label})...")
    
    adata_raw = sc.read_h5ad(filepath)
    n_cells = adata_raw.n_obs
    actual_pcs = min(n_pcs, n_cells - 1, adata_raw.obsm['X_pca'].shape[1] if 'X_pca' in adata_raw.obsm else n_pcs)
    
    results_list = []

    
    for k in k_grid:
        if k >= n_cells: 
            continue
            
        sc.pp.neighbors(adata_raw, n_neighbors=k, n_pcs=actual_pcs, knn=True, use_rep='X_pca')
        adj = adata_raw.obsp['connectivities']
        sources, targets = adj.nonzero()
        
        g = ig.Graph(n=n_cells, directed=False)
        g.add_edges(list(zip(sources, targets)))
        g.es['weight'] = adj[sources, targets].A1
        
        for r in r_grid:
            sc.tl.leiden(adata_raw, resolution=r, key_added='temp_cluster', flavor='leidenalg')
            labels = adata_raw.obs['temp_cluster'].astype(str)
            n_clusters = len(labels.unique())
            
            map_dict = {l: i for i, l in enumerate(labels.unique())}
            membership = [map_dict[l] for l in labels]
            
            # Calculate the structural modularity (Q) of the resulting partition
            modularity_val = ig.VertexClustering(g, membership=membership).modularity
            
            results_list.append({
                'k_neighbors': k, 
                'resolution_r': r, 
                'modularity': modularity_val, 
                'n_clusters': n_clusters
            })

    df = pd.DataFrame(results_list)
    if df.empty: 
        print(f"[{label}] WARNING: Grid failure. Defaulting to safe baseline.")
        return 30, 0.05

    
    pivot_modularity = df.pivot(index='k_neighbors', columns='resolution_r', values='modularity')
    pivot_clusters = df.pivot(index='k_neighbors', columns='resolution_r', values='n_clusters')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))

    sns.heatmap(
        pivot_modularity, annot=pivot_clusters, fmt=".0f", cmap="viridis", 
        cbar_kws={'label': 'Structural Modularity (Q)'},
        annot_kws={"size": 12, "weight": "bold"}, ax=ax1
    )
    ax1.set_title(f"[{label}] Biological State Map\n[Numbers = Clusters | Color = Modularity]", fontweight='bold')
    ax1.set_xlabel("Leiden Resolution (Temperature, r)", fontweight='bold')
    ax1.set_ylabel("KNN Neighbors (Scaffolding, k)", fontweight='bold')

    X = pivot_modularity.columns.values
    Y = pivot_modularity.index.values
    Z = pivot_modularity.values
    z_min = np.nanmin(pivot_modularity.values)
    z_max = np.nanmax(pivot_modularity.values)
    high_res_levels = np.linspace(z_min, z_max, 100)

    contour_filled = ax2.contourf(X, Y, Z, levels=high_res_levels, cmap="viridis",vmin=z_min,vmax=z_max)
    
    contour_lines = ax2.contour(X, Y, Z, levels=high_res_levels, colors='black', linewidths=0.8, alpha=0.7)
    ax2.clabel(contour_lines, inline=True, fontsize=9, fmt='%.2f')

    fig.colorbar(contour_filled, ax=ax2, label='Graph Modularity (Q')
    ax2.set_title(f"[{label}] Parameter Modularity Coordinates\n[Wide Spacing = Stable | Tight Lines = Volatile Cliff]", fontweight='bold')
    ax2.set_xlabel("Leiden Resolution (Temperature, r)", fontweight='bold')
    ax2.set_ylabel("KNN Neighbors (Scaffolding, k)", fontweight='bold')
    
    ax2.invert_yaxis()

    plt.tight_layout()
    
    svg_path = f"{plt_fig_dir}/{key_name}_grid_search_surface.svg"
    plt.savefig(svg_path, format="svg")
    plt.close()


    state_metrics = []
    for n_state in df['n_clusters'].unique():
        df_state = df[df['n_clusters'] == n_state]
        state_metrics.append({
            'n_clusters': n_state, 
            'area': len(df_state), 
            'elevation': df_state['modularity'].quantile(0.75)
        })
        
    df_metrics = pd.DataFrame(state_metrics)
    
  
    df_metrics['area_norm'] = df_metrics['area'] / df_metrics['area'].max()
    df_metrics['elevation_norm'] = df_metrics['elevation'] / df_metrics['elevation'].max()
    df_metrics['mesa_score'] = df_metrics['area_norm'] * df_metrics['elevation_norm']
    
   
    target_n_clusters = int(df_metrics.loc[df_metrics['mesa_score'].idxmax()]['n_clusters'])
    
    df_winner = df[df['n_clusters'] == target_n_clusters]
    df_flat_top = df_winner[df_winner['modularity'] >= df_winner['modularity'].median()].copy()
    
   
    theoretical_k = df_flat_top['k_neighbors'].mean()
    theoretical_r = df_flat_top['resolution_r'].mean()
    
    k_min, k_max = df['k_neighbors'].min(), df['k_neighbors'].max()
    r_min, r_max = df['resolution_r'].min(), df['resolution_r'].max()
    
    
    df_flat_top['dist'] = df_flat_top.apply(
        lambda row: np.sqrt(
            ((row['k_neighbors'] - theoretical_k) / (k_max - k_min + 1e-9))**2 + 
            ((row['resolution_r'] - theoretical_r) / (r_max - r_min + 1e-9))**2
        ), axis=1
    )
    
    anchor = df_flat_top.loc[df_flat_top['dist'].idxmin()]
    final_k = int(anchor['k_neighbors'])
    final_r = float(anchor['resolution_r'])
    
    print(f"[{label}] Default Suggestion: k={final_k}, r={final_r:.4f}")
    
    del adata_raw
    gc.collect()

    return final_k, final_r

def evaluate_subcluster_variance(adata: ad.AnnData, min_cells: int = 100, variance_threshold: float = 3.5) -> bool:
    """
    Evaluates PCA eigenvalue distributions to determine if cluster subdivisions are statistically supported.
    """
    
    if adata.n_obs < min_cells:
        return True


    n_comps = min(50, adata.n_obs - 1)
    if 'pca' not in adata.uns or 'variance_ratio' not in adata.uns['pca']:
        try:
            sc.experimental.pp.highly_variable_genes(
                adata, theta=100.0, n_top_genes=2000, flavor='pearson_residuals', subset=False
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                sc.experimental.pp.normalize_pearson_residuals(adata, theta=100.0)
                sc.pp.pca(adata, n_comps=n_comps, zero_center=True, svd_solver='arpack', mask_var='highly_variable')
        except Exception:
            return True

    variance_ratios = adata.uns['pca']['variance_ratio']
    
    if len(variance_ratios) < 10:
        return True 

    pc1_energy = variance_ratios[0]
    start_noise = 10
    end_noise = min(25, len(variance_ratios))
    if end_noise <= start_noise:
        pc_baseline_energy = np.median(variance_ratios[max(1, len(variance_ratios)//2):])
    else:
        pc_baseline_energy = np.median(variance_ratios[start_noise:end_noise])
     
    structural_ratio = pc1_energy / pc_baseline_energy


    if pc1_energy < 0.02 or structural_ratio < variance_threshold:
        return True  
    
    return False

def subset_and_save_clusters(
    file_path_used: str, 
    leiden_key_as_cluster_column_name: str,
    enforce_variance_audit: bool = True
) -> dict:
    """
    Subsets the expression matrix according to cluster labels and saves vectors independently.

    Parameters
    ----------
    file_path_used : str
        Path to the parent AnnData file.
    leiden_key_as_cluster_column_name : str
        The observation column containing the clustering assignments.

    Returns
    -------
    dict
        A mapping of cluster-specific keys to their new physical file paths.
    """
    print(f"[LOG] Subsetting matrix vectors based on parameter: {leiden_key_as_cluster_column_name}")
    adata = load_h5ad_matrix(file_path_used)
    grouped_barcodes = adata.obs.groupby(leiden_key_as_cluster_column_name).groups
    
    saved_filepaths_dictionary = {}
    path_object = Path(file_path_used)
    directory = path_object.parent
    base_name = path_object.stem
    
    for cluster_id, barcodes in grouped_barcodes.items():
        lineage_key = f"{leiden_key_as_cluster_column_name}_{cluster_id}"
        adata_subset = adata[barcodes].copy()
        if  enforce_variance_audit:
            if evaluate_subcluster_variance(adata_subset):
                print(f"[LOG] Parameter variance limit reached: Cluster {cluster_id} (N={adata_subset.n_obs}). Halted further Clustering")
                print(" [SUGGESTION] If a continuous transition is suspected, consider Trajectory Inference methods.")
                lineage_key = f"{lineage_key}_Terminal_State"
            else:
                print(f"[LOG] Cluster {cluster_id} (N={adata_subset.n_obs}): High internal variance detected. Proceeding to sub-clustering.")
        else:
            print(f"[LOG] Subsetting Cluster {cluster_id} without variance evaluation.")   
        new_filename = f"{base_name}_{lineage_key}.h5ad"
        new_filepath_obj = directory / new_filename
        
        adata_subset.write_h5ad(str(new_filepath_obj))
        saved_filepaths_dictionary[lineage_key] = str(new_filepath_obj)
        
        del adata_subset
        gc.collect()

    del adata
    gc.collect()
    return saved_filepaths_dictionary


def npr_hvg_pca_recal(filepath: str, keys: str) -> None:
    """
    Recalculates Pearson Residuals, HVG and PCA components for isolated subset matrices.

    Parameters
    ----------
    filepath : str
        Path to the subset AnnData file.
    keys : str
        Prefix for plotting outputs.

    Returns
    -------
    None
    """
    print(f"[LOG] Recalculating variance components for: {keys}")
    adata = load_h5ad_matrix(filepath)
    adata.X = adata.layers['counts'].copy()
    
    if adata.n_obs > 50:
        n_comps = min(50,adata.n_obs-1)
        sc.experimental.pp.highly_variable_genes(
            adata, theta=100.0, n_top_genes=2500, flavor='pearson_residuals', subset=False
        )
        mt_mask = adata.var['mt']
        ribo_mask = adata.var['ribo']

        del_count = (adata.var['highly_variable'] & (mt_mask | ribo_mask)).sum()
        print(f"[LOG] Removing {del_count} mitochondrial/ribosomal genes from highly variable set prior to PCA.")

        adata.var.loc[mt_mask | ribo_mask, 'highly_variable'] = False
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
        sc.experimental.pp.normalize_pearson_residuals(adata, theta=100.0)
        sc.pp.pca(
            adata, n_comps=n_comps, zero_center=True, svd_solver='arpack', mask_var='highly_variable'
        )
        
        sc.pl.pca_variance_ratio(adata, n_pcs=n_comps, save=f"_{keys}_.svg", show=False)
        adata.write_h5ad(filepath)
        
    del adata
    gc.collect()


def cast_projectable_data_on_training_data(
    adata_project_file_path: str, 
    adata_training_file_path: str, 
    neighbors_key_training: str, 
    leiden_key_training: str
) -> None:
    """
    Applies projection algorithms to map holdout data onto pre-computed training boundaries.

    Parameters
    ----------
    adata_project_file_path : str
        Path to the AnnData file awaiting projection.
    adata_training_file_path : str
        Path to the reference AnnData file.
    neighbors_key_training : str
        The neighborhood graph key in the reference object.
    leiden_key_training : str
        The clustering label key in the reference object.

    Returns
    -------
    None
    """
    print(f"[LOG] Executing matrix projection mapping: {leiden_key_training}")
    adata_project = load_h5ad_matrix(adata_project_file_path)
    adata_training = load_h5ad_matrix(adata_training_file_path)
    
    adata_project.X = adata_project.layers['counts'].copy()
    
    counts_A = adata_training.layers['counts']
    gene_sums_A = np.asarray(counts_A.sum(axis=0)).flatten()
    total_sum_A = counts_A.sum()
    p_j_A = gene_sums_A / total_sum_A 
    
    counts_B = adata_project.layers['counts']
    cell_depths_B = np.asarray(counts_B.sum(axis=1)).flatten()
    expected_B = np.outer(cell_depths_B, p_j_A)
    
    theta = 100.0
    variance_B = expected_B + (np.square(expected_B) / theta)
    
    variance_B = np.clip(variance_B, a_min=1e-12, a_max=None)

    dense_counts_B = counts_B.toarray() if hasattr(counts_B, 'toarray') else np.asarray(counts_B)
    residuals_B = (dense_counts_B - expected_B) / np.sqrt(variance_B)
    
    clip_threshold = np.sqrt(adata_training.n_obs)
    residuals_B = np.clip(residuals_B, -clip_threshold, clip_threshold)
    
    adata_project.X = residuals_B
    
    sc.tl.ingest(
        adata=adata_project, adata_ref=adata_training, obs=leiden_key_training, 
        embedding_method=['pca'], labeling_method='knn', 
        neighbors_key=neighbors_key_training, inplace=True
    )
    
    try:
        pca_train_x = adata_training.obsm['X_pca'][:, 0]
        pca_train_y = adata_training.obsm['X_pca'][:, 1]
        pca_proj_x = adata_project.obsm['X_pca'][:, 0]
        pca_proj_y = adata_project.obsm['X_pca'][:, 1]
        
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.scatter(
            pca_train_x, pca_train_y, c='lightgray', s=30, alpha=0.8, edgecolors='none', 
            label=f'Reference (N={adata_training.n_obs})'
        )
        ax.scatter(
            pca_proj_x, pca_proj_y, c='#0033a0', s=15, alpha=0.6, edgecolors='none', 
            label=f'Projected (N={adata_project.n_obs})'
        )
        ax.set_xlabel('Principal Component 1', fontweight='bold')
        ax.set_ylabel('Principal Component 2', fontweight='bold')
        ax.set_title(f'Projection Overlap : {leiden_key_training}', fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
        legend = ax.legend(loc='best', frameon=True, shadow=True)
        for handle in legend.legend_handles:
            handle.set_alpha(1.0)
            
        plt.tight_layout()
        plt.savefig(f'{plt_fig_dir}/{leiden_key_training}_Projection_Overlap.svg')
        
    except KeyError:
        print("[WARNING] 'X_pca' missing. Visual projection bypassed.")
        
    adata_project.write_h5ad(adata_project_file_path)
    del adata_project, adata_training
    gc.collect()



def execute_macro_sweep(h5ad_path: str, save_folder_path: str) -> dict:
    """
    API endpoint for UI: Executes initial grid search across parameter space.
    """
    sc.settings.figdir = str(plt_fig_dir)
    
    print("[LOG] Phase II parameter grid search initiated over parameters k and r.")
    
    file_path_dict = random_split_data(h5ad_path, save_folder_path)
    training_file_path = file_path_dict['training_file']
    
    npr_hvg_pca_recal(training_file_path, 'training_file')
    
    macro_k_grid = np.arange(5, 105, 5).tolist()
    macro_r_grid = np.round(np.arange(0.01, 0.21, 0.02), 2).tolist()
    
    suggested_k, suggested_r = calculate_modularity_grid(
        filepath=training_file_path, 
        key_name='macro', 
        k_grid=macro_k_grid, 
        r_grid=macro_r_grid, 
        plt_fig_dir=str(plt_fig_dir), 
        n_pcs=10
    )
    
    print("\n[LOG] Grid Search Complete. Halting backend. Please Enter the optimal pair (K,r) you have finalised ")
    
    return {
        'training_file_path': training_file_path,
        'file_path_dict': file_path_dict,
        'suggested_k': suggested_k,
        'suggested_r': suggested_r
    }


def lock_macro_and_extract_micro_queue(
    training_file_path: str, 
    human_k: int, 
    human_r: float, 
    cell_cycle_genes_path: str
) -> dict:
    """
    API endpoint for UI: Applies approved parameters and subsets matrix.
    """
    human_r = round(human_r,2)
    sc.settings.figdir = str(plt_fig_dir)

    print(f"\n[INFO] Locking Macro-State at k={human_k}, r={human_r}")
    
    cell_cycle_check(
        training_file_path, cell_cycle_genes_path, n_neighbors=human_k, 
        n_pcs=10, leiden_res=human_r, file_save_key='training'
    )
    
    macro_leiden_key, macro_neighbors_key = compute_knn_umap_leiden(
        training_file_path, n_neighbors=human_k, n_pcs=10, 
        leiden_res=human_r, key_name='macro',embedding_dots_size=5.0
    )
    
    micro_filepaths_dict = subset_and_save_clusters(
        training_file_path, macro_leiden_key
    )
    
    return {
        'macro_leiden_key': macro_leiden_key,
        'macro_neighbors_key': macro_neighbors_key,
        'micro_filepaths_dict': micro_filepaths_dict
    }

def execute_micro_sweep(filepath: str, micro_key: str, plt_fig_dir: str) -> dict:
    """
    API endpoint for UI: Executes sub-cluster grid search.
    """
    sc.settings.figdir = str(plt_fig_dir)
    print(f"\n[SYSTEM] Sweeping Micro-State: {micro_key}")
    npr_hvg_pca_recal(filepath, micro_key)
    
    micro_k_grid = np.arange(5, 65, 5).tolist()
    micro_r_grid = np.round(np.arange(0.1, 2.10, 0.2), 2).tolist()
    
    suggested_k, suggested_r = calculate_modularity_grid(
        filepath=filepath, key_name=micro_key, k_grid=micro_k_grid, 
        r_grid=micro_r_grid, plt_fig_dir=plt_fig_dir, n_pcs=10
    )
    suggested_r = round(suggested_r,2)
    return {'suggested_k': suggested_k, 'suggested_r': suggested_r}

def lock_micro_state(
    filepath: str, micro_key: str, human_k: int, human_r: float, cell_cycle_genes_path: str
) -> dict:
    """
    API endpoint for UI: Applies approved sub-cluster parameters.
    """
    human_r = round(human_r,2)
    sc.settings.figdir = str(plt_fig_dir)
    print(f"\n[LOG] Locking Micro-State {micro_key} at k={human_k}, r={human_r}")
    cell_cycle_check(
        filepath, cell_cycle_genes_path, n_neighbors=human_k,
        n_pcs=10, leiden_res=human_r, file_save_key=micro_key
    )
    
    m_leiden, m_neighbors = compute_knn_umap_leiden(
        filepath, n_neighbors=human_k, n_pcs=10, 
        leiden_res=human_r, key_name=f'{micro_key}_micro',embedding_dots_size=40.0
    )
    
    
    return {'m_leiden': m_leiden, 'm_neighbors': m_neighbors}

def project_subclusters(dict_A: dict) -> dict:
    """
    Applies cluster vectors to holdout data structure.
    """
    macro_project_path = dict_A['file_path_dictionary_from_the_split_step']['projectable_file']
    macro_training_path = dict_A['file_path_dictionary_from_the_split_step']['training_file']
    macro_neighbors_key = dict_A['training_macro_neighbors_key']
    macro_leiden_key = dict_A['training_macro_leiden_key']
    
    cast_projectable_data_on_training_data(
        macro_project_path, macro_training_path, macro_neighbors_key, macro_leiden_key
    )
    
    projected_micro_file_dict = subset_and_save_clusters(
        macro_project_path, macro_leiden_key,enforce_variance_audit=False
    )
    
    training_micro_file_dict = dict_A['training_micro_file_path_dictionary']
    project_micro_leiden_dict = {}
    project_micro_neighbors_dict = {}
    
    for keys_projected, projected_micro_filepath in projected_micro_file_dict.items():
        root_key = keys_projected.replace('_Terminal_State', '')
        target_key_training = None
        
        if root_key in training_micro_file_dict:
            target_key_training = root_key
        elif f'{root_key}_Terminal_State' in training_micro_file_dict:
            target_key_training = f'{root_key}_Terminal_State'
            
        if target_key_training is not None:
            training_micro_filepath = training_micro_file_dict[target_key_training]
            if 'Terminal_State' in target_key_training:
                continue
                
            p_leiden = dict_A['training_micro_leiden_key_dictionary'].get(target_key_training)
            p_neighbors = dict_A['training_micro_neighbors_key_dictionary'].get(target_key_training)
            
            if p_leiden and p_neighbors:
                cast_projectable_data_on_training_data(
                    projected_micro_filepath, training_micro_filepath, p_neighbors, p_leiden
                )
                project_micro_leiden_dict[keys_projected] = p_leiden
                project_micro_neighbors_dict[keys_projected] = p_neighbors
                
    return {
        'macro_adata_project_file_path': macro_project_path,
        'macro_neighbors_key_training': macro_neighbors_key,
        'macro_leiden_key_training': macro_leiden_key,
        'projected_micro_file_path_dictionary': projected_micro_file_dict,
        'projected_micro_leiden_key_dictionary': project_micro_leiden_dict,
        'projected_micro_neighbors_key_dictionary': project_micro_neighbors_dict
    }

def seal_phase_II_pipeline(
    h5ad_path: str, save_folder_path: str, file_path_dict: dict,
    macro_leiden_key: str, macro_neighbors_key: str,
    micro_filepaths_dict: dict, micro_leiden_dict: dict, micro_neighbors_dict: dict
) -> None:
    """
    API endpoint for UI: Compiles parameter dictionaries and executes final matrix projections.
    """
    print("\n[LOG] Exporting parameter variables to disk.")
    dict_A = {
        'main_pca_artifact_path': h5ad_path,
        'writing_files_folder_path': save_folder_path,
        'file_path_dictionary_from_the_split_step': file_path_dict,
        'training_macro_leiden_key': macro_leiden_key,
        'training_macro_neighbors_key': macro_neighbors_key,
        'training_micro_file_path_dictionary': micro_filepaths_dict,
        'training_micro_leiden_key_dictionary': micro_leiden_dict,
        'training_micro_neighbors_key_dictionary': micro_neighbors_dict
    }
    
    dict_B = project_subclusters(dict_A)
    
    import json
    with open(f'{save_folder_path}/Dictionary_of_returns_from_orch_A.json', 'w') as f:
        json.dump(dict_A, f, indent=4)
    with open(f'{save_folder_path}/Dictionary_of_returns_from_orch_B.json', 'w') as f:
        json.dump(dict_B, f, indent=4)
        
    print("[LOG] Data structures established.")