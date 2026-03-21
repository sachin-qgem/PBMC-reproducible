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
import seaborn as sns
from math import log2
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import train_test_split

# Global environment settings
ad.settings.allow_write_nullable_strings = True
sc.settings.figdir = "./results/figures/p04_clustering"
os.makedirs(sc.settings.figdir, exist_ok=True)
sc.settings.verbosity = 0
plt_fig_dir = Path('./results/figures/p04_clustering')
plt_fig_dir.mkdir(parents=True, exist_ok=True)



def load_evidence(h5ad_path: str) -> ad.AnnData:
    """
    Loads an AnnData artifact from disk.

    Parameters
    ----------
    h5ad_path : str
        The absolute or relative path to the .h5ad file.

    Returns
    -------
    ad.AnnData
        The loaded expression matrix.
    """
    print(f"[INFO] Loading matrix from {h5ad_path}")
    adata = sc.read_h5ad(h5ad_path)
    print(f"[INFO] Loaded dimensions: {adata.n_obs} cells x {adata.n_vars} genes")
    
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
    Evaluates and plots cell cycle gene expression scores to determine 
    if regression is biologically necessary.

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
    print(f"[AUDIT] Executing cell cycle scoring for {file_save_key}...")
    adata = load_evidence(h5ad_path)
    
    with open(cell_cycle_genes_path, 'r') as f:
        cell_cycle_genes = [x.strip() for x in f.readlines()]
        
    s_genes = cell_cycle_genes[:43]
    g2m_genes = cell_cycle_genes[43:]
    
    # Filter genes to those physically present in the dataset
    s_genes = [x for x in s_genes if x in adata.var_names]
    g2m_genes = [x for x in g2m_genes if x in adata.var_names]
    
    sc.pp.pca(adata)
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs, method='umap')
    sc.tl.umap(adata)
    sc.tl.leiden(adata, resolution=leiden_res)
    
    sc.tl.score_genes_cell_cycle(
        adata, s_genes=s_genes, g2m_genes=g2m_genes, layer='log1p_norm'
    )
    
    save_path = f"{file_save_key}_cell_cycle.png"
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
    
    print("[INFO] Cell cycle check complete. Review plots for potential regression.")
    del adata
    gc.collect()

def random_split_data(h5ad_path: str, save_folder_path: str) -> dict:
    """
    Randomly fractures the global matrix into two equal halves for 
    training and projection validation.

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
    print("[INFO] Executing 50/50 randomized data split...")
    adata = load_evidence(h5ad_path)
    
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
    
    paths = {
        'training_file': train_path,
        'projectable_file': project_path
    }
    
    print("[SUCCESS] Split completed. Paths secured.")
    return paths


def knn_umap_leiden(
    training_side_file_path: str, 
    n_neighbors: int, 
    n_pcs: int, 
    leiden_res: float, 
    key_name: str
) -> tuple:
    """
    Computes baseline neighborhood graphs and UMAP embeddings. 
    Evaluates topological stability via subsampling and Jaccard overlap.

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
    adata = load_evidence(training_side_file_path)
    adata_su_check = adata.copy()
    
    leiden_key = None
    neighbors_key = None
    umap_key_added = None
    
    if adata.n_obs > 250:
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
                components ='all',size= 50.0,color_map = 'Blues',show=False,
                title = 'Training Manifold',legend_loc = 'on data',
                legend_fontsize = 'x-small',legend_fontweight = 'bold',
                legend_fontoutline = 3,save=f".png")
        
        print(f"\n[AUDIT] Subsampling stability evaluation for '{key_name}'...")
        n_iterations = 20
        subsample_fraction = 0.8
        
        original_labels = adata.obs[leiden_key].astype(str)
        unique_clusters = original_labels.unique()
        jaccard_ledger = {cluster: [] for cluster in unique_clusters}
        
        for i in range(n_iterations):
            n_keep = int(adata_su_check.n_obs * subsample_fraction)
            surviving_indices = np.random.choice(
                adata_su_check.obs_names, size=n_keep, replace=False
            )
            adata_sub = adata_su_check[surviving_indices].copy()
            
            sc.pp.neighbors(
                adata_sub, n_neighbors=n_neighbors, n_pcs=n_pcs, method='umap', 
                knn=True, metric='euclidean', random_state=42, 
                use_rep='X_pca', key_added='boot_neighbors'
            )
            sc.tl.leiden(
                adata_sub, resolution=leiden_res, neighbors_key='boot_neighbors', 
                key_added='boot_leiden', random_state=42
            )
            
            new_labels = adata_sub.obs['boot_leiden'].astype(str)
            
            for orig_cluster in unique_clusters:
                orig_cells_in_sub = adata_sub.obs_names[
                    original_labels[surviving_indices] == orig_cluster
                ]
                
                if len(orig_cells_in_sub) == 0:
                    jaccard_ledger[orig_cluster].append(0.0)
                    continue
                    
                best_match_cluster = new_labels[orig_cells_in_sub].value_counts().index[0]
                set_A = set(orig_cells_in_sub)
                set_B = set(adata_sub.obs_names[new_labels == best_match_cluster])
                
                intersection = len(set_A.intersection(set_B))
                union = len(set_A.union(set_B))
                jaccard_score = intersection / union if union > 0 else 0.0
                jaccard_ledger[orig_cluster].append(jaccard_score)
                
            del adata_sub
            gc.collect()

        print(f"--- JACCARD UNCERTAINTY DIAGNOSTIC: {key_name} ---")
        su_grades_for_disk = {}
        for orig_cluster, scores in jaccard_ledger.items():
            mean_score = np.mean(scores)
            if mean_score >= 0.85:
                grade = "[HIGH STABILITY]"
            elif mean_score >= 0.60:
                grade = "[MODERATE STABILITY]"
            else:
                grade = "[LOW STABILITY]"
                
            print(f" Cluster {orig_cluster}: Jaccard = {mean_score:.3f} {grade}")
            su_grades_for_disk[orig_cluster] = float(mean_score)
            
        adata.uns[f'{leiden_key}_SU_grades'] = su_grades_for_disk
        adata.write_h5ad(training_side_file_path)
        
    del adata, adata_su_check
    gc.collect()
    
    return leiden_key, neighbors_key
           
          
def stability_audit(training_filepath: str, key_for_saving_images: str,res_start: float,res_end: float,res_step: float,n_neighbors: int) -> float:
    """
    Executes an automated resolution sweep to identify the most stable 
    clustering plateau, evaluates robustness via subsampling, and checks 
    canonical biological markers.

    Parameters
    ----------
    training_filepath : str
        Path to the target AnnData file.
    key_for_saving_images : str
        Prefix for generated output plots.

    Returns
    -------
    float
        The automatically selected reference resolution.
    """
    adata_A = load_evidence(training_filepath)
    ref_res = None
    
    if adata_A.n_obs > 250:
        print("\n[AUDIT] Initiating clustering stability audit...")
        print("  -> [TEST 1] Resolution Sweep (0.1 to 2.0)...")
        
        resolutions = np.arange(res_start,res_end,res_step).tolist()
        results = []
        
        sc.pp.neighbors(
            adata_A, n_neighbors=n_neighbors, n_pcs=10, method='umap', 
            knn=True, metric='euclidean', random_state=42
        )
        
        for res in resolutions:
            key = f"leiden_res_{res}"
            sc.tl.leiden(adata_A, resolution=res, key_added=key, flavor='leidenalg')
            n_clusters = len(adata_A.obs[key].unique())
            results.append({'res': res, 'n_clusters': n_clusters, 'key': key})
            
        df_res = pd.DataFrame(results)
        aris = [0.0]
        
        for i in range(1, len(df_res)):
            prev_key = df_res.iloc[i-1]['key']
            curr_key = df_res.iloc[i]['key']
            score = adjusted_rand_score(adata_A.obs[prev_key], adata_A.obs[curr_key])
            aris.append(score)
            
        df_res['neighbor_ari'] = aris
        
        plt.figure(figsize=(25, 13))
        sns.lineplot(data=df_res, x='res', y='n_clusters', marker='o', label='Cluster Count')
        ax2 = plt.twinx()
        sns.lineplot(
            data=df_res, x='res', y='neighbor_ari', color='red', 
            marker='x', ax=ax2, label='Stability (ARI)'
        )
        plt.title("Parameter Stability: Cluster Count vs. ARI")
        plt.savefig(f"{plt_fig_dir}/{key_for_saving_images}_stability_resolution_sweep.png")
        plt.close()

        # Automated Plateau Extraction Logic
        print("  -> [INFO] Scanning arrays for topological plateaus...")
        carved_plateaus = []
        current_cluster_count = None
        current_block = {}

        for index, row in df_res.iterrows():
            nc = row['n_clusters']
            res = row['res']
            ari = row['neighbor_ari']

            if nc != current_cluster_count:
                if current_block:
                    carved_plateaus.append(current_block)
                current_cluster_count = nc
                current_block = {
                    'n_clusters': nc, 'count': 1, 'resolutions': [res], 'aris': [ari]
                }
            else:
                current_block['count'] += 1
                current_block['resolutions'].append(res)
                current_block['aris'].append(ari)

        if current_block:
            carved_plateaus.append(current_block)

        audited_plateaus = []
        print("  -> [INFO] Validated Plateaus:")
        for block in carved_plateaus:
            if block['count'] >= 3 and block['n_clusters'] > 1:
                mean_ari = np.mean(block['aris'])
                std_ari = np.std(block['aris'])
                
                block['mean_ari'] = mean_ari
                block['std_ari'] = std_ari
                block['res_bounds'] = (min(block['resolutions']), max(block['resolutions']))
                audited_plateaus.append(block)
                print(
                    f"     Clusters: {block['n_clusters']:<2} | Width: {block['count']:<2} | "
                    f"Bounds: {block['res_bounds'][0]:.2f}-{block['res_bounds'][1]:.2f} | "
                    f"Mean ARI: {mean_ari:.3f} | Variance: {std_ari:.4f}"
                )

        strict_candidates = [
            p for p in audited_plateaus 
            if p['mean_ari'] >= 0.85 and p['std_ari'] <= 0.25 and p['count'] >= 4
        ]

        if strict_candidates:
            strict_candidates.sort(key=lambda x: x['n_clusters'])
            winner = strict_candidates[0]
        elif audited_plateaus:
            print("  -> [WARNING] Strict stability not met. Relaxing variance constraint...")
            audited_plateaus.sort(key=lambda x: (x['n_clusters'], -x['mean_ari']))
            winner = audited_plateaus[0]
        else:
            print("  -> [ERROR] Matrix is unstable. Defaulting to maximum absolute ARI...")
            ref_res = round(df_res.loc[df_res['neighbor_ari'].idxmax(), 'res'], 3)
            winner = None

        if winner:
            target_index = int(winner['count'] * 0.6)
            ref_res = round(winner['resolutions'][target_index], 3)
            print(f"\n  -> [SUCCESS] Optimal Target: {winner['n_clusters']} Clusters (Mean ARI: {winner['mean_ari']:.3f}).")
            print(f"  -> Extracted coordinate: {ref_res:.3f}")

        # Subsampling Robustness
        print(f"  -> [TEST 2] Subsampling evaluation at Res {ref_res} (80% Retention)...")
        if f'leiden_res_{ref_res}' not in adata_A.obs:
            sc.tl.leiden(
                adata_A, resolution=ref_res, key_added=f'leiden_res_{ref_res}', flavor='leidenalg'
            )
            
        A_sub_idx, _ = train_test_split(
            adata_A.obs_names, test_size=0.2, random_state=42, shuffle=True
        )
        adata_A_test_sub = adata_A[A_sub_idx].copy()
        
        sc.pp.neighbors(adata_A_test_sub, n_neighbors=n_neighbors, n_pcs=10, use_rep='X_pca')
        sc.tl.leiden(
            adata_A_test_sub, resolution=ref_res, key_added='leiden_sub', flavor='leidenalg'
        )
        
        original_labels = adata_A.obs.loc[adata_A_test_sub.obs_names, f'leiden_res_{ref_res}']
        new_labels = adata_A_test_sub.obs['leiden_sub']
        robustness_score = adjusted_rand_score(original_labels, new_labels)
        
        print(f"  -> Structural Robustness Score (ARI): {robustness_score:.3f}")
        if robustness_score < 0.7:
            print("     [WARNING] Clusters exhibit structural instability under subsampling.")
        else:
            print("     [SUCCESS] Clusters are robust.")

        # Marker Gene Enrichment Audit
        print(f"  -> [TEST 3] Marker Gene Enrichment Audit (at Res {ref_res})...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            sc.tl.rank_genes_groups(adata_A, groupby=f'leiden_res_{ref_res}', method='wilcoxon')
            
        result_df = pd.DataFrame(adata_A.uns['rank_genes_groups']['names']).head(3)
        print("  -> Top 3 Marker Genes per Cluster:")
        print(result_df)
        
        marker_genes_dict = {
            'B-cell': ['MS4A1', 'CD79A'],
            'T-cell': ['CD3D', 'IL7R', 'CD8A'],
            'NK': ['GNLY', 'NKG7'],
            'Mono': ['CD14', 'FCGR3A'],
            'Dendritic': ['FCER1A', 'CST3'],
            'Platelet': ['PPBP']
        }
        
        valid_markers = {
            k: [g for g in v if g in adata_A.var_names] 
            for k, v in marker_genes_dict.items()
        }
        
        print("  -> Generating sanity dotplot...")
        sc.pl.dotplot(
            adata_A, valid_markers, groupby=f'leiden_res_{ref_res}', 
            standard_scale='var', show=False, save= f"_{key_for_saving_images}_stability_biological_sanity.png"
        )
        
        del adata_A_test_sub
        
    del adata_A
    gc.collect()
    
    return ref_res


def divide_and_save_dataset_based_on_macro_or_micro_clusters(
    file_path_used: str, 
    leiden_key_as_cluster_column_name: str
) -> dict:
    """
    Subsets the AnnData object by cluster labels and saves the independent 
    matrices to disk.

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
    print(f"[INFO] Dividing matrix based on topology: {leiden_key_as_cluster_column_name}")
    adata = load_evidence(file_path_used)
    grouped_barcodes = adata.obs.groupby(leiden_key_as_cluster_column_name).groups
    
    saved_filepaths_dictionary = {}
    path_object = Path(file_path_used)
    directory = path_object.parent
    base_name = path_object.stem
    
    for cluster_id, barcodes in grouped_barcodes.items():
        lineage_key = f"{leiden_key_as_cluster_column_name}_{cluster_id}"
        adata_subset = adata[barcodes].copy()
        
        if adata_subset.n_obs <= 250:
            print(f"[INFO] Cluster {cluster_id} (N={adata_subset.n_obs}) flagged as Terminal_State.")
            lineage_key = f"{lineage_key}_Terminal_State"
            
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
    Recalculates Pearson Residuals, highly variable genes, and PCA 
    for newly isolated subsets.

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
    print(f"[INFO] Recalculating variance components for {keys}...")
    adata = load_evidence(filepath)
    adata.X = adata.layers['counts'].copy()
    
    if adata.n_obs > 250:
        sc.experimental.pp.highly_variable_genes(
            adata, theta=100.0, n_top_genes=2500, flavor='pearson_residuals', subset=False
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            sc.experimental.pp.normalize_pearson_residuals(adata, theta=100.0)
        sc.pp.pca(
            adata, n_comps=100, zero_center=True, svd_solver='arpack', mask_var='highly_variable'
        )
        
        sc.pl.pca_variance_ratio(adata, n_pcs=100, save=f"_{keys}_.png", show=False)
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
    Executes standard Ingest projection of a target matrix onto a 
    pre-computed reference manifold using Pearson Residual expectations.

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
    print(f"[INFO] Projecting matrix onto reference manifold: {leiden_key_training}")
    adata_project = load_evidence(adata_project_file_path)
    adata_training = load_evidence(adata_training_file_path)
    
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
    
    # The Thermodynamic Floor: Prevent zero-division for completely silent genes
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
        ax.set_title(f'Projection Overlap Diagnostic: {leiden_key_training}', fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
        legend = ax.legend(loc='best', frameon=True, shadow=True)
        for handle in legend.legend_handles:
            handle.set_alpha(1.0)
            
        plt.tight_layout()
        plt.savefig(f'{plt_fig_dir}/{leiden_key_training}_Projection_Overlap.png')
        
    except KeyError:
        print("[WARNING] 'X_pca' missing. Visual projection diagnostic bypassed.")
        
    adata_project.write_h5ad(adata_project_file_path)
    del adata_project, adata_training
    gc.collect()

def calculate_dynamic_gravity(file_path:str) -> int:
    """
    Deterministically scales the KNN n_neighbors based on the 
    total number of cells in the manifold. 
    
    
    Parameters
    ----------
    file_path: str
      of the adata whose n_cells to check.
        
    Returns
    -------
    int
        The optimal n_neighbors (k) parameter.
    """
    adata_temp = load_evidence(file_path)
    n_cells = adata_temp.n_obs
    del adata_temp
    if n_cells < 1000:
        return 15
    elif n_cells < 5000:
        return 20
    elif n_cells < 15000:
        return 25
    else:
        return 30


def orchestrator_A(h5ad_path: str, save_folder_path: str, cell_cycle_genes_path: str) -> dict:
    """
    Master orchestrator for the Training Matrix processing sequence.

    Parameters
    ----------
    h5ad_path : str
        Path to the primary QC'd AnnData file.
    save_folder_path : str
        Directory to export processed matrices and logs.
    cell_cycle_genes_path : str
        Path to the cell cycle marker text file.

    Returns
    -------
    dict
        State dictionary containing all operational paths and topology keys.
    """
    print("\n===========================================================")
    print(" INITIATING ORCHESTRATOR A: TRAINING SEQUENCE")
    print("===========================================================")
    
    file_path_dict = random_split_data(h5ad_path, save_folder_path)
    training_file_path = file_path_dict['training_file']
    
    cell_cycle_check(training_file_path, cell_cycle_genes_path, 10, 10, 0.05, 'training')
    npr_hvg_pca_recal(training_file_path, 'training_file')
    
    macro_neighbors_numbers = calculate_dynamic_gravity(training_file_path)
    print(f"\n[PHYSICS] Setting Macro neighbors numbers (k) to {macro_neighbors_numbers}")
    macro_res = stability_audit(training_file_path,'macro',0.01,0.21,0.003,macro_neighbors_numbers)
    macro_leiden_key, macro_neighbors_key = knn_umap_leiden(
        training_file_path, n_neighbors=macro_neighbors_numbers, n_pcs=10, leiden_res=macro_res, key_name='macro'
    )
    
    micro_filepaths_dict = divide_and_save_dataset_based_on_macro_or_micro_clusters(
        training_file_path, macro_leiden_key
    )
    
    micro_leiden_key_dict = {}
    micro_neighbors_key_dict = {}
    
    for keys, filepath in micro_filepaths_dict.items():
        if 'Terminal_State' in keys:
            continue
            
        npr_hvg_pca_recal(filepath, keys)
        micro_n_neighbors = calculate_dynamic_gravity(file_path=filepath)
        ref_res = stability_audit(filepath, keys,0.1,2.1,0.03,micro_n_neighbors)
        
        if ref_res is not None:
            cell_cycle_check(filepath, cell_cycle_genes_path, n_neighbors=micro_n_neighbors,
                              n_pcs=10, leiden_res=ref_res, file_save_key=keys)
            m_leiden, m_neighbors = knn_umap_leiden(
                filepath, n_neighbors=micro_n_neighbors, n_pcs=15, leiden_res=ref_res, key_name=f'{keys}_micro'
            )
            if m_leiden is not None:
                micro_leiden_key_dict[keys] = m_leiden
                micro_neighbors_key_dict[keys] = m_neighbors
        else:
            print(f"[WARNING] Sub-clustering failed to stabilize for {keys}.")

    return {
        'main_pca_artifact_path': h5ad_path,
        'writing_files_folder_path': save_folder_path,
        'file_path_dictionary_from_the_split_step': file_path_dict,
        'training_macro_leiden_key': macro_leiden_key,
        'training_macro_neighbors_key': macro_neighbors_key,
        'training_micro_file_path_dictionary': micro_filepaths_dict,
        'training_micro_leiden_key_dictionary': micro_leiden_key_dict,
        'training_micro_neighbors_key_dictionary': micro_neighbors_key_dict
    }


def orchestrator_B(dict_A: dict) -> dict:
    """
    Master orchestrator for the Projection Sequence.

    Parameters
    ----------
    dict_A : dict
        The state dictionary generated by Orchestrator A.

    Returns
    -------
    dict
        State dictionary containing validation projection paths and keys.
    """
    print("\n===========================================================")
    print(" INITIATING ORCHESTRATOR B: PROJECTION SEQUENCE")
    print("===========================================================")
    
    macro_project_path = dict_A['file_path_dictionary_from_the_split_step']['projectable_file']
    macro_training_path = dict_A['file_path_dictionary_from_the_split_step']['training_file']
    macro_neighbors_key = dict_A['training_macro_neighbors_key']
    macro_leiden_key = dict_A['training_macro_leiden_key']
    
    cast_projectable_data_on_training_data(
        macro_project_path, macro_training_path, macro_neighbors_key, macro_leiden_key
    )
    
    projected_micro_file_dict = divide_and_save_dataset_based_on_macro_or_micro_clusters(
        macro_project_path, macro_leiden_key
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
                print(f"[INFO] Bypassing micro-cast for Terminal State: {root_key}")
                continue
                
            p_leiden = dict_A['training_micro_leiden_key_dictionary'].get(target_key_training)
            p_neighbors = dict_A['training_micro_neighbors_key_dictionary'].get(target_key_training)
            
            if p_leiden and p_neighbors:
                cast_projectable_data_on_training_data(
                    projected_micro_filepath, training_micro_filepath, p_neighbors, p_leiden
                )
                project_micro_leiden_dict[keys_projected] = p_leiden
                project_micro_neighbors_dict[keys_projected] = p_neighbors
        else:
            print(f"[WARNING] Projected cluster {keys_projected} not found in training set. Discarding.")

    return {
        'macro_adata_project_file_path': macro_project_path,
        'macro_neighbors_key_training': macro_neighbors_key,
        'macro_leiden_key_training': macro_leiden_key,
        'projected_micro_file_path_dictionary': projected_micro_file_dict,
        'projected_micro_leiden_key_dictionary': project_micro_leiden_dict,
        'projected_micro_neighbors_key_dictionary': project_micro_neighbors_dict
    }



if __name__ == '__main__':
    # Relative paths
    h5ad_path = './data/objects/pbmc3k_qc.h5ad'
    save_folder_path = './data/objects'
    os.makedirs(os.path.dirname(save_folder_path), exist_ok=True)

    cell_cycle_genes_path = './data/regev_lab_cell_cycle_genes.txt'
    
    dict_A = orchestrator_A(h5ad_path, save_folder_path, cell_cycle_genes_path)
    dict_B = orchestrator_B(dict_A)
    
    with open(f'{save_folder_path}/Dictionary_of_returns_from_orch_A.json', 'w') as json_file:
        json.dump(dict_A, json_file, indent=4)
        
    with open(f'{save_folder_path}/Dictionary_of_returns_from_orch_B.json', 'w') as json_file:
        json.dump(dict_B, json_file, indent=4)
        
    print("\n[SUCCESS] PHASE II COMPLETE. READY FOR PHASE III.")


