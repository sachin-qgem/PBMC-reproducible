import gc
import json
import os
import os.path as op
import warnings
from pathlib import Path

import anndata as ad
import celltypist as ct
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns

# Global environment settings

ct.models.models_path = './data/celltypist_models'
os.makedirs(ct.models.models_path, exist_ok=True)
sc.settings.verbosity = 0
plt_fig_dir = Path('./results/figures/p05_top_markers')
plt_fig_dir.mkdir(parents=True, exist_ok=True)


def load_evidence(h5ad_path: str) -> ad.AnnData:
    """
    Loads an AnnData artifact from disk for marker extraction.

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

def calculate_elastic_threshold(group, min_survivors=3, base_quantile=0.9375):
        """
        Dynamically adjusts the distillation threshold. 
        Guarantees a Minimum Viable Payload (MVP) of markers for fragile micro-states 
        while preserving the strict Q93 filter for massive continents.
        """
        # Calculate the strict theoretical threshold
        q_val = group.quantile(base_quantile)
        
        # Calculate how many physical genes would survive this cut
        surviving_mass = (group >= q_val).sum()
        
        if surviving_mass < min_survivors:
            # The Starvation Limit is breached. Engage fallback.
            if len(group) >= min_survivors:
                # Lower the threshold just enough to capture exactly the top 'min_survivors'
                q_val = group.nlargest(min_survivors).iloc[-1]
            else:
                # The cluster has fewer than 3 total valid genes.
                q_val = group.min()
                
        return q_val

    

def rank_gene_groups_wilcoxon(
    adata_path: str, 
    leiden_key: str, 
    annotation_manual_dict: dict,
    ontology_cl_id_manual_dict : dict,
    quantile: float = 0.9375
) -> tuple:
    """
    Executes a Wilcoxon Rank-Sum test to extract defining thermodynamic 
    markers for each topological state. Updates the master annotation ledger.

    Parameters
    ----------
    adata_path : str
        Path to the target AnnData file.
    leiden_key : str
        The observation column defining the cluster topologies.
    annotation_manual_dict : dict
        The active tracking dictionary for manual annotation states.
    ontology_cl_id_manual_dict : dict
        The active tracking dictionary for manual cl_ID states.
    quantile : float, default 0.9375
        The statistical threshold for isolating highly significant markers.

    Returns
    -------
    dict
        The updated annotation ledger containing newly extracted null states.
    """
    print(f"[AUDIT] Executing Wilcoxon thermodynamic extraction for {leiden_key}...")
    adata = load_evidence(adata_path)


    #clusters having less than 10 cells are ingnored from test stats
    cluster_populations = adata.obs[leiden_key].value_counts()
    viable_states = cluster_populations[cluster_populations >= 10].index.astype(str).tolist()
    ghost_states = cluster_populations[cluster_populations < 10].index.astype(str).tolist()
    if ghost_states:
        print(f"[WARNING]⚠️ Topology Anomaly in {leiden_key}: Clusters {ghost_states} lack minimum mass (N<10).")
        print("[SYSTEM] Exiling microscopic ghosts from Wilcoxon variance calculation...")
    if len(viable_states) < 2:
        print(f" [SYSTEM] {leiden_key} possesses fewer than 2 viable states ({len(viable_states)}). Thermodynamic comparison impossible.")
        print(" [SYSTEM] Bypassing Wilcoxon matrix geometry to prevent spatial collapse...")
        
        
        if leiden_key in adata.obs:
            clusters = sorted(
                adata.obs[leiden_key].dropna().unique().tolist(), 
                key=lambda x: int(x) if str(x).isdigit() else x
            )
            annotation_manual_dict[leiden_key] = {str(c): None for c in clusters}
            ontology_cl_id_manual_dict[leiden_key] = {str(c): None for c in clusters}
            print(f"[SUCCESS] Appended {len(clusters)} raw states to ledger via bypass.")
            
        
        adata.write_h5ad(adata_path)
        del adata
        gc.collect()
        return annotation_manual_dict, ontology_cl_id_manual_dict
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        sc.tl.rank_genes_groups(
            adata, groupby=leiden_key, method='wilcoxon', tie_correct=True, 
            groups=viable_states,layer='log1p_norm', pts=True
        )
        sc.tl.filter_rank_genes_groups(adata, use_raw=False)
        
    df = sc.get.rank_genes_groups_df(adata, group=None, key='rank_genes_groups_filtered')
    
    # Statistical isolation
    pvals_adj_logfc_mask = (df['pvals_adj'] < 0.05) & (df['logfoldchanges'] > 1.0) & (df['logfoldchanges'] < 10.0)
    df_new = df[pvals_adj_logfc_mask].dropna(subset=['names']).copy()
    
    # Avoid log(0) warnings by adding a tiny epsilon if needed, though scanpy usually handles it.
    df_new['nlog10pval_adj'] = -np.log10(df_new['pvals_adj'] + 1e-300)
    
    # Apply the elastic threshold to the tensor
    df_new['local_Q93'] = df_new.groupby('group')['nlog10pval_adj'].transform(
        lambda x: calculate_elastic_threshold(x, min_survivors=3, base_quantile=quantile)
    )
    df_new['violin_delta'] = df_new['pct_nz_group'] - df_new['pct_nz_reference']
    
    mask_1 = (df_new['nlog10pval_adj'] >= df_new['local_Q93'])
    df_mask_1 = df_new[mask_1].copy()
    
    df_mask_1_sorted = df_mask_1.sort_values(
        by=['group', 'violin_delta'], ascending=[True, False]
    )
    df_final = df_mask_1_sorted.groupby('group').head(3).copy()
    
    grouped_top_genes = {}
    for cluster_id in df_final['group'].unique():
        genes = df_final[df_final['group'] == cluster_id]['names'].tolist()
        grouped_top_genes[str(cluster_id)] = genes

    if not grouped_top_genes:
        print(f"[WARNING] Thermodynamic Void in {leiden_key}: Zero genes survived the absolute floor (LogFC > 1.0).")
        print("[SYSTEM] Bypassing Matplotlib rendering to prevent X-axis spatial collapse...")
    else:
        print("[INFO] Rendering dendrogram, dotplot, and matrixplot evidence...")
        
        # Mathematically strip the ghost categories from the pandas index to prevent the rendering engine from hunting for missing variables.
        adata_plot = adata[adata.obs[leiden_key].isin(viable_states)].copy()
        adata_plot.obs[leiden_key] = adata_plot.obs[leiden_key].cat.remove_unused_categories()
        
        sc.tl.dendrogram(adata_plot, groupby=leiden_key)
        
        safe_fig_name = f"_{leiden_key}_top_genes.svg"
        
        sc.pl.dotplot(
            adata_plot, var_names=grouped_top_genes, groupby=leiden_key, standard_scale='var',
            dendrogram=True, cmap='Reds', use_raw=False, show=False,
            save=safe_fig_name, layer='log1p_norm', title=f"dotplot_{leiden_key}_top_genes"
        )
        
        sc.pl.matrixplot(
            adata_plot, var_names=grouped_top_genes, groupby=leiden_key, dendrogram=True,
            standard_scale='var', save=safe_fig_name, show=False, layer='log1p_norm', title=f"matrixplot_{leiden_key}_top_genes"
        )
        
        
        plt.close('all')
        del adata_plot

    # State storage
    adata.uns['final_top_genes_per_cluster'] = df_final
    if 'rank_genes_groups_filtered' in adata.uns:
        del adata.uns['rank_genes_groups_filtered']
        
    # Append to the Annotation Ledger
    if leiden_key in adata.obs:
        clusters = sorted(
            adata.obs[leiden_key].dropna().unique().tolist(), 
            key=lambda x: int(x) if str(x).isdigit() else x
        )
        annotation_manual_dict[leiden_key] = {str(c): None for c in clusters}
        ontology_cl_id_manual_dict[leiden_key] = {str(c): None for c in clusters}

        print(f"[SUCCESS] Appended {len(clusters)} null states to ledger.")
    else:
        print(f"[WARNING] Key '{leiden_key}' not found in matrix. Ledger bypassed.")
        
    adata.write_h5ad(adata_path)
    del adata, df, df_new, df_mask_1, df_mask_1_sorted, df_final
    gc.collect()
    
    return annotation_manual_dict,ontology_cl_id_manual_dict


def execute_absence_cross_validation(
    target_macro_id: str, 
    current_micro_path: str, 
    current_leiden_key: str, 
    macro_path: str, 
    micro_paths_dict: dict
) -> None:
    """
    Forensic algorithmic cross-validation. Extracts top markers from all 
    foreign matrices and projects them onto the target matrix to prove 
    lineage isolation (Epigenetic Silencing).

    Parameters
    ----------
    target_macro_id : str
        The numeric string identifier of the target lineage.
    current_micro_path : str
        Path to the target AnnData file.
    current_leiden_key : str
        The observation column defining the cluster topologies.
    macro_path : str
        Path to the global Macro AnnData file.
    micro_paths_dict : dict
        Dictionary of all active Micro matrix paths.

    Returns
    -------
    None
    """
    print(f"\n[AUDIT] Initiating Cross-Validation for Lineage: {target_macro_id}")
    contamination_dict = {}
    
    # PHASE 1: The Macro Sweep
    print("  -> Sweeping Global Macro Matrix for foreign lineages...")
    adata_macro = load_evidence(macro_path)
    
    if 'final_top_genes_per_cluster' in adata_macro.uns:
        df_macro = adata_macro.uns['final_top_genes_per_cluster']
        foreign_macro_groups = [
            g for g in df_macro['group'].unique() if str(g) != str(target_macro_id)
        ]
        
        for f_group in foreign_macro_groups:
            genes = df_macro[df_macro['group'] == f_group]['names'].tolist()
            contamination_dict[f'Macro_Source_{f_group}'] = genes
            
    del adata_macro
    gc.collect()
    
    # PHASE 2: The Micro Sweep
    print("  -> Sweeping Foreign Micro Matrices for high-resolution contaminants...")
    for dict_key, file_path in micro_paths_dict.items():
        source_id = dict_key.split('_')[-1]
        
        if str(source_id) == str(target_macro_id):
            continue
            
        if op.exists(file_path):
            adata_foreign_micro = load_evidence(file_path)
            if 'final_top_genes_per_cluster' in adata_foreign_micro.uns:
                df_micro = adata_foreign_micro.uns['final_top_genes_per_cluster']
                for m_group in df_micro['group'].unique():
                    genes = df_micro[df_micro['group'] == m_group]['names'].tolist()
                    contamination_dict[f'Micro_Source_{source_id}_cluster_{m_group}'] = list(set(genes))
                    
            del adata_foreign_micro
            gc.collect()
            
    # PHASE 3: The Projection
    print(f"  -> Projection Dictionary Assembled. Total Foreign Vectors: {len(contamination_dict)}")
    adata_target = load_evidence(current_micro_path)
    raw_clusters = adata_target.obs[current_leiden_key].dropna().unique()
    sorted_clusters = sorted(raw_clusters, key=lambda x: int(x) if str(x).isdigit() else x)
    adata_target.obs[current_leiden_key] = pd.Categorical(
        adata_target.obs[current_leiden_key].astype(str), 
        categories=[str(c) for c in sorted_clusters], 
        ordered=True
    )
    available_genes = set(adata_target.var_names)
    
    safe_contamination_dict = {}
    for source, genes in contamination_dict.items():
        valid_genes = [g for g in genes if g in available_genes]
        if valid_genes:
            safe_contamination_dict[source] = valid_genes
            
    safe_fig_name = f"_absence_audit_macro_{target_macro_id}.svg"
    
    sc.pl.matrixplot(
        adata_target, var_names=safe_contamination_dict, groupby=current_leiden_key,
        standard_scale=None, use_raw=False, show=False, save=safe_fig_name,
        colorbar_title='Absolute Log-Mean Expression', layer='log1p_norm', title=f"matrix_plot_absence_audit_macro_{target_macro_id}"
    )
    
    plt.close('all')
    print(f"[SUCCESS] Absence Topology rendered and saved.")
    
    del adata_target
    gc.collect()


def auto_ref_mapping(adata_path: str, model_type: ct.models.Model, leiden_key: str) -> None:
    """
    Executes automated reference-based annotation using CellTypist.

    Parameters
    ----------
    adata_path : str
        Path to the target AnnData file.
    model_type : ct.models.Model
        The loaded CellTypist model.
    leiden_key : str
        The observation column to utilize for over-clustering majority voting.

    Returns
    -------
    None
    """
    print(f"[INFO] Executing CellTypist mapping against {leiden_key}...")
    adata = load_evidence(adata_path)
    adata_hold = adata.X
    adata.X = adata.layers['log1p_norm'].copy()
    
    predictions = ct.annotate(
        adata, model=model_type, over_clustering=leiden_key, majority_voting=True
    )
    
    # Safely transfer predictions back to the matrix
    adata = predictions.to_adata()
    adata.X = adata_hold
    
    adata.write_h5ad(adata_path)
    del adata, adata_hold, predictions
    gc.collect()


def wide_span_plots(
    adata_path: str, 
    groupby_key: str, 
    curated_marker_wide_span_list_path: str
) -> None:
    """
    Validates the matrix against a pre-curated JSON dictionary of canonical markers.

    Parameters
    ----------
    adata_path : str
        Path to the target AnnData file.
    groupby_key : str
        The observation column dictating the clusters.
    curated_marker_wide_span_list_path : str
        Path to the JSON file containing canonical markers.

    Returns
    -------
    None
    """
    print(f"[INFO] Generating wide-span canonical validation plots for {groupby_key}...")
    adata = load_evidence(adata_path)

    cluster_populations = adata.obs[groupby_key].value_counts()
    viable_states = cluster_populations[cluster_populations >= 10].index.astype(str).tolist()
    if len(viable_states) == 0:
        print(f" [WARNING] No viable states (N>=10) in {groupby_key}. Aborting wide-span plots.")
        del adata
        gc.collect()
        return None
    adata_plot = adata[adata.obs[groupby_key].isin(viable_states)].copy()
    adata_plot.obs[groupby_key] = adata_plot.obs[groupby_key].cat.remove_unused_categories()


    available_genes = set(adata_plot.var_names)
    
    with open(curated_marker_wide_span_list_path, 'r') as file:
        curated_markers = json.load(file)
        
    valid_genes = [gene for gene in curated_markers if gene in available_genes]
    safe_fig_name = f"_curated_genes_audit_widespan_{groupby_key}.svg"
    
    if len(valid_genes) > 0:
        sc.pl.matrixplot(
            adata_plot, var_names=valid_genes, groupby=groupby_key, standard_scale=None,
            use_raw=False, show=False, save=safe_fig_name,
            colorbar_title='Absolute Log-Mean Expression', layer='log1p_norm',title = f"matrixplot_curated_genes_audit_widespan_{groupby_key}"
        )
        sc.pl.dotplot(
            adata_plot, var_names=valid_genes, groupby=groupby_key, standard_scale=None,
            use_raw=False, show=False, save=safe_fig_name,
            colorbar_title='Absolute Log-Mean Expression', layer='log1p_norm', title= f"dotplot_curated_genes_audit_widespan_{groupby_key}"
        )
    else:
        print("[ERROR] Curated valid_genes list is empty. Visual rendering aborted.")
        
    del adata,adata_plot
    gc.collect()


def orc_project(
    dict_b_path: str, 
    curated_marker_list_path: str, 
    annotation_save_path: str,
    ontology_cl_id_path: str
) -> None:
    """
    Master orchestrator for Phase III. Ingests the Orchestrator B map, 
    drives marker extraction, executes cross-validation audits, and 
    constructs the final nested annotation ledger.

    Parameters
    ----------
    dict_b_path : str
        Path to the state dictionary output by Phase II.
    curated_marker_list_path : str
        Path to the JSON of canonical lineage markers.
    annotation_save_path : str
        Target output path for the generated `annotation_manual.json` ledger.
    ontology_cl_id_path : str
        Target output path for the generated `ontology_cl_id_manual.json` ledger
    Returns
    -------
    None
    """
    print("\n===========================================================")
    print(" INITIATING ORCHESTRATOR C: MARKER EXTRACTION & AUDIT")
    print("===========================================================")
    
    macro_model = ct.models.Model.load(model='Immune_All_High.pkl')
    micro_model = ct.models.Model.load(model='Immune_All_Low.pkl')
    
    if not op.exists(dict_b_path):
        print(f"[ERROR] Dictionary missing at {dict_b_path}")
        return None
        
    with open(dict_b_path, 'r') as json_file:
        dict_b = json.load(json_file)
        
    annotation_manual_dict = {}
    ontology_cl_id_manual_dict = {}
    macro_path_key, macro_leiden_key, micro_paths_key, micro_leiden_key = None, None, None, None
    
    # Robust dictionary key extraction
    for k in dict_b.keys():
        if "macro" in k and "file_path" in k and "dictionary" not in k:
            macro_path_key = k
        elif "macro" in k and "leiden_key" in k and "dictionary" not in k:
            macro_leiden_key = k
        elif "micro" in k and "file_path" in k and "dictionary" in k:
            micro_paths_key = k
        elif "micro" in k and "leiden_key" in k and "dictionary" in k:
            micro_leiden_key = k
            
    if not all([macro_path_key, macro_leiden_key, micro_paths_key, micro_leiden_key]):
        print("[ERROR] Matrix collapse. Could not autonomously identify core keys.")
        return None
        
    macro_path = dict_b.get(macro_path_key)
    macro_leiden = dict_b.get(macro_leiden_key)
    
    if macro_path and macro_leiden:
        if op.exists(macro_path):
            print(f"\n[MACRO] Locking anchor '{macro_leiden}' for path: {macro_path}")
            annotation_manual_dict,ontology_cl_id_manual_dict = rank_gene_groups_wilcoxon(
                macro_path, macro_leiden, annotation_manual_dict,ontology_cl_id_manual_dict
            )
            print("[MACRO] Thermodynamic extraction complete.")
            auto_ref_mapping(macro_path, macro_model, macro_leiden)
        else:
            print(f"[ERROR] File missing at {macro_path}")
            
    micro_paths_dict = dict_b.get(micro_paths_key, {})
    micro_leiden_dict = dict_b.get(micro_leiden_key, {})
    print("\n[ORCHESTRATOR] Initiating Phase 1: Global State Generation...")
    for leiden_dict_key, file_path in micro_paths_dict.items():
        print(f"\n[MICRO] Scanning topology: {leiden_dict_key}")
        
        if not op.exists(file_path):
            print(f"[ERROR] Matrix missing at {file_path}. Bypassing.")
            continue
            
        active_leiden_col = micro_leiden_dict.get(leiden_dict_key)
        
        if active_leiden_col is None:
            clean_key = leiden_dict_key.replace('_Terminal_State', '')
            parts = clean_key.split('_')
            parent_dict_key = '_'.join(parts[:-1])
            
            print(f"  -> Bypassing extraction. Inheriting Parent key: '{parent_dict_key}'")
            auto_ref_mapping(file_path, micro_model, parent_dict_key)
            continue
            
        print(f"  -> Anchor locked: '{active_leiden_col}'. Executing Wilcoxon Engine...")
        annotation_manual_dict,ontology_cl_id_manual_dict = rank_gene_groups_wilcoxon(
            file_path, active_leiden_col, annotation_manual_dict,ontology_cl_id_manual_dict
        )
        auto_ref_mapping(file_path, micro_model, active_leiden_col)
        
    print("\n[ORCHESTRATOR] Initiating Phase 2: Global Forensic Cross-Validation...")
    for leiden_dict_key, file_path in micro_paths_dict.items():
        print(f"\n[MICRO AUDIT] Executing visual proofs for: {leiden_dict_key}")
        
        if not op.exists(file_path):
            continue  # Already logged in Phase 1
            
        active_leiden_col = micro_leiden_dict.get(leiden_dict_key)
        
        # If it was a terminal state with no active leiden col, we still want to 
        # run the wide_span_plots on the inherited parent key, but absence audit requires 
        # a target macro ID.
        if active_leiden_col is None:
             clean_key = leiden_dict_key.replace('_Terminal_State', '')
             parts = clean_key.split('_')
             parent_dict_key = '_'.join(parts[:-1])
             target_id = str(leiden_dict_key).split('_')[-2] # Fetch the number before Terminal_State
             
             execute_absence_cross_validation(
                 target_macro_id=target_id, 
                 current_micro_path=file_path,
                 current_leiden_key=parent_dict_key, 
                 macro_path=macro_path,
                 micro_paths_dict=micro_paths_dict
             )
             wide_span_plots(file_path, parent_dict_key, curated_marker_list_path)
             continue

        # Standard Execution for active clusters
        target_id = str(leiden_dict_key).split('_')[-1]
        
        execute_absence_cross_validation(
            target_macro_id=target_id, 
            current_micro_path=file_path,
            current_leiden_key=active_leiden_col, 
            macro_path=macro_path,
            micro_paths_dict=micro_paths_dict
        )
        wide_span_plots(file_path, active_leiden_col, curated_marker_list_path)
        
    # Terminal Seal
    with open(annotation_save_path, 'w') as f:
        json.dump(annotation_manual_dict, f, indent=4)
    with open(ontology_cl_id_path, 'w') as f:
        json.dump(ontology_cl_id_manual_dict, f, indent=4)
        
    print(f"\n[SEALED] Master Annotation Ledger written to: {annotation_save_path}")
    print(f"\n[SEALED] Master CL_ID Ledger written to: {ontology_cl_id_path}")



def main():
    # Initialize CellTypist environment
    ad.settings.allow_write_nullable_strings = True
    sc.settings.figdir = "./results/figures/p05_top_markers"
    os.makedirs(sc.settings.figdir, exist_ok=True)
    
    ct.models.download_models(force_update=True, model=['Immune_All_Low.pkl', 'Immune_All_High.pkl'])
    
    # Absolute paths
    dict_file_path = './data/objects/Dictionary_of_returns_from_orch_B.json'
    curated_marker_path = './data/Teichlab_curated_markers.json'
    annotation_save_path = './data/objects/annotation_manual_empty.json'
    ontology_cl_id_path = './data/objects/ontology_cl_id_manual_empty.json'
    orc_project(
        dict_b_path=dict_file_path, 
        curated_marker_list_path=curated_marker_path, 
        annotation_save_path=annotation_save_path,
        ontology_cl_id_path=ontology_cl_id_path
    )
    
    print("\n[SUCCESS] PHASE III COMPLETE. AWAITING HUMAN ANNOTATION.")

if __name__ == '__main__':
    main()