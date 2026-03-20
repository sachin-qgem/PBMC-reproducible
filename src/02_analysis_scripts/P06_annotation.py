import gc
import json
import os
import os.path as op
from pathlib import Path

import anndata as ad
import pandas as pd
import scanpy as sc
from sklearn.metrics import adjusted_rand_score

# Global environment settings
ad.settings.allow_write_nullable_strings = True
sc.settings.figdir = "./results/figures/p06_annotation"
os.makedirs(sc.settings.figdir, exist_ok=True)
sc.settings.verbosity = 0
plt_fig_dir = Path('./results/figures/p06_annotation')
plt_fig_dir.mkdir(parents=True, exist_ok=True)


def load_evidence(h5ad_path: str) -> ad.AnnData:
    """
    Loads an AnnData artifact from disk for final annotation mapping.

    Parameters
    ----------
    h5ad_path : str
        The absolute or relative path to the .h5ad file.

    Returns
    -------
    ad.AnnData
        The loaded expression matrix.
    """
    print(f"[INFO] Loading target matrix from {h5ad_path}")
    adata = sc.read_h5ad(h5ad_path)
    print(f"[INFO] Loaded dimensions: {adata.n_obs} cells x {adata.n_vars} genes")
    
    return adata


def orc_annotation(
    dict_path: str, 
    annotation_manual_path: str, 
    ontology_cl_id_dict_path: str
) -> None:
    """
    Injects human-verified biological annotations and standard Cell Ontology (CL) 
    IDs into the localized Macro and Micro matrices. Calculates alignment 
    scores against automated reference models.

    Parameters
    ----------
    dict_path : str
        Path to the state dictionary (Orchestrator A or B).
    annotation_manual_path : str
        Path to the human-populated JSON ledger of biological identities.
    ontology_cl_id_dict_path : str
        Path to the JSON ledger mapping biological identities to CL IDs.

    Returns
    -------
    None
    """
    print("\n===========================================================")
    print(" INITIATING ANNOTATION INJECTION ENGINE")
    print("===========================================================")
    
    if not op.exists(dict_path):
        print(f"[ERROR] State dictionary missing at {dict_path}")
        return None
        
    with open(dict_path, 'r') as json_file:
        state_dict = json.load(json_file)
        
    if not op.exists(annotation_manual_path):
        print(f"[ERROR] Manual annotation ledger missing at {annotation_manual_path}")
        return None
        
    with open(annotation_manual_path, 'r') as json_file:
        annotation_manual = json.load(json_file)
        
    if not op.exists(ontology_cl_id_dict_path):
        print(f"[ERROR] Ontology dictionary missing at {ontology_cl_id_dict_path}")
        return None
        
    with open(ontology_cl_id_dict_path, 'r') as json_file:
        ontology_cl_id_dict_manual = json.load(json_file)

    macro_path_key, macro_leiden_key, micro_paths_key, micro_leiden_key = None, None, None, None
    
    for k in state_dict.keys():
        if 'split' in k or ("macro" in k and "file_path" in k and "dictionary" not in k):
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

    # MACRO INJECTION
    macro_payload = state_dict.get(macro_path_key)
    macro_path = None
    
    if isinstance(macro_payload, dict):
        for k in macro_payload.keys():
            if 'training' in k:
                macro_path = macro_payload.get(k)
                break
    else:
        macro_path = macro_payload
        
    macro_leiden = state_dict.get(macro_leiden_key)
    
    if macro_path and macro_leiden:
        if op.exists(macro_path):
            print(f"[MACRO] Mapping annotations for manifold: {macro_leiden}")
            adata = load_evidence(macro_path)
            
            # Vectorized assignment
            adata.obs['manual_labels'] = adata.obs[macro_leiden].map(annotation_manual[macro_leiden])
            adata.obs['human_CL_ID'] = adata.obs['manual_labels'].map(ontology_cl_id_dict_manual)
            
            if 'majority_voting' in adata.obs:
                adata.obs['oracle_CL_ID'] = adata.obs['majority_voting'].map(ontology_cl_id_dict_manual)
                
                # Drop NaNs to allow strict ARI comparison
                valid_mask = adata.obs['human_CL_ID'].notna() & adata.obs['oracle_CL_ID'].notna()
                if valid_mask.sum() > 0:
                    ari_score = adjusted_rand_score(
                        adata.obs.loc[valid_mask, 'human_CL_ID'].astype(str),
                        adata.obs.loc[valid_mask, 'oracle_CL_ID'].astype(str)
                    )
                    adata.uns['Oracle_ARI_Score'] = float(ari_score)
                    print(f"  -> Oracle Alignment Score (ARI): {ari_score:.3f}")   
            
            adata.write_h5ad(macro_path)
            del adata
            gc.collect()
        else:
            print(f"[ERROR] Macro file missing at {macro_path}")

    # MICRO INJECTION
    micro_paths_dict = state_dict.get(micro_paths_key, {})
    micro_leiden_dict = state_dict.get(micro_leiden_key, {})
    
    for leiden_dict_key, file_path in micro_paths_dict.items():
        print(f"\n[MICRO] Scanning topology: {leiden_dict_key}")
        
        if not op.exists(file_path):
            print(f"  -> [WARNING] Matrix missing at {file_path}. Bypassing.")
            continue
            
        active_leiden_col = micro_leiden_dict.get(leiden_dict_key)
        adata = load_evidence(file_path)
        
        if active_leiden_col is None:
            # Handle Terminal States via Parent Inheritance
            clean_key = leiden_dict_key.replace('_Terminal_State', '')
            parts = clean_key.split('_')
            cluster_id = parts[-1]
            parent_dict_key = '_'.join(parts[:-1])
            
            inherited_label = annotation_manual.get(parent_dict_key, {}).get(cluster_id)
            print(f"  -> Terminal State detected. Inheriting Parent Label: '{inherited_label}'")
            
            adata.obs['manual_labels'] = inherited_label
            adata.obs['human_CL_ID'] = adata.obs['manual_labels'].map(ontology_cl_id_dict_manual)
            
        else:
            # Handle Standard Active Micro States
            print(f"  -> Active State detected. Mapping via: '{active_leiden_col}'")
            adata.obs['manual_labels'] = adata.obs[active_leiden_col].map(
                annotation_manual[active_leiden_col]
            )
            adata.obs['human_CL_ID'] = adata.obs['manual_labels'].map(ontology_cl_id_dict_manual)

        # Oracle Alignment Check
        if 'majority_voting' in adata.obs:
            adata.obs['oracle_CL_ID'] = adata.obs['majority_voting'].map(ontology_cl_id_dict_manual)
            
            valid_mask = adata.obs['human_CL_ID'].notna() & adata.obs['oracle_CL_ID'].notna()
            if valid_mask.sum() > 0:
                ari_score = adjusted_rand_score(
                    adata.obs.loc[valid_mask, 'human_CL_ID'].astype(str),
                    adata.obs.loc[valid_mask, 'oracle_CL_ID'].astype(str)
                )
                adata.uns['Oracle_ARI_Score'] = float(ari_score)
                print(f"  -> Oracle Alignment Score (ARI): {ari_score:.3f}")

        adata.write_h5ad(file_path)
        del adata
        gc.collect()


def label_mapping_data_frame_all(dict_path: str, master_df_csv_path: str) -> None:
    """
    Extracts cell barcodes and injected labels from all isolated matrices 
    and concatenates them into a master CSV ledger. Resolves barcode collisions 
    by prioritizing the most recent execution state.

    Parameters
    ----------
    dict_path : str
        Path to the state dictionary containing matrix paths.
    master_df_csv_path : str
        Target output path for the central CSV ledger.

    Returns
    -------
    None
    """
    print("\n[AUDIT] Aggregating localized labels into Master Ledger...")
    
    if not op.exists(dict_path):
        print(f"[ERROR] State dictionary missing at {dict_path}")
        return None
        
    with open(dict_path, 'r') as json_file:
        state_dict = json.load(json_file)
        
    micro_paths_key = None
    for k in state_dict.keys():
        if "micro" in k and "file_path" in k and "dictionary" in k:
            micro_paths_key = k
            
    if not micro_paths_key:
        print("[ERROR] Matrix collapse. Could not identify micro paths.")
        return None
        
    micro_paths_dict = state_dict.get(micro_paths_key, {})
    new_ledgers = []
    
    for key_cluster_id, file_path in micro_paths_dict.items():
        if not op.exists(file_path):
            print(f"  -> [WARNING] Missing matrix at {file_path}. Skipping.")
            continue
            
        adata_micro = load_evidence(file_path)
        
        if 'manual_labels' in adata_micro.obs and 'human_CL_ID' in adata_micro.obs:
            clean_df = adata_micro.obs[['manual_labels', 'human_CL_ID']].copy()
            new_ledgers.append(clean_df)
        else:
            print(f"  -> [ERROR] Labels missing in {key_cluster_id}. Cannot extract.")
            
        del adata_micro
        gc.collect()
        
    if not new_ledgers:
        print("[ERROR] No valid ledgers extracted. Terminating aggregation.")
        return None
        
    current_run_ledger = pd.concat(new_ledgers)
    
    if op.exists(master_df_csv_path):
        print("  -> Existing ledger detected. Loading into RAM for integration...")
        master_df = pd.read_csv(master_df_csv_path, index_col=0)
        combined_ledger = pd.concat([master_df, current_run_ledger])
        
        duplicate_count = combined_ledger.index.duplicated().sum()
        if duplicate_count > 0:
            print(f"  -> [INFO] {duplicate_count} barcodes exist in historical ledger.")
            print("  -> Overwriting historical state with fresh RAM state...")
            combined_ledger = combined_ledger[~combined_ledger.index.duplicated(keep='last')]
    else:
        print("  -> No existing ledger found. Initiating genesis ledger...")
        combined_ledger = current_run_ledger
        
    final_duplicate_count = combined_ledger.index.duplicated().sum()
    if final_duplicate_count > 0:
        print("[ERROR] Unresolvable BARCODE COLLISION DETECTED.")
        return None
        
    Path(master_df_csv_path).parent.mkdir(parents=True, exist_ok=True)
    combined_ledger.to_csv(master_df_csv_path, index_label='cell_barcode')
    
    print(f"[SUCCESS] Ledger permanently sealed. Total unique cells: {len(combined_ledger)}")



def main_artifact_labelling(main_h5ad_path: str, master_df_csv_path: str) -> str:
    """
    Ingests the master CSV ledger and maps the final biological identities 
    onto the raw, un-split global matrix. Exports the final ML-Ready artifact.

    Parameters
    ----------
    main_h5ad_path : str
        Path to the primary QC'd AnnData file.
    master_df_csv_path : str
        Path to the aggregated master CSV ledger.

    Returns
    -------
    str
        Path to the finalized ML-Ready .h5ad file.
    """
    print("\n===========================================================")
    print(" EXECUTING FINAL TENSOR RECOMBINATION")
    print("===========================================================")
    
    if not op.exists(main_h5ad_path):
        print(f"[ERROR] Global matrix missing at {main_h5ad_path}")
        return None
        
    if not op.exists(master_df_csv_path):
        print(f"[ERROR] Master ledger missing at {master_df_csv_path}")
        return None
        
    master_df = pd.read_csv(master_df_csv_path, index_col=0)
    adata_main = load_evidence(main_h5ad_path)
    
    # The Vectorized Recombination
    adata_main.obs['Final_ML_Ready_Label'] = adata_main.obs_names.map(master_df['manual_labels'])
    adata_main.obs['Final_ML_Ready_CL_ID'] = adata_main.obs_names.map(master_df['human_CL_ID'])
    
    void_count = adata_main.obs['Final_ML_Ready_Label'].isna().sum()
    if void_count > 0:
        print(f"  -> [WARNING] {void_count} cells did not map (QC voids or unannotated).")
        print("  -> Standardizing voids as 'Unknown/Filtered'.")
        adata_main.obs['Final_ML_Ready_Label'].fillna('Unknown/Filtered', inplace=True)
        adata_main.obs['Final_ML_Ready_CL_ID'].fillna('Unknown/Filtered', inplace=True)
        
    base_name, ext = op.splitext(main_h5ad_path)
    ml_ready_path = f"{base_name}_ML_Ready{ext}"
    
    print(f"[SEALED] Writing finalized Machine Learning Tensor to: {ml_ready_path}")
    adata_main.write_h5ad(ml_ready_path)
    
    del adata_main
    gc.collect()
    
    return ml_ready_path



if __name__ == '__main__':
    # Absolute Definitions
    main_h5ad_path = './data/objects/pbmc3k_qc.h5ad'
    dict_file_training_path = './data/objects/Dictionary_of_returns_from_orch_A.json'
    dict_file_projected_path = './data/objects/Dictionary_of_returns_from_orch_B.json'
    annotations_path = './data/objects/annotation_manual_empty.json'
    universal_ontology_id_path = './data/universal_ontology_id_dict.json'
    master_df_csv_path = './data/objects/master_labels_df.csv'
    
    # 1. Map Training Set
    orc_annotation(dict_file_training_path, annotations_path, universal_ontology_id_path)
    label_mapping_data_frame_all(dict_file_training_path, master_df_csv_path)
    
    # 2. Map Projected Set
    orc_annotation(dict_file_projected_path, annotations_path, universal_ontology_id_path)
    label_mapping_data_frame_all(dict_file_projected_path, master_df_csv_path)
    
    # 3. Final Artifact Recombination
    main_artifact_labelling(main_h5ad_path, master_df_csv_path)
    
    print("\n[SUCCESS] PIPELINE COMPLETE. ML TENSOR IS READY FOR DEPLOYMENT.")