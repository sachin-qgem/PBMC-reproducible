import gc
import json
import os
import os.path as op
from pathlib import Path

import anndata as ad
import pandas as pd
import scanpy as sc
from sklearn.metrics import adjusted_rand_score

ad.settings.allow_write_nullable_strings = True

sc.settings.verbosity = 0
plt_fig_dir = Path('./results/figures/p06_annotation')
plt_fig_dir.mkdir(parents=True, exist_ok=True)

def load_matrix(h5ad_path: str) -> ad.AnnData:
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
    print(f"[LOG] Loading matrix: {h5ad_path}")
    adata = sc.read_h5ad(h5ad_path)
    print(f"[LOG] Dimensions: {adata.n_obs} cells x {adata.n_vars} genes")
    
    return adata


def map_annotations(
    dict_path: str, 
    annotation_manual_path: str, 
    ontology_cl_id_dict_path: str
) -> None:
    """
    Assigns biological annotations and standard Cell Ontology (CL) IDs to matrices.

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
    print("[LOG] Annotation mapping initiated.")
    
    if not op.exists(dict_path):
        print(f"[ERROR] Matrix dictionary missing at {dict_path}")
        return None
        
    with open(dict_path, 'r') as json_file:
        state_dict = json.load(json_file)
        
    if not op.exists(annotation_manual_path):
        print(f"[ERROR] Required annotation JSON ledgers are missing at {annotation_manual_path}")
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
        print("[ERROR] Required keys missing from matrix dictionary.")
        return None

    
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
            print(f"[LOG] MACRO Mapping annotations for clustering key: {macro_leiden}")
            adata = load_matrix(macro_path)
            
            if not annotation_manual[macro_leiden].values():
                print(f"[ERROR] Annotation JSON is empty. Manual population required.")
            else:
                adata.obs['manual_labels'] = adata.obs[macro_leiden].map(annotation_manual[macro_leiden])
                adata.obs['human_CL_ID'] = adata.obs['manual_labels'].map(ontology_cl_id_dict_manual[macro_leiden])
            
            # --- V3 GITHUB ISSUE: CELLTYPIST ORACLE ARI TEMPORARILY Removed ---
                # if 'majority_voting' in adata.obs:
                #     adata.obs['oracle_CL_ID'] = adata.obs['majority_voting'].map(ontology_cl_id_dict_manual[macro_leiden_key])
                #
                # valid_mask = adata.obs['human_CL_ID'].notna() & adata.obs['oracle_CL_ID'].notna()
                # if valid_mask.sum() > 0:
                #     ari_score = adjusted_rand_score(
                #         adata.obs.loc[valid_mask, 'human_CL_ID'].astype(str),
                #         adata.obs.loc[valid_mask, 'oracle_CL_ID'].astype(str)
                #     )
                #     adata.uns['Oracle_ARI_Score'] = float(ari_score)
                #     print(f" -> Oracle Alignment Score (ARI): {ari_score:.3f}")
                # ------------------------------------------------------------------

            if '_index' in adata.obs.columns:
                del adata.obs['_index']
            if adata.obs.index.name == '_index':
                adata.obs.index.name = None

            adata.write_h5ad(macro_path)
            del adata
            gc.collect()
        else:
            print(f"[ERROR] Macro file missing at {macro_path}")


    micro_paths_dict = state_dict.get(micro_paths_key, {})
    micro_leiden_dict = state_dict.get(micro_leiden_key, {})
    
    for leiden_dict_key, file_path in micro_paths_dict.items():
        if not op.exists(file_path):
            print(f"[LOG] Matrix missing at {file_path}")
            continue
            
        active_leiden_col = micro_leiden_dict.get(leiden_dict_key)
        adata = load_matrix(file_path)
        
        if active_leiden_col is None:
            clean_key = leiden_dict_key.replace('_Terminal_State', '')
            parts = clean_key.split('_')
            cluster_id = parts[-1]
            parent_dict_key = '_'.join(parts[:-1])
            
            inherited_label = annotation_manual.get(parent_dict_key, {}).get(cluster_id)
            inherited_cl_id = ontology_cl_id_dict_manual.get(parent_dict_key, {}).get(cluster_id)
            print(f"[LOG] Isotropic variance detected. Inheriting Parent Label: '{inherited_label}'")
            if inherited_label is None:
                print(f"[ERROR] Annotation JSON is empty for parent key: {parent_dict_key}.")
            else:
                adata.obs['manual_labels'] = inherited_label
                adata.obs['human_CL_ID'] = inherited_cl_id
            
        else:
            
            print(f"[LOG] Active State detected. Mapping via: '{active_leiden_col}'")
            if not annotation_manual[active_leiden_col].values():
                print(f"[ERROR] Annotation JSON is empty for key: {active_leiden_col}.")
            else:
                adata.obs['manual_labels'] = adata.obs[active_leiden_col].map(
                annotation_manual[active_leiden_col]
            )
                adata.obs['human_CL_ID'] = adata.obs['manual_labels'].map(ontology_cl_id_dict_manual[active_leiden_col])
                
                # --- V3 GITHUB ISSUE: CELLTYPIST ORACLE ARI TEMPORARILY Removed ---
                # if 'majority_voting' in adata.obs:
                #     adata.obs['oracle_CL_ID'] = adata.obs['majority_voting'].map(ontology_cl_id_dict_manual[active_leiden_col])
                # 
                # valid_mask = adata.obs['human_CL_ID'].notna() & adata.obs['oracle_CL_ID'].notna()
                # if valid_mask.sum() > 0:
                #     ari_score = adjusted_rand_score(
                #         adata.obs.loc[valid_mask, 'human_CL_ID'].astype(str),
                #         adata.obs.loc[valid_mask, 'oracle_CL_ID'].astype(str)
                #     )
                #     adata.uns['Oracle_ARI_Score'] = float(ari_score)
                #     print(f" -> Oracle Alignment Score (ARI): {ari_score:.3f}")
                # ------------------------------------------------------------------
        
        if '_index' in adata.obs.columns:
            del adata.obs['_index']
        if adata.obs.index.name == '_index':
            adata.obs.index.name = None    
        
        adata.write_h5ad(file_path)
        del adata
        gc.collect()


def aggregate_dataframes(dict_path: str, master_df_csv_path: str) -> None:
    """
    Extracts labels from all isolated matrices and concatenates them into a central CSV.

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
    print("\n[LOG] Aggregating labels into central dataframe...")
    
    if not op.exists(dict_path):
        print(f"[ERROR] Matrix dictionary missing at {dict_path}")
        return None
        
    with open(dict_path, 'r') as json_file:
        state_dict = json.load(json_file)
        
    micro_paths_key = None
    for k in state_dict.keys():
        if "micro" in k and "file_path" in k and "dictionary" in k:
            micro_paths_key = k
            
    if not micro_paths_key:
        print("[ERROR] Required micro paths key missing from matrix dictionary")
        return None
        
    micro_paths_dict = state_dict.get(micro_paths_key, {})
    new_df_list = []
    
    for key_cluster_id, file_path in micro_paths_dict.items():
        if not op.exists(file_path):
            print(f"[ERROR] Missing matrix at {file_path}.")
            continue
            
        adata_micro = load_matrix(file_path)
        
        if 'manual_labels' in adata_micro.obs and 'human_CL_ID' in adata_micro.obs:
            clean_df = adata_micro.obs[['manual_labels', 'human_CL_ID']].copy()
            new_df_list.append(clean_df)
        else:
            print(f"[ERROR] Labels missing in {key_cluster_id}.")
            
        del adata_micro
        gc.collect()
        
    if not new_df_list:
        print("[ERROR] No valid dataframes extracted.")
        return None
        
    current_run_dataframe = pd.concat(new_df_list)
    
    if op.exists(master_df_csv_path):
        print("  -> Existing ledger detected. Loading into RAM for integration...")
        master_df = pd.read_csv(master_df_csv_path, index_col=0)
        combined_dataframe = pd.concat([master_df, current_run_dataframe])
        duplicate_count = combined_dataframe.index.duplicated().sum()

        if duplicate_count > 0:
            combined_dataframe = combined_dataframe[~combined_dataframe.index.duplicated(keep='last')]

    else:
        print("[LOG] Initializing primary dataframe...")
        combined_dataframe = current_run_dataframe
        
    final_duplicate_count = combined_dataframe.index.duplicated().sum()
    if final_duplicate_count > 0:
        return None
        
    Path(master_df_csv_path).parent.mkdir(parents=True, exist_ok=True)
    combined_dataframe.to_csv(master_df_csv_path, index_label='cell_barcode')
    
    print(f"[LOG] Dataframe saved. Unique barcodes: {len(combined_dataframe)}")



def integrate_global_metadata(main_h5ad_path: str, master_df_csv_path: str) -> str:
    """
    Maps annotations from the central CSV onto the primary global matrix.

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
    print("[LOG] Integrating metadata into global matrix.")

    if not op.exists(main_h5ad_path) or not op.exists(master_df_csv_path):
        return None
        
    master_df = pd.read_csv(master_df_csv_path, index_col=0)
    adata_main = load_matrix(main_h5ad_path)
    
    adata_main.obs['Final_ML_Ready_Label'] = adata_main.obs_names.map(master_df['manual_labels'])
    adata_main.obs['Final_ML_Ready_CL_ID'] = adata_main.obs_names.map(master_df['human_CL_ID'])
    
    void_count = adata_main.obs['Final_ML_Ready_Label'].isna().sum()
    if void_count > 0:
        print(f"[ERROR] {void_count} cells did not map.")
        adata_main.obs['Final_ML_Ready_Label'].fillna('Unknown/Filtered', inplace=True)
        adata_main.obs['Final_ML_Ready_CL_ID'].fillna('Unknown/Filtered', inplace=True)
        
    base_name, ext = op.splitext(main_h5ad_path)
    ml_ready_path = f"{base_name}_ML_Ready{ext}"
    
    print(f"[LOG] Saving ML-ready matrix to: {ml_ready_path}")
    
    if '_index' in adata_main.obs.columns:
        del adata_main.obs['_index']
    if adata_main.obs.index.name == '_index':
        adata_main.obs.index.name = None
    if '_index' in adata_main.var.columns:
        del adata_main.var['_index']
    if adata_main.var.index.name == '_index':
        adata_main.var.index.name = None

    adata_main.write_h5ad(ml_ready_path)
    
    del adata_main
    gc.collect()
    
    return ml_ready_path



def main():
    
    sc.settings.figdir = "./results/figures/p06_annotation"
    os.makedirs(sc.settings.figdir, exist_ok=True)
    
    
    main_h5ad_path = './data/objects/pbmc3k_qc.h5ad'
    dict_file_training_path = './data/objects/Dictionary_of_returns_from_orch_A.json'
    dict_file_projected_path = './data/objects/Dictionary_of_returns_from_orch_B.json'
    annotations_path = './data/objects/annotation_manual_empty.json'
    ontology_id_path = './data/objects/ontology_cl_id_manual_empty.json'
    master_df_csv_path = './data/objects/master_labels_df.csv'
    
    map_annotations(dict_file_training_path, annotations_path, ontology_id_path)
    aggregate_dataframes(dict_file_training_path, master_df_csv_path)
    
    map_annotations(dict_file_projected_path, annotations_path, ontology_id_path)
    aggregate_dataframes(dict_file_projected_path, master_df_csv_path)
    
    integrate_global_metadata(main_h5ad_path, master_df_csv_path)
    
    print("\n[LOG] Phase IV pipeline complete. ML-ready matrix exported.")

if __name__ == '__main__':
    main()
