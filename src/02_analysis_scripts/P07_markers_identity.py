import scanpy as sc
import anndata as ad
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
import os.path as op
import celltypist as ct
ad.settings.allow_write_nullable_strings = True
sc.settings.verbosity = 0

def load_evidence(h5ad_path):
    print('Loading Evidence from the previous Projected dataset split B')
    adata = sc.read_h5ad(h5ad_path)
    print(f'Loaded B set matrix of dimensions \
          {adata.n_obs} cells and {adata.n_vars} genes')
    return adata

def rank_gene_groups_wilcoxon(adata_path,leiden_key,quantile=0.9375):

    adata = load_evidence(adata_path)
    sc.tl.rank_genes_groups(adata,groupby=leiden_key,method='wilcoxon',tie_correct=True,layer='log1p_norm',pts=True)
    sc.tl.filter_rank_genes_groups(adata,use_raw=False)
    df = sc.get.rank_genes_groups_df(adata,group=None,key='rank_genes_groups_filtered')
    pvals_adj_logfc_mask = (df['pvals_adj']<0.05) & (df['logfoldchanges']<10.0)
    df_new = df[pvals_adj_logfc_mask].dropna(subset=['names']).copy()
    df_new['nlog10pval_adj'] = -np.log10(df_new['pvals_adj'])
    df_new['local_Q93'] = df_new.groupby('group')['nlog10pval_adj'].transform(lambda x:x.quantile(quantile))
    df_new['violin_delta'] = (df_new['pct_nz_group']) - (df_new['pct_nz_reference'])
    mask_1 = (df_new['nlog10pval_adj'] >= df_new['local_Q93'])
    df_mask_1 = df_new[mask_1].copy()
    df_mask_1_sorted = df_mask_1.sort_values(by=['group','violin_delta'],ascending=[True,False])
    df_final = df_mask_1_sorted.groupby('group').head(3).copy()
    
    grouped_top_genes = {}
    for cluster_id in df_final['group'].unique():
        grouped_top_genes[str(cluster_id)] = df_final[df_final['group'] == cluster_id]['names'].tolist()
    
    sc.tl.dendrogram(adata, groupby=leiden_key)

    safe_fig_name = f'_{leiden_key}_top_genes.svg'
    
    sc.pl.dotplot(
        adata,
        var_names=grouped_top_genes,
        groupby=leiden_key,
        standard_scale='var', 
        dendrogram=True,
        cmap='Reds',           
        use_raw=False,
        show=False,            
        save=safe_fig_name
    )
    plt.close('all')
    adata.uns['final_top_genes_per_cluster'] = df_final
    if 'rank_genes_groups_filtered' in adata.uns:
        del adata.uns['rank_genes_groups_filtered']
    adata.write_h5ad(adata_path)
    del adata,df,df_new,df_mask_1,df_mask_1_sorted,df_final
    import gc
    gc.collect()
    
def auto_ref_mapping(adata_path,model_type,leiden_key):
    adata = load_evidence(adata_path)

    adata_hold = adata.X
    adata.X = adata.layers['log1p_norm'].copy()
    predictions = ct.annotate(adata, model=model_type,over_clustering=leiden_key, majority_voting=True)
    predictions.to_adata()
    #adata.obs['celltypist_prediction'] = predictions.predicted_labels['predicted_labels']
    #adata.obs['celltypist_majority_voting'] = predictions.predicted_labels['majority_voting']
    #adata.obs['celltypist_confidence'] = predictions.predicted_labels['conf_score']
    adata.X = adata_hold
    adata.write_h5ad(adata_path)
    del adata,adata_hold,predictions
    import gc
    gc.collect()

def orc_project(Dictionary_of_returns_from_orch_B_path):
    import json
    macro_model = ct.models.Model.load(model = 'Immune_All_High.pkl')
    micro_model = ct.models.Model.load(model = 'Immune_All_Low.pkl')
    if not op.exists(Dictionary_of_returns_from_orch_B_path):
        print(f'Dictionary does not exists at {Dictionary_of_returns_from_orch_B_path}')
        return None
    with open(Dictionary_of_returns_from_orch_B_path,'r') as json_file:
        Dictionary_of_returns_from_orch_B = json.load(json_file)
    print('Please choose from the keys to access values')

    macro_path_key = None
    macro_leiden_key = None
    micro_paths_key = None
    micro_leiden_key = None
    for k in Dictionary_of_returns_from_orch_B.keys():
        if "macro" in k and "file_path" in k and "dictionary" not in k:
            macro_path_key = k
        elif "macro" in k and "leiden_key" in k and "dictionary" not in k:
            macro_leiden_key = k
        elif "micro" in k and "file_path" in k and "dictionary" in k:
            micro_paths_key = k
        elif "micro" in k and "leiden_key" in k and "dictionary" in k:
            micro_leiden_key = k
    if not all([macro_path_key, macro_leiden_key, micro_paths_key, micro_leiden_key]):
        print("CRITICAL: Matrix collapse. Could not autonomously identify core keys.")
        return None
    macro_path = Dictionary_of_returns_from_orch_B.get(macro_path_key)
    macro_leiden = Dictionary_of_returns_from_orch_B.get(macro_leiden_key)
    if macro_path and macro_leiden:
        if op.exists(macro_path):
            print(f"\n[MACRO] Locking anchor '{macro_leiden}' for path: {macro_path}")
            rank_gene_groups_wilcoxon(macro_path, macro_leiden)
            print("[MACRO] Thermodynamic extraction complete and sealed to disk.")
            auto_ref_mapping(macro_path,macro_model,macro_leiden)
        else:
            print(f"[MACRO] ERROR: File missing at {macro_path}")
    micro_paths_dict = Dictionary_of_returns_from_orch_B.get(micro_paths_key, {})
    micro_leiden_dict = Dictionary_of_returns_from_orch_B.get(micro_leiden_key, {})
    for leiden_key, file_path in micro_paths_dict.items():
        print(f"\nScanning: {leiden_key}")
        
        if not op.exists(file_path):
            print(f"  -> ERROR: Matrix missing at {file_path}. Bypassing.")
            continue
            
        # The Safety Gate for the Terminal State
        leiden_key = micro_leiden_dict.get(leiden_key)
        
        if leiden_key is None:
            print(f"  -> TERMINAL STATE DETECTED. No Leiden key exists. Bypassing Wilcoxon.")
            continue
            
        print(f"  -> Anchor locked: '{leiden_key}'. Executing Wilcoxon Engine...")
        rank_gene_groups_wilcoxon(file_path, leiden_key)
        auto_ref_mapping(file_path,micro_model,leiden_key)
    

if __name__ == '__main__':
    ct.models.models_path = '/Users/qgem/GitHub/PBMC3k-reproducible/data/celltypist_models'
    ct.models.download_models(force_update=True, model=['Immune_All_Low.pkl','Immune_All_High.pkl'])

    Dictionary_file_path = '/Users/qgem/GitHub/PBMC3k-reproducible/notebooks/results/Dictionary_of_returns_from_orch_B.json'
    orc_project(Dictionary_file_path)