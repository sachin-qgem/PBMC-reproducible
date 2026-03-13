import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np
import anndata as ad
import pandas as pd
from sklearn.metrics import adjusted_rand_score
import seaborn as sns
from pathlib import Path
import json
ad.settings.allow_write_nullable_strings = True
sc.settings.figdir = "./results/figures/phase5_B_clustered_geometry"
sc.settings.verbosity = 3


def load_evidence(h5ad_path):
    '''
     Loading Evidence
    '''
    # Loads the artifact
    print(f"   -> Loading artifact from: {h5ad_path}")
    adata = sc.read_h5ad(h5ad_path)
    print(f"   -> Loaded dimensions: {adata.n_obs} cells x {adata.n_vars} genes")
    return adata


def random_split_data(h5ad_path,save_folder_path):
    '''
    This function Divide the main adata into two datasets
      as the training and the projectable (A and B)
    '''
    adata = load_evidence(h5ad_path)
    from sklearn.model_selection import train_test_split
    train_data_indices, project_data_indices = train_test_split(adata.obs_names,test_size=0.5,
                                                     random_state=42,shuffle=True)
    adata_train = adata[train_data_indices].copy()
    adata_project = adata[project_data_indices].copy()
    adata_train_file_path = f'{save_folder_path}/adata_train.h5ad'
    adata_project_file_path = f'{save_folder_path}/adata_project.h5ad'
    adata_train.write_h5ad(adata_train_file_path)
    adata_project.write_h5ad(adata_project_file_path)
    del adata_project,adata_train
    import gc
    gc.collect()
    
    file_path_dictionary = {
        'training_file': adata_train_file_path,
        'projectable_file': adata_project_file_path
    }
    print(f'returning {file_path_dictionary} for refrence of the training and projectable file paths to be used')
    return file_path_dictionary


def knn_umap_leiden(training_side_file_path,n_neighbors:int,n_pcs:int,leiden_res:float,key_name:str|None=None):
    '''
    This can be used to run KNN and umap based on either to create macro neighbors
    and UMAPs. Based on the key_name passed as example 'macro' or 'micro' 
    This would then have the key names as 'macro_neighbors' and 'macro_umap'
    OR 'micro_neighbors' and 'micro_umap' or anything thats chosen
    '''
    import gc
    leiden_key = None
    neighbors_key = None
    umap_key_added = None
    adata = load_evidence(training_side_file_path)
    adata_su_check = adata.copy()
    if adata.n_obs>250:
        key_added = f"{key_name}_neighbors" if key_name is not None else None
        
        sc.pp.neighbors(adata,n_neighbors = n_neighbors,n_pcs=n_pcs,method='umap',
                        knn=True,metric='euclidean',random_state=42,
                        key_added=key_added)
        
        
        neighbors_key = f"{key_name}_neighbors" if key_name is not None else 'neighbors'
        umap_key_added = f"{key_name}_umap" if key_name is not None else None
        
        sc.tl.umap(adata,maxiter=500,random_state=42,key_added=umap_key_added,
                neighbors_key=neighbors_key,min_dist=0.1,spread=1.0)
        

        leiden_key = f'{key_name}_leiden' if key_name is not None else 'leiden'
        sc.tl.leiden(adata,resolution=leiden_res,n_iterations= -1,
                    flavor='leidenalg',random_state=42,key_added=leiden_key,
                    neighbors_key=neighbors_key)


        plot_basis = umap_key_added if umap_key_added is not None else 'X_umap'
        sc.pl.embedding(adata,basis=plot_basis,color=leiden_key,
                components ='all',size= 50.0,color_map = 'Blues',show=True,
                title = 'Training Manifold',legend_loc = 'on data',
                legend_fontsize = 'x-small',legend_fontweight = 'bold',
                legend_fontoutline = 3)
        
        print(f"\nSubjecting '{key_name}' boundaries to thermodynamic stress test...")
        n_iterations = 20
        subsample_fraction = 0.8
        
        original_labels = adata.obs[leiden_key].astype(str)
        unique_clusters = original_labels.unique()
        jaccard_ledger = {cluster: [] for cluster in unique_clusters}
        
        for i in range(n_iterations):
            
            n_keep = int(adata_su_check.n_obs * subsample_fraction)
            surviving_indices = np.random.choice(adata_su_check.obs_names, size=n_keep, replace=False)
            adata_sub = adata_su_check[surviving_indices].copy()
            
            
            sc.pp.neighbors(adata_sub, n_neighbors=n_neighbors,n_pcs =n_pcs,method = 'umap',knn = True,metric = 'euclidean',
                            random_state=42, use_rep='X_pca', key_added='boot_neighbors')
            sc.tl.leiden(adata_sub, resolution=leiden_res, neighbors_key='boot_neighbors', key_added='boot_leiden',n_iterations=-1,flavor='leidenalg',
                         random_state=42,)
            new_labels = adata_sub.obs['boot_leiden'].astype(str)
            
            # The Jaccard Intersection
            for orig_cluster in unique_clusters:
                # Find physical barcodes of the original cluster that survived
                orig_cells_in_sub = adata_sub.obs_names[original_labels[surviving_indices] == orig_cluster]
                
                if len(orig_cells_in_sub) == 0:
                    jaccard_ledger[orig_cluster].append(0.0)
                    continue
                    
                # Maximum Majority Rule: Find new amnesiac cluster identity
                best_match_cluster = new_labels[orig_cells_in_sub].value_counts().index[0]
                
                # Jaccard calculation: |A ∩ B| / |A ∪ B|
                set_A = set(orig_cells_in_sub)
                set_B = set(adata_sub.obs_names[new_labels == best_match_cluster])
                
                intersection = len(set_A.intersection(set_B))
                union = len(set_A.union(set_B))
                
                jaccard_score = intersection / union if union > 0 else 0.0
                jaccard_ledger[orig_cluster].append(jaccard_score)
                
            # Internal Loop Cleanup
            del adata_sub
            gc.collect()

        # 3. THE FINAL JUDGMENT & LEDGER INSCRIPTION
        print(f"--- 5-SIGMA UNCERTAINTY DIAGNOSTIC: {key_name} ---")
        su_grades_for_disk = {}
        for orig_cluster, scores in jaccard_ledger.items():
            mean_score = np.mean(scores)
            
            if mean_score >= 0.85:
                grade = "[ABSOLUTE WALL]"
            elif mean_score >= 0.60:
                grade = "[STABLE BIOLOGY]"
            else:
                grade = "[FRAGILE GRADIENT]"
                
            print(f" Cluster {orig_cluster}: Jaccard = {mean_score:.3f}  {grade}")
            # Save strictly the float value for the matrix ledger
            su_grades_for_disk[orig_cluster] = float(mean_score)
            
        # Mathematically burn the grades into the matrix memory
        adata.uns[f'{leiden_key}_SU_grades'] = su_grades_for_disk
        # =========================================================================


        adata.write_h5ad(training_side_file_path)
    del adata,adata_su_check
    gc.collect()
    return leiden_key,neighbors_key
    
def stability_audit(training_filepath,key_for_saving_images):
    '''
    This stability audit function is to be later made more dynamic and subdivided to have input from the resolution
    sweep, to be taken into account for the subsampling robustness ARI check. A reminder is being put here
    '''
    from sklearn.model_selection import train_test_split

    adata_A = load_evidence(training_filepath)
    if adata_A.n_obs>250:
        print("   -> INITIATING FORENSIC STABILITY AUDIT...")
        
        # ---------------------------------------------------------
        # TEST 1: PARAMETER STABILITY (The Resolution Sweep)
        # ---------------------------------------------------------
        print("   -> [TEST 1] Resolution Sweep (0.2 to 2.0)...")
        resolutions = np.arange(0.1,2.1,0.03).tolist()
        results = []
        sc.pp.neighbors(adata_A,n_neighbors = 15,n_pcs=10,method='umap',
                        knn=True,metric='euclidean',random_state=42)
        # Run Clustering at all resolutions
        for res in resolutions:
            key = f"leiden_res_{res}"
            sc.tl.leiden(adata_A, resolution=res, key_added=key, flavor='leidenalg')
            # Store count
            n_clusters = len(adata_A.obs[key].unique())
            results.append({'res': res, 'n_clusters': n_clusters, 'key': key})
        
        # Compute ARI between NEIGHBORS (The Plateau Check)
        df_res = pd.DataFrame(results)
        aris = [0.0] # First one has no previous neighbor
        for i in range(1, len(df_res)):
            prev_key = df_res.iloc[i-1]['key']
            curr_key = df_res.iloc[i]['key']
            # PHYSICS: Compare N vs N-1
            score = adjusted_rand_score(adata_A.obs[prev_key],
                                        adata_A.obs[curr_key])
            aris.append(score)
        
        df_res['neighbor_ari'] = aris
        
        # VISUALIZE THE PLATEAU
        plt.figure(figsize=(25, 13))
        sns.lineplot(data=df_res, x='res', y='n_clusters', marker='o', label='Cluster Count')
        ax2 = plt.twinx()
        sns.lineplot(data=df_res, x='res', y='neighbor_ari',
                    color='red', marker='x', ax=ax2, 
                    label='Stability (ARI)') # type: ignore
        plt.title("Parameter Stability: Look for High ARI + Flat Cluster Count")
        plt.savefig(f"/Users/qgem/GitHub/PBMC3k-reproducible/notebooks/figures/{key_for_saving_images}_stability_resolution_sweep.png")
        plt.close()
        

        # ---------------------------------------------------------
        # TEST 2: STRUCTURAL STABILITY (The Subsampling Shake)
        # ---------------------------------------------------------
        print("   -> [TEST 2] Subsampling Earthquake (80% Retention)...")
        print("   -> Please Pick a reference resolution (e.g., 1.375 or whatever the sweep suggests)")
        # 1. Pick a reference resolution (e.g., 1.375 or whatever the sweep suggests)
        ref_res = float(input(f'reference resolution for {key_for_saving_images}: '))
        # Ensure reference exists
        if f'leiden_res_{ref_res}' not in adata_A.obs:
            sc.tl.leiden(adata_A, resolution=ref_res,
                        key_added=f'leiden_res_{ref_res}', flavor='leidenalg')
        
        # 2. The Shake (Subsample 80%)
        # We use sc.pp.subsample to get a NEW object
        A_data_test_sub_indices, B_data_test_sub_indices = train_test_split(
            adata_A.obs_names,test_size=0.2,random_state=42,shuffle=True)
        adata_A_test_sub = adata_A[A_data_test_sub_indices].copy()
        
        # 3. The Re-Build (Must re-run Neighbors to be valid!)
        sc.pp.neighbors(adata_A_test_sub, n_neighbors=15, n_pcs=10, use_rep='X_pca')
        sc.tl.leiden(adata_A_test_sub, resolution=ref_res, key_added='leiden_sub', 
                    flavor='leidenalg')
        
        # 4. The Comparison (Intersecting Indices)
        # We compare the Original Label (in adata) vs New Label (in adata_sub) 
        # for the same cells
        original_labels = adata_A.obs.loc[adata_A_test_sub.obs_names,
                                                f'leiden_res_{ref_res}']
        new_labels = adata_A_test_sub.obs['leiden_sub']
        
        robustness_score = adjusted_rand_score(original_labels, new_labels)
        print(f"   -> Structural Robustness Score (ARI): {robustness_score:.3f}")
        
        if robustness_score < 0.7:
            print("     WARNING: Clusters are unstable! They collapsed under subsampling.")
        else:
            print("     PASSED: Clusters are physically robust.")

        # ---------------------------------------------------------
        # TEST 3: BIOLOGICAL SANITY CHECK (The Identity Verification)
        # ---------------------------------------------------------
        
        print(f"   -> [TEST 3] Biological Sanity Check (at Res {ref_res})...")
        # Ensure reference exists
        if f'leiden_res_{ref_res}' not in adata_A.obs:
            sc.tl.leiden(adata_A, resolution=ref_res,
                        key_added=f'leiden_res_{ref_res}', flavor='leidenalg')
        
        # A. Quick Rank Genes (Wilcoxon is standard, t-test is faster for audit)
        # Physics: We ask "What makes Cluster X different from the rest?"
        sc.tl.rank_genes_groups(adata_A, groupby=f'leiden_res_{ref_res}',
                                method='wilcoxon')
        
        # B. Print Top 3 Markers per Cluster
        print("      -> Top 3 Marker Genes per Cluster:")
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        
        # Extract results structure
        result_df = pd.DataFrame(adata_A.uns['rank_genes_groups']['names']).head(3)
        print(result_df)
        
        # C. VISUAL SANITY: The Canonical Marker DotPlot
        # These are the "Known Truths" for PBMC. If clusters don't light up 
        # correctly, the resolution is wrong.
        marker_genes_dict = {
            'B-cell': ['MS4A1', 'CD79A'],
            'T-cell': ['CD3D', 'IL7R', 'CD8A'],
            'NK': ['GNLY', 'NKG7'],
            'Mono': ['CD14', 'FCGR3A'],
            'Dendritic': ['FCER1A', 'CST3'],
        'Platelet': ['PPBP']
        }
        
        # Filter markers to ensure they exist in the dataset (prevent errors)
        valid_markers = {k: [g for g in v if g in adata_A.var_names] 
                        for k, v in marker_genes_dict.items()}
        
        print("      -> Generating Sanity DotPlot...")
        dp = sc.pl.dotplot(adata_A, valid_markers, groupby=f'leiden_res_{ref_res}',
                            standard_scale='var', show=False)
        plt.savefig(f"/Users/qgem/GitHub/PBMC3k-reproducible/notebooks/figures/{key_for_saving_images}_stability_biological_sanity.png")
        plt.close()
        del adata_A,adata_A_test_sub
        import gc
        gc.collect()
        return ref_res

def divide_and_save_dataset_based_on_macro_or_micro_clusters(file_path_used,leiden_key_as_cluster_column_name):
    '''
    This divides the set A according to macro clustered ID cells to be later 
    run PCAs, knn etc on them, using the key from the leiden cluster method 
    return value of 'leiden_key' as to be passed in leiden_key_as_cluster_column_name
    '''
    import gc
    adata = load_evidence(file_path_used)
    grouped_barcodes = adata.obs.groupby(leiden_key_as_cluster_column_name).groups
    saved_filepaths_dictionary = {} 
    path_object = Path(file_path_used)
    directory = path_object.parent
    base_name = path_object.stem

    for cluster_id,barcodes in grouped_barcodes.items():
        lineage_key = f'{leiden_key_as_cluster_column_name}_{cluster_id}'
        adata_subset = adata[barcodes].copy()
        
        if adata_subset.n_obs>250:
            new_filename = f'{base_name}_{lineage_key}.h5ad'
        else:
            print(f"Island {cluster_id} has mass {adata_subset.n_obs}. Branded as Terminal_State.")
            lineage_key = f'{leiden_key_as_cluster_column_name}_{cluster_id}_Terminal_State'
            new_filename = f'{base_name}_{lineage_key}.h5ad'

        new_filepath_obj = directory/new_filename
        new_filepath_string = str(new_filepath_obj)
        adata_subset.write_h5ad(new_filepath_string)
        saved_filepaths_dictionary[lineage_key] = new_filepath_string
        del adata_subset
        gc.collect()
    
    print(f'returning {saved_filepaths_dictionary} for refrence of the cluster based divided file paths to be used')
    del adata
    gc.collect()
    return saved_filepaths_dictionary

def npr_hvg_pca_recal(filepath,keys):
    '''
    Recalculating npr and HVG on the clsutered micro files
    '''
    # we will tag the genes as HVG or not
    # Taking theta Value to be 100.0 as from experiments  
    
    adata = load_evidence(filepath)
   
    adata.X = adata.layers['counts'].copy()
        
    if adata.n_obs>250:
        sc.experimental.pp.highly_variable_genes(adata,theta=100.0,n_top_genes=2500,flavor='pearson_residuals',subset=False,layer='counts')
        # Pearson residuals calculation
        sc.experimental.pp.normalize_pearson_residuals(adata, theta = 100.0)
        sc.pp.pca(adata,n_comps=100,zero_center=True,svd_solver='arpack',
                mask_var='highly_variable')
        sc.pl.pca_variance_ratio(adata,n_pcs=100,save=f"_{keys}_.png",show=False)
        adata.write_h5ad(filepath)
    
    del adata
    import gc
    gc.collect()
        
def cast_projectable_data_on_training_data(adata_project_file_path, adata_training_file_path,neighbors_key_training,leiden_key_training):
    '''
    This we can run on the any stage to project B on macro or micro , didvide using the
    divide function created earlier, takes the obs key of color from the return of the
    clustering_leiden_plot_umap method, at this mmoment I think it to be inlcuded in the orchestrator A
    function. But would think tommmorow if there could be another way
    '''

    adata_project = load_evidence(adata_project_file_path)
    adata_training = load_evidence(adata_training_file_path)
    adata_project.X = adata_project.layers['counts'].copy()

    # =========================================================================
    # THE MANUAL PEARSON RESIDUAL ENGINE
    # =========================================================================
    
    # 2. Extract the Architect's Baseline (Set A's p_j)
    counts_A = adata_training.layers['counts']
    gene_sums_A = np.asarray(counts_A.sum(axis=0)).flatten()
    total_sum_A = counts_A.sum()
    p_j_A = gene_sums_A / total_sum_A  # The absolute reference frequencies
    
    # 3. Extract the Shadow's Mass (Set B's n_i)
    counts_B = adata_project.layers['counts']
    cell_depths_B = np.asarray(counts_B.sum(axis=1)).flatten()
    
    # 4. Forge the Cross-Matrix Expected Value (\mu)
    # Multiply Set B's cell depths by Set A's gene frequencies
    expected_B = np.outer(cell_depths_B, p_j_A)
    
    # 5. The Thermodynamic Variance Formula
    theta = 100.0
    variance_B = expected_B + (np.square(expected_B) / theta)
    
    # 6. Calculate the Final Residuals
    # Convert Set B's raw counts to a dense array for the subtraction
    dense_counts_B = counts_B.toarray() if hasattr(counts_B, 'toarray') else np.asarray(counts_B)
    residuals_B = (dense_counts_B - expected_B) / np.sqrt(variance_B)
    
    # 7. The Scanpy Hard-Limit (Clipping)
    # Scanpy strictly clips residuals at sqrt(N_cells) to prevent extreme outliers
    clip_threshold = np.sqrt(adata_training.n_obs)
    residuals_B = np.clip(residuals_B, -clip_threshold, clip_threshold)
    
    # 8. Lock the forged matrix back into Set B
    adata_project.X = residuals_B
    
    # =========================================================================
    # PROJECTION CONTINUES
    # =========================================================================

    
    
    sc.tl.ingest(adata = adata_project,adata_ref = adata_training,obs=leiden_key_training,embedding_method=['pca'],
                 labeling_method='knn',neighbors_key=neighbors_key_training,inplace=True)
    

    try:
        pca_train_x = adata_training.obsm['X_pca'][:, 0]
        pca_train_y = adata_training.obsm['X_pca'][:, 1]
        
        pca_proj_x = adata_project.obsm['X_pca'][:, 0]
        pca_proj_y = adata_project.obsm['X_pca'][:, 1]
    except KeyError:
        print("'X_pca' not found in one of the matrices. Projection failed or file is raw.")
        return
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.scatter(
        pca_train_x, pca_train_y, 
        c='lightgray', 
        s=30, 
        alpha=0.8, 
        edgecolors='none', 
        label=f'Set A (Reference, N={adata_training.n_obs})'
    )
    ax.scatter(
        pca_proj_x, pca_proj_y, 
        c='#0033a0', # Deep structural blue
        s=15, 
        alpha=0.6, 
        edgecolors='none', 
        label=f'Set B (Projected, N={adata_project.n_obs})'
    )
    ax.set_xlabel('Principal Component 1 (Maximum Variance)', fontweight='bold')
    ax.set_ylabel('Principal Component 2 (Orthogonal Variance)', fontweight='bold')
    ax.set_title(f'{leiden_key_training}: Thermodynamic Overlap Diagnostic', fontsize=14, fontweight='bold')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
    legend = ax.legend(loc='best', frameon=True, shadow=True)
    for handle in legend.legend_handles:
        handle.set_alpha(1.0)
    plt.tight_layout()
    plt.savefig(f'{leiden_key_training}_Thermodynamic Overlap Diagnostic.png')
    adata_project.write_h5ad(adata_project_file_path)
    del adata_project,adata_training
    import gc
    gc.collect()

def orchestrator_A(h5ad_path,save_folder_path):
    '''
    To orchestrate the above functions For A side
    '''
    file_path_dictionary = random_split_data(h5ad_path,save_folder_path)
    training_file_path= file_path_dictionary['training_file']
    npr_hvg_pca_recal(training_file_path,'training_file')
    macro_leiden_key,macro_neighbors_key= knn_umap_leiden(training_file_path,n_neighbors=15,n_pcs=10,leiden_res=0.05,key_name='macro')
    ref_res_macro = stability_audit(training_file_path,macro_leiden_key)
    micro_filepaths_dictionary = divide_and_save_dataset_based_on_macro_or_micro_clusters(training_file_path,macro_leiden_key)
    for keys, filepath in micro_filepaths_dictionary.items():
        npr_hvg_pca_recal(filepath,keys)
    
    micro_leiden_key_dictionary= {}
    micro_neighbors_key_dictionary = {}
    
    for keys, micro_filepath in micro_filepaths_dictionary.items():
        ref_res = stability_audit(micro_filepath,keys)
        micro_leiden_key,micro_neighbors_key= knn_umap_leiden(micro_filepath,n_neighbors=10,n_pcs=15,leiden_res=ref_res,key_name=f'{keys}_micro')
        if micro_leiden_key is not None:
            micro_leiden_key_dictionary[keys] = micro_leiden_key
            micro_neighbors_key_dictionary[keys] = micro_neighbors_key
            
        else:
            print(f"Island '{keys}' is Terminal State (N<= 250). Bypassing Micro forging")

    Dictionary_of_returns_from_orch_A_to_be_used_in_orch_B = {
        'main_pca_artifact_path' : h5ad_path,
        'writing_files_folder_path': save_folder_path,
        'file_path_dictionary_from_the_split_step': file_path_dictionary,
        'training_macro_leiden_key' : macro_leiden_key,
        'training_macro_neighbors_key' : macro_neighbors_key,
        
        'training_micro_file_path_dictionary': micro_filepaths_dictionary,
        'training_micro_leiden_key_dictionary' : micro_leiden_key_dictionary,
        'training_micro_neighbors_key_dictionary' : micro_neighbors_key_dictionary
                                
    }
    for names,returns in Dictionary_of_returns_from_orch_A_to_be_used_in_orch_B.items():
        print(f'please use the {names} : {returns} to access the map from the training datasets')
    return Dictionary_of_returns_from_orch_A_to_be_used_in_orch_B
    
def orchestrator_B(Dictionary_of_returns_from_orch_A_to_be_used_in_orch_B):
    '''
    To orhcestrate the above functions For B side
    '''
    
    macro_adata_project_file_path = Dictionary_of_returns_from_orch_A_to_be_used_in_orch_B['file_path_dictionary_from_the_split_step']['projectable_file']
    macro_adata_training_file_path = Dictionary_of_returns_from_orch_A_to_be_used_in_orch_B['file_path_dictionary_from_the_split_step']['training_file']
    macro_neighbors_key_training = Dictionary_of_returns_from_orch_A_to_be_used_in_orch_B['training_macro_neighbors_key']
    macro_leiden_key_training = Dictionary_of_returns_from_orch_A_to_be_used_in_orch_B['training_macro_leiden_key']
    training_micro_leiden_key_dictionary = Dictionary_of_returns_from_orch_A_to_be_used_in_orch_B['training_micro_leiden_key_dictionary']
    training_micro_neighbors_key_dictionary = Dictionary_of_returns_from_orch_A_to_be_used_in_orch_B['training_micro_neighbors_key_dictionary']
    
    cast_projectable_data_on_training_data(macro_adata_project_file_path,macro_adata_training_file_path,macro_neighbors_key_training,macro_leiden_key_training)
    
    projected_micro_file_path_dictionary = divide_and_save_dataset_based_on_macro_or_micro_clusters(macro_adata_project_file_path,macro_leiden_key_training)
    training_micro_file_path_dictionary = Dictionary_of_returns_from_orch_A_to_be_used_in_orch_B['training_micro_file_path_dictionary']
    project_micro_leiden_key_dictionary= {}
    project_micro_neighbors_key_dictionary = {}
    for keys_projected, projected_micro_filepath in projected_micro_file_path_dictionary.items():
        
        root_key = keys_projected.replace('_Terminal_State','')
        target_key_training = None
        if root_key in training_micro_file_path_dictionary:
            target_key_training = root_key
        elif f'{root_key}_Terminal_State' in training_micro_file_path_dictionary:
            target_key_training = f'{root_key}_Terminal_State'
        if target_key_training is not None:
            training_micro_filepath = training_micro_file_path_dictionary[target_key_training]
            if 'Terminal_State' in target_key_training:
                print(f'Training Set declared {root_key} a Terminal State. Bypassing micro-cast for projected Set')
                continue
            projected_micro_leiden_key = training_micro_leiden_key_dictionary[target_key_training]
            projected_micro_neighbors_key = training_micro_neighbors_key_dictionary[target_key_training]
        
            cast_projectable_data_on_training_data(projected_micro_filepath,training_micro_filepath,projected_micro_neighbors_key,projected_micro_leiden_key)
            project_micro_leiden_key_dictionary[keys_projected]= projected_micro_leiden_key
            project_micro_neighbors_key_dictionary[keys_projected]=projected_micro_neighbors_key
        else:
            print(f'Projected Set contains alien biology {keys_projected} not found in training set. Discarding projection')
    Dictionary_of_returns_from_orch_B = {
        
        'macro_adata_project_file_path'             :macro_adata_project_file_path,
        'macro_neighbors_key_training'              :macro_neighbors_key_training,
        'macro_leiden_key_training'                 :macro_leiden_key_training,
        'projected_micro_file_path_dictionary'      :projected_micro_file_path_dictionary,
        'projected_micro_leiden_key_dictionary'     :project_micro_leiden_key_dictionary,
        'projected_micro_neighbors_key_dictionary'  :project_micro_neighbors_key_dictionary                        
    }
    for names,returns in Dictionary_of_returns_from_orch_B.items():
        print(f'please use the {names} : {returns} to access the map from the training datasets')
    return Dictionary_of_returns_from_orch_B

if __name__ == '__main__':
    h5ad_path = '/Users/qgem/GitHub/PBMC3k-reproducible/data/objects/pbmc3k_qc.h5ad'
    save_folder_path = '/Users/qgem/GitHub/PBMC3k-reproducible/notebooks/results'
    Dictionary_of_returns_from_orch_A_to_be_used_in_orch_B = orchestrator_A(h5ad_path,save_folder_path)
    Dictionary_of_returns_from_orch_B = orchestrator_B(Dictionary_of_returns_from_orch_A_to_be_used_in_orch_B)
    with open(f'{save_folder_path}/Dictionary_of_returns_from_orch_B.json','w') as json_file:
        json.dump(Dictionary_of_returns_from_orch_B,json_file,indent=4)
