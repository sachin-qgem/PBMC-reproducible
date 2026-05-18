"""
Streamlit Application Interface

This module provides a graphical interface for data pipeline Pipeline Execution,
parameter selection, and biological annotation mapping.
"""
import json
import os
import re
import os.path as op
import shutil
import base64
from pathlib import Path
from typing import Dict, Any, Optional

import anndata as ad
import pandas as pd
import scanpy as sc
import streamlit as st

st.set_page_config(page_title="PBMC Analysis Interface", layout="wide")

from src.pbmc3k_pipeline import P03_qc_filtering, P04_clustering, P05_top_markers, P06_annotation

DICT_B_PATH = "./data/objects/Dictionary_of_returns_from_orch_B.json"
ANNOTATION_PATH = "./data/objects/annotation_manual_empty.json"
ONTOLOGY_PATH = "./data/objects/ontology_cl_id_manual_empty.json"

def initialize_session_state() -> None:
    """
    State variables are initialized for UI interaction.
    """
    if "annotations" not in st.session_state:
        st.session_state.annotations = {}
    if "ontologies" not in st.session_state:
        st.session_state.ontologies = {}
    if "phase2_grid_swept" not in st.session_state:
        st.session_state.phase2_grid_swept = False
    if "phase2_complete" not in st.session_state:
        st.session_state.phase2_complete = False
    if "temp_jaccard_scores" not in st.session_state:
        st.session_state.temp_jaccard_scores = None
    if "phase2_macro_swept" not in st.session_state:
        st.session_state.phase2_macro_swept = False
    if "phase2_macro_locked" not in st.session_state:
        st.session_state.phase2_macro_locked = False
    if "micro_queue" not in st.session_state:
        st.session_state.micro_queue = []
    if "current_micro_key" not in st.session_state:
        st.session_state.current_micro_key = None
    if "current_micro_swept" not in st.session_state:
        st.session_state.current_micro_swept = False
    if "final_micro_leiden_dict" not in st.session_state:
        st.session_state.final_micro_leiden_dict = {}
    if "final_micro_neighbors_dict" not in st.session_state:
        st.session_state.final_micro_neighbors_dict = {}
    if "phase2_complete" not in st.session_state:
        st.session_state.phase2_complete = False

@st.cache_resource(show_spinner="Loading matrix into memory...")
def load_anndata(filepath: str) -> Optional[ad.AnnData]:
    """
    Caches array data.
    """
    if op.exists(filepath):
        return sc.read_h5ad(filepath)
    return None

def load_json(filepath: str) -> Dict[str, Any]:
    """
    Loads a JSON file to a dictionary.
    """
    if op.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_json(filepath: str, data: Dict[str, Any]) -> None:
    """
    Writes dictionary data to a JSON file.
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

def render_plots(sub_dir: str, title: str) -> None:
    """
    Parses visual outputs for interface display.
    """
    target_dir = op.join("./results/figures/", sub_dir)
    st.markdown(f"#### {title}")
    
    if not op.exists(target_dir):
        st.info(f"Directory not found: {target_dir}")
        return
        
    valid_extensions = {".png", ".svg"}
    image_files = [p for p in Path(target_dir).rglob("*") if p.suffix.lower() in valid_extensions]
    
    if not image_files:
        st.info(f"No plot files found in {sub_dir}.")
        return
        
    cols = st.columns(2)
    for idx, img_path in enumerate(image_files):
        with cols[idx % 2]:
            if img_path.suffix.lower() == ".svg":
                with open(img_path, "r", encoding="utf-8") as f:
                    svg_code = f.read()
                svg_code = re.sub(r'width="[^"]+"', 'width="100%"', svg_code, count=1)
                svg_code = re.sub(r'height="[^"]+"', 'height="auto"', svg_code, count=1)
                b64_svg = base64.b64encode(svg_code.encode("utf-8")).decode("utf-8")
                img_src = f"data:image/svg+xml;base64,{b64_svg}"
                css_container = "display: flex; justify-content: center; align-items: center; overflow: hidden; border: 1px solid rgba(128,128,128,0.2); border-radius: 5px; padding: 10px; margin-bottom: 10px; background-color: white;"
                styled_svg = f"<div style='{css_container}'><img src='{img_src}' style='width: 100%; height: auto;' alt='{img_path.name}'></div>"
                st.markdown(styled_svg, unsafe_allow_html=True)
                st.caption(f" {img_path.name}")
            else:
                st.image(str(img_path), caption=img_path.name, use_container_width=True)


def main() -> None:
    
    st.title(" PBMC Analysis Interface")
    
    initialize_session_state()
    
    master_map = load_json(DICT_B_PATH)
    
    if master_map:
        if not st.session_state.annotations:
            st.session_state.annotations = load_json(ANNOTATION_PATH)
        if not st.session_state.ontologies:
            st.session_state.ontologies = load_json(ONTOLOGY_PATH)

    st.sidebar.header("Data Selection")
    active_path, active_leiden, active_label_key = None, None, None
    
    if master_map:
        macro_key = master_map.get('macro_leiden_key_training')
        micro_paths = master_map.get('projected_micro_file_path_dictionary', {})
        micro_leiden_dict = master_map.get('projected_micro_leiden_key_dictionary', {})
        
        view_mode = st.sidebar.radio("Clustering Tier", ["Primary Clustering", "Sub-Clustering"])
    
        if view_mode == "Primary Clustering":
            active_path = master_map.get('macro_adata_project_file_path')
            active_leiden = macro_key
            active_label_key = macro_key
        else:
            selected_micro = st.sidebar.selectbox("Select Sub-Cluster Matrix", list(micro_paths.keys()))
            if selected_micro:
                active_path = micro_paths[selected_micro]
                active_leiden = micro_leiden_dict.get(selected_micro)
                active_label_key = active_leiden
                
                if active_leiden is None:
                    st.sidebar.warning("Isotropic Variance State Detected. Inheriting parent key.")
                    clean_key = selected_micro.replace('_Terminal_State', '')
                    parts = clean_key.split('_')
                    active_label_key = '_'.join(parts[:-1])
    else:
        st.sidebar.info("Workspace empty. Awaiting file upload.")

    tab_pipeline,tab_annotate, tab_plots = st.tabs(["Pipeline Execution", "Annotation Mapping", "Data Visualization"])


    with tab_pipeline:
        
        if st.session_state.get("purge_success", False):
            st.success("Workspace directory cleared.")
            st.session_state.purge_success = False
        
        st.markdown("### 1. File Management")
        st.write("Remove existing matrices and figures to reset the workspace directory.")
        
        if st.button("Clear Workspace", type="primary"):

            directories_to_clean = [
                "data/raw/pbmc3k_filtered_gene_bc_matrices/hg19", 
                "data/objects", 
                "results/figures",
                "cache"
            ]
            for directory in directories_to_clean:
                if os.path.exists(directory):
                    shutil.rmtree(directory)
            
            directories_to_rebuild = [
                "data/raw/pbmc3k_filtered_gene_bc_matrices/hg19", 
                "data/objects", 
                "results/figures/p03_qc_filtering",
                "results/figures/p04_clustering",
                "results/figures/p05_top_markers",
                "results/figures/p06_annotation"
            ]
            for directory in directories_to_rebuild:
                os.makedirs(directory, exist_ok=True)
            
            st.cache_resource.clear()
            st.cache_data.clear()
            
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.session_state.purge_success = True    
            st.rerun()
            
        st.divider()

        st.markdown("### 2. Processing Modules")
        st.write("Upload the 3 standard 10X Genomics files: `matrix.mtx`, `barcodes.tsv`, `genes.tsv`")
        
        uploaded_files = st.file_uploader("Drop 10X files here", accept_multiple_files=True)
        
        if uploaded_files:
            if st.button("Save Files to Directory"):
                target_dir = "data/raw/pbmc3k_filtered_gene_bc_matrices/hg19"
                os.makedirs(target_dir, exist_ok=True)
                for f in uploaded_files:
                    file_path = os.path.join(target_dir, f.name)
                    with open(file_path, "wb") as disk_file:
                        disk_file.write(f.getbuffer())
                st.success(f"{len(uploaded_files)} files saved to {target_dir}")

        st.divider()

        st.markdown("### 3. The Pipeline Execution")
        st.write("Note: Matrix operations require atleast 16GB of available system memory.")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Execute Filtering (P03)"):
                target_dir = "data/raw/pbmc3k_filtered_gene_bc_matrices/hg19"
                required_files = {
                    "matrix": {"matrix.mtx", "matrix.mtx.gz"},
                    "barcodes": {"barcodes.tsv", "barcodes.tsv.gz"},
                    "genes": {"genes.tsv", "genes.tsv.gz", "features.tsv", "features.tsv.gz"}
                }
                if not os.path.exists(target_dir):
                    st.error("Error: The workspace directory is empty. Upload input files to proceed.")
                else:
                    actual_files = set(os.listdir(target_dir))
                    missing_files = []
                    for concept, acceptable_variants in required_files.items():
                        if not acceptable_variants.intersection(actual_files):
                            missing_files.append(concept)
                    if missing_files:
                        st.error(f"Error: The following required files are missing from the directory: {missing_files}.")
                    else:
                        with st.spinner("Processing matrix array..."):
                            try:
                                P03_qc_filtering.main()
                                st.success("Filtering complete.")
                            except Exception as e:
                                st.error(f"Phase I Failed: {e}")

        with col2:
            if not st.session_state.phase2_macro_swept:
                if st.button("Execute Primary Clustering (P04)"):
                    if not os.path.exists("data/objects/pbmc3k_qc.h5ad"):
                        st.error("Error: Phase I output matrix not found.")
                    else:
                        with st.spinner("Computing initial neighborhood graph and Leiden clustering..."):
                            sweep_state = P04_clustering.execute_macro_sweep(
                                h5ad_path="data/objects/pbmc3k_qc.h5ad", 
                                save_folder_path="data/objects"
                            )
                            st.session_state.p04_training_file_path = sweep_state['training_file_path']
                            st.session_state.p04_file_path_dict = sweep_state['file_path_dict']
                            st.session_state.p04_suggested_k = sweep_state['suggested_k']
                            st.session_state.p04_suggested_r = sweep_state['suggested_r']
                            st.session_state.phase2_macro_swept = True
                            st.rerun()
            else:
                if st.session_state.phase2_complete:
                    st.success("Phase II complete: Clustering parameters established.")
                else:
                    st.warning("Execution paused: Awaiting manual parameter confirmation.")
        with col3:
            if st.button("Execute Marker Evaluation (P05)"):
                if not os.path.exists("data/objects/Dictionary_of_returns_from_orch_A.json"):
                    st.error("Error: Phase II parameter output not found.")
                else:
                    with st.spinner("Processing matrix array..."):
                        try:
                            P05_top_markers.main()
                            st.success("Marker evaluation complete.")
                        except Exception as e:
                            st.error(f"Phase III Failed: {e}")

        if st.session_state.phase2_macro_swept and not st.session_state.phase2_macro_locked:
            st.divider()
            st.markdown("### Manual Parameter Selection: Primary Clustering")
            st.write("Review the suggested parameters and evaluate stability before confirming.")

            svg_file = "./results/figures/p04_clustering/macro_grid_search_surface.svg"
            if os.path.exists(svg_file):
                with open(svg_file, "r", encoding="utf-8") as f:
                    svg_code = f.read()
                svg_code = re.sub(r'width="[^"]+"', 'width="100%"', svg_code, count=1)
                svg_code = re.sub(r'height="[^"]+"', 'height="auto"', svg_code, count=1)
                b64_svg = base64.b64encode(svg_code.encode("utf-8")).decode("utf-8")
                img_src = f"data:image/svg+xml;base64,{b64_svg}"
                st.markdown(f"<div style='border: 1px solid #444; padding: 10px; background: white;'><img src='{img_src}' style='width: 100%;'></div>", unsafe_allow_html=True)

            
            c1, c2 = st.columns(2)
            with c1:
                chosen_k = st.number_input("Optimal k (Neighbors)", value=int(st.session_state.p04_suggested_k), min_value=5, max_value=200, step=5, key="macro_k")
            with c2:
                chosen_r = st.number_input("Optimal r (Resolution)", value=float(st.session_state.p04_suggested_r), min_value=0.01, max_value=3.0, step=0.01, key="macro_r")

            col_test, col_lock = st.columns(2)
            
            
            with col_test:
                if st.button("Evaluate Parameters", type="secondary", key="test_macro"):
                    with st.spinner(f"Evaluating Jaccard stability at k={chosen_k}, r={chosen_r}..."):
                        st.session_state.temp_jaccard_scores = P04_clustering.test_jaccard_stability(
                            st.session_state.p04_training_file_path, chosen_k, chosen_r
                        )
                        st.rerun()

            
            if st.session_state.temp_jaccard_scores:
                st.markdown("#### Jaccard Stability")
                for cluster_id, score in st.session_state.temp_jaccard_scores.items():
                    if score >= 0.85:
                        st.success(f"Cluster {cluster_id}: {score:.3f} [HIGH STABILITY]")
                    elif score >= 0.60:
                        st.warning(f"Cluster {cluster_id}: {score:.3f} [MODERATE STABILITY]")
                    else:
                        st.error(f"Cluster {cluster_id}: {score:.3f} [LOW STABILITY]")
                st.divider()

            
            with col_lock:
                if st.button("Confirm Parameters", type="primary", key="lock_macro"):
                    with st.spinner("Applying parameters and subsetting matrices..."):
                        st.session_state.temp_jaccard_scores = None 
                        
                        macro_state = P04_clustering.lock_macro_and_extract_micro_queue(
                            st.session_state.p04_training_file_path, chosen_k, chosen_r, './data/regev_lab_cell_cycle_genes.txt'
                        )
                        st.session_state.macro_leiden_key = macro_state['macro_leiden_key']
                        st.session_state.macro_neighbors_key = macro_state['macro_neighbors_key']
                        st.session_state.micro_filepaths_dict = macro_state['micro_filepaths_dict']
                        
                        st.session_state.micro_queue = [k for k in macro_state['micro_filepaths_dict'].keys() if 'Terminal_State' not in k]
                        st.session_state.phase2_macro_locked = True
                        st.rerun()

        
        if st.session_state.phase2_macro_locked and not st.session_state.phase2_complete:
            st.divider()
            
            
            if len(st.session_state.micro_queue) > 0:
                current_micro = st.session_state.micro_queue[0]
                filepath = st.session_state.micro_filepaths_dict[current_micro]
                
                st.markdown(f"### Manual Parameter Selection: Sub-Cluster `{current_micro}`]")
                st.info(f"{len(st.session_state.micro_queue)} Sub-Clusters remaining in queue.")
                
                if not st.session_state.current_micro_swept:
                    with st.spinner(f"Evaluating parameters for {current_micro}..."):
                        micro_sweep = P04_clustering.execute_micro_sweep(filepath, current_micro, "./results/figures/p04_clustering")
                        st.session_state.current_micro_k = micro_sweep['suggested_k']
                        st.session_state.current_micro_r = micro_sweep['suggested_r']
                        st.session_state.current_micro_swept = True
                        st.rerun()
                
                if st.session_state.current_micro_swept:
                    svg_file = f"./results/figures/p04_clustering/{current_micro}_grid_search_surface.svg"
                    if os.path.exists(svg_file):
                        with open(svg_file, "r", encoding="utf-8") as f:
                            svg_code = f.read()
                        svg_code = re.sub(r'width="[^"]+"', 'width="100%"', svg_code, count=1)
                        svg_code = re.sub(r'height="[^"]+"', 'height="auto"', svg_code, count=1)
                        b64_svg = base64.b64encode(svg_code.encode("utf-8")).decode("utf-8")
                        img_src = f"data:image/svg+xml;base64,{b64_svg}"
                        st.markdown(f"<div style='border: 1px solid #444; padding: 10px; background: white;'><img src='{img_src}' style='width: 100%;'></div>", unsafe_allow_html=True)
                    
                    c1, c2 = st.columns(2)
                    with c1:
                        chosen_k = st.number_input("Optimal k (Neighbors)", value=int(st.session_state.current_micro_k), min_value=5, max_value=200, step=5, key=f"k_{current_micro}")
                    with c2:
                        chosen_r = st.number_input("Optimal r (Resolution)", value=float(st.session_state.current_micro_r), min_value=0.01, max_value=3.0, step=0.01, key=f"r_{current_micro}")

                    col_test, col_lock = st.columns(2)
                    
                    with col_test:
                        if st.button("Evaluate Jaccard stability", type="secondary", key=f"test_{current_micro}"):
                            with st.spinner(f"Evaluating Jaccard stability for {current_micro} at k={chosen_k}, r={chosen_r}..."):
                                st.session_state.temp_jaccard_scores = P04_clustering.test_jaccard_stability(
                                    filepath, chosen_k, chosen_r
                                )
                                st.rerun()

                    if st.session_state.temp_jaccard_scores:
                        st.markdown(f"#### Jaccard Stability: {current_micro}")
                        for cluster_id, score in st.session_state.temp_jaccard_scores.items():
                            if score >= 0.85:
                                st.success(f"Cluster {cluster_id}: {score:.3f} [HIGH STABILITY]")
                            elif score >= 0.60:
                                st.warning(f"Cluster {cluster_id}: {score:.3f} [MODERATE STABILITY]")
                            else:
                                st.error(f"Cluster {cluster_id}: {score:.3f} [LOW STABILITY]")
                        st.divider()

                    with col_lock:
                        if st.button(f"Confirm Parameters:", type="primary", key=f"lock_{current_micro}"):
                            with st.spinner(f"Saving parameters for {current_micro}..."):
                                st.session_state.temp_jaccard_scores = None 
                                
                                micro_result = P04_clustering.lock_micro_state(
                                    filepath, current_micro, chosen_k, chosen_r, './data/regev_lab_cell_cycle_genes.txt'
                                )
                                
                                if micro_result['m_leiden']:
                                    st.session_state.final_micro_leiden_dict[current_micro] = micro_result['m_leiden']
                                    st.session_state.final_micro_neighbors_dict[current_micro] = micro_result['m_neighbors']
                                
                                st.session_state.micro_queue.pop(0)
                                st.session_state.current_micro_swept = False
                                st.rerun()
            
            else:
                st.success("### Sub-Clustering Complete")
                st.write("All matrix partitions have been successfully processed.")
                if st.button("Finalize Sub-Clustering", type="primary"):
                    with st.spinner("Exporting metadata dictionaries and finalizing Phase II..."):
                        P04_clustering.seal_phase_II_pipeline(
                            h5ad_path="data/objects/pbmc3k_qc.h5ad",
                            save_folder_path="data/objects",
                            file_path_dict=st.session_state.p04_file_path_dict,
                            macro_leiden_key=st.session_state.macro_leiden_key,
                            macro_neighbors_key=st.session_state.macro_neighbors_key,
                            micro_filepaths_dict=st.session_state.micro_filepaths_dict,
                            micro_leiden_dict=st.session_state.final_micro_leiden_dict,
                            micro_neighbors_dict=st.session_state.final_micro_neighbors_dict
                        )
                        st.session_state.phase2_complete = True
                        st.rerun()
                    
        st.divider()


    pipeline_state_file = "data/objects/Dictionary_of_returns_from_orch_B.json"
    with tab_plots:
        if not master_map:
            st.info("Visualizations are not available until the matrix processing steps are complete.")
        else:
            st.header("Visual Outputs")
            st.markdown("Select an output Category to show the visual plots")
 
            SECTOR_MAP = {
                "Filtering": "p03_qc_filtering",
                "Clustering": "p04_clustering",
                "Markers": "p05_top_markers"
            }

            selection = st.selectbox(
                "Select Output Category", 
                options=list(SECTOR_MAP.keys()),
                index=0,
                key="telemetry_sector_selector" 
            )

            target_sub_dir = SECTOR_MAP[selection]
            
            st.caption(f"Scanning Physical Path: `./results/figures/{target_sub_dir}/`")
            st.divider()
            with st.container():
                render_plots(target_sub_dir, selection)
            
    with tab_annotate:
        if not master_map:
            st.info("Annotation mapping unavailable: Waiting for clustering output. Execute the processing pipeline to enable this module.")
        else:
            st.markdown("### Manual Annotation and Label Mapping")
            if active_path and active_label_key:
                st.subheader(f"Active Cluster Key: `{active_label_key}`")
                adata = load_anndata(active_path)
                if adata is not None:
                    if 'final_top_genes_per_cluster' in adata.uns:
                        df_markers = adata.uns['final_top_genes_per_cluster']
                        st.markdown("**Top Extracted Genes Markers (Wilcoxon Rank-Sum)**")
                        st.dataframe(
                            df_markers[['group', 'names', 'pvals_adj', 'logfoldchanges', 'expression_delta']], 
                            use_container_width=True
                        )
                    else:
                        if "Terminal_State" in active_path:
                            st.info(
                                "**Terminal State Confirmed.**\n"
                                "This sub-cluster is statistically homogeneous. Differential expression testing "
                                "requires at least two groups to calculate variance. "
                                "Refer to the parent clustering level to view marker genes for this lineage."
                            )
                        else:
                            st.warning("Marker dictionary missing from the active matrix. Execute Phase III `P05_top_markers.py`.")

                    st.divider()
                    st.markdown("### Annotation and Ontology Mapping")
                    
                    cluster_col = active_leiden if active_leiden else active_label_key
                    
                    if cluster_col in adata.obs.columns:
                        clusters = sorted(
                            adata.obs[cluster_col].dropna().unique().tolist(), 
                            key=lambda x: int(x) if str(x).isdigit() else x
                        )
                        
                        if active_label_key not in st.session_state.annotations:
                            st.session_state.annotations[active_label_key] = {str(c): "" for c in clusters}
                        if active_label_key not in st.session_state.ontologies:
                            st.session_state.ontologies[active_label_key] = {str(c): "" for c in clusters}
                            
                        df_state = []
                        for c in clusters:
                            c_str = str(c)
                            current_label = st.session_state.annotations[active_label_key].get(c_str, "")
                            current_cl = st.session_state.ontologies[active_label_key].get(c_str, "")
                            
                            df_state.append({
                                "Cluster ID": c_str,
                                "Biological Identity": current_label,
                                "Cell Ontology (CL) ID": current_cl
                            })
                            
                        df_ui = pd.DataFrame(df_state)
                        
                        st.markdown("Double-click a cell to edit. Press **Enter** to confirm the value.")
                        edited_df = st.data_editor(
                            df_ui, 
                            use_container_width=True, 
                            hide_index=True,
                            disabled=["Cluster ID"]
                        )

                        if st.button("Save Annotation Dictionaries", type="primary"):
                            for _, row in edited_df.iterrows():
                                c_id = row["Cluster ID"]
                                label = row["Biological Identity"]
                                cl_id = row["Cell Ontology (CL) ID"]
                                
                                st.session_state.annotations[active_label_key][c_id] = label
                                st.session_state.ontologies[active_label_key][c_id] = cl_id
                                    
                            save_json(ANNOTATION_PATH, st.session_state.annotations)
                            save_json(ONTOLOGY_PATH, st.session_state.ontologies)
                            
                            st.success("Annotation dictionaries saved to disk. Matrix is ready for metadata integration")

                        st.divider()
                        st.markdown("### Phase IV: Matrix Metadata Integration")
                        st.markdown("Execute this step only after all annotation dictionaries have been populated and saved.")
                        
                        if st.button("Execute Metadata Integration (P06)", type="secondary"):
                            with st.spinner("Integrating annotations and combining global matrix..."):
                                try:
                                    
                                    P06_annotation.main()
                                    st.success("Matrix metadata integration complete.")
                                        
                                    ml_ready_path = "./data/objects/pbmc3k_qc_ML_Ready.h5ad"
                                    if op.exists(ml_ready_path):
                                        with open(ml_ready_path, "rb") as file:
                                            st.download_button(
                                                label="Download ML-Ready Matrix",
                                                data=file,
                                                file_name="pbmc3k_ML_Ready.h5ad",
                                                mime="application/octet-stream"
                                            )
                                except Exception as e:
                                    st.error(f"Execution error encountered during integration:")
                                    st.code(str(e))
                    else:
                        st.error(f"Required observation column '{cluster_col}' missing from matrix.")
                else:
                    st.error(f"Failed to load matrix at path: {active_path}")

            st.markdown("---")
            st.markdown("### Session Memory Reset")
            st.markdown("Execute this to clear cached annotations or clustering states from previous sessions.") 
            if st.button("Reset Session Memory", type="primary"):
                keys_to_destroy = [
                    key for key in st.session_state.keys() 
                    if "annotation" in key.lower() or "ontolog" in key.lower() or "dict" in key.lower()
                ]
                for key in keys_to_destroy:
                    del st.session_state[key]
                st.cache_data.clear()
                st.cache_resource.clear()
                import gc
                gc.collect()
                st.rerun()       
if __name__ == "__main__":
    main()