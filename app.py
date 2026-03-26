"""
Phase IV: Comprehensive Orchestration & UI Bridge (Streamlit Engine)

This module operates as the master control interface for human-in-the-loop 
topological mapping. It utilizes cached memory to pin massive .h5ad tensors 
in RAM, constructs an interactive 2D matrix for biological identity injection, 
and features a Visual Telemetry scanner to project physical pipeline artifacts 
(PNGs) into the browser. Finally, it commands a Subprocess engine to execute 
the Phase IV Recombination script entirely in the background.
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

# =============================================================================
# ABSOLUTE LAW: Page config must be the very first Streamlit command executed
# =============================================================================
st.set_page_config(page_title="PBMC Biological Observatory", layout="wide")
# =============================================================================
# THE WORMHOLE: Importing the Packaged Pipeline
# =============================================================================

from src.pbmc3k_pipeline import P03_qc_filtering, P04_clustering, P05_top_markers, P06_annotation

# =============================================================================
# GLOBAL THERMODYNAMIC CONSTANTS & UI CONFIGURATION
# =============================================================================


# Paths mathematically aligned to P04, P05, and P06 outputs
DICT_B_PATH = "./data/objects/Dictionary_of_returns_from_orch_B.json"
ANNOTATION_PATH = "./data/objects/annotation_manual_empty.json"
ONTOLOGY_PATH = "./data/objects/ontology_cl_id_manual_empty.json"

# =============================================================================
# MEMORY I/O ENGINES
# =============================================================================

def initialize_session_vault() -> None:
    """
    Initializes the protected RAM vault to survive Streamlit's reactive loop.
    
    Variables stored in `st.session_state` persist across user interactions,
    preventing the erasure of human inputs when the script re-runs.
    """
    if "annotations" not in st.session_state:
        st.session_state.annotations = {}
    if "ontologies" not in st.session_state:
        st.session_state.ontologies = {}
    if "phase2_grid_swept" not in st.session_state:
        st.session_state.phase2_grid_swept = False
    if "phase2_complete" not in st.session_state:
        st.session_state.phase2_complete = False
    # --- PHASE II: ITERATIVE STATE MACHINE ---
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

@st.cache_resource(show_spinner="Loading Heavy Tensor into RAM...")
def load_tensor(filepath: str) -> Optional[ad.AnnData]:
    """
    Loads an AnnData tensor into cached memory outside the Streamlit loop.

    By decorating this function with `@st.cache_resource`, we physically pin 
    the matrix in RAM. This prevents the engine from continuously querying the 
    disk and reloading the massive .h5ad file on every UI interaction.

    Parameters
    ----------
    filepath : str
        The absolute or relative path to the physical .h5ad matrix.

    Returns
    -------
    Optional[ad.AnnData]
        The loaded tensor object, or None if the physical file does not exist.
    """
    if op.exists(filepath):
        return sc.read_h5ad(filepath)
    return None

def load_json_ledger(filepath: str) -> Dict[str, Any]:
    """
    Safely ingests a JSON ledger from the physical disk into Python dictionaries.
    """
    if op.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_json_ledger(filepath: str, data: Dict[str, Any]) -> None:
    """
    Atomically writes the mapped dictionary to disk, forging directories if required.
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

# =============================================================================
# VISUAL TELEMETRY SCANNER
# =============================================================================

def render_visual_telemetry(sub_dir: str, title: str) -> None:
    """
    Scans the physical disk for both PNG and SVG artifacts and renders them
    using responsive CSS containment to prevent overlapping.
    """
    target_dir = op.join("./results/figures/", sub_dir)
    st.markdown(f"#### {title}")
    
    if not op.exists(target_dir):
        st.info(f"Directory not found: {target_dir}")
        return
        
    valid_extensions = {".png", ".svg"}
    image_files = [p for p in Path(target_dir).rglob("*") if p.suffix.lower() in valid_extensions]
    
    if not image_files:
        st.info(f"No telemetry artifacts found in {sub_dir}.")
        return
        
    # Build a 2-column mathematical grid
    cols = st.columns(2)
    for idx, img_path in enumerate(image_files):
        with cols[idx % 2]:
            if img_path.suffix.lower() == ".svg":
                # 1. Read the raw SVG XML
                with open(img_path, "r", encoding="utf-8") as f:
                    svg_code = f.read()
                svg_code = re.sub(r'width="[^"]+"', 'width="100%"', svg_code, count=1)
                svg_code = re.sub(r'height="[^"]+"', 'height="auto"', svg_code, count=1)
                b64_svg = base64.b64encode(svg_code.encode("utf-8")).decode("utf-8")
                img_src = f"data:image/svg+xml;base64,{b64_svg}"
                css_container = "display: flex; justify-content: center; align-items: center; overflow: hidden; border: 1px solid rgba(128,128,128,0.2); border-radius: 5px; padding: 10px; margin-bottom: 10px; background-color: white;"
                styled_svg = f"<div style='{css_container}'><img src='{img_src}' style='width: 100%; height: auto;' alt='{img_path.name}'></div>"
                st.markdown(styled_svg, unsafe_allow_html=True)
                st.caption(f"📍 {img_path.name}")
            else:
                # Standard fallback for standard PNGs
                st.image(str(img_path), caption=img_path.name, use_container_width=True)

# =============================================================================
# MAIN ORCHESTRATION ENGINE
# =============================================================================

def main() -> None:
    """
    The primary execution loop for the Streamlit UI. Handles state initialization, 
    matrix selection routing, transient dataframe rendering, Subprocess execution,
    and the final fracture protocol to seal data to disk.
    """
    st.title("🧬 PBMC Human-in-the-Loop Pipeline")
    
    # 1. Establish the RAM Vault
    initialize_session_vault()
    
    # 2. Ingest the Master Map (The Orchestrator B Output)
    master_map = load_json_ledger(DICT_B_PATH)
    
        
    # Synchronize physical ledgers with RAM on initial load only
    if master_map:
        if not st.session_state.annotations:
            st.session_state.annotations = load_json_ledger(ANNOTATION_PATH)
        if not st.session_state.ontologies:
            st.session_state.ontologies = load_json_ledger(ONTOLOGY_PATH)

    # 3. Sidebar Navigation Architecture
    st.sidebar.header("Navigation")
    active_path = None
    active_leiden = None
    active_label_key = None
    if master_map:
        macro_key = master_map.get('macro_leiden_key_training')
        micro_paths = master_map.get('projected_micro_file_path_dictionary', {})
        micro_leiden_dict = master_map.get('projected_micro_leiden_key_dictionary', {})
        
        view_mode = st.sidebar.radio("Topology Level", ["Macro Level", "Micro Level"])
    
        if view_mode == "Macro Level":
            active_path = master_map.get('macro_adata_project_file_path')
            active_leiden = macro_key
            active_label_key = macro_key
        else:
            selected_micro = st.sidebar.selectbox("Select Micro Topology", list(micro_paths.keys()))
            if selected_micro:
                active_path = micro_paths[selected_micro]
                active_leiden = micro_leiden_dict.get(selected_micro)
                active_label_key = active_leiden
                
                # Absolute logical inheritance for Terminal States
                # Matches the exact string manipulation physics established in P06
                if active_leiden is None:
                    st.sidebar.warning("Terminal State Detected. Inheriting parent topology logic.")
                    clean_key = selected_micro.replace('_Terminal_State', '')
                    parts = clean_key.split('_')
                    active_label_key = '_'.join(parts[:-1])
    else:
        st.sidebar.info("Workspace sterile. Awaiting ingestion from Control Room.")

    # 4. The Tab Architecture
    tab_control_room,tab_annotate, tab_telemetry = st.tabs(["🎛️ Control Room (Execution)","🧬 Annotation Engine", "📊 Visual Telemetry"])

    # =========================================================================
    # TAB 1: THE CONTROL ROOM
    # =========================================================================
    with tab_control_room:
        
        if st.session_state.get("purge_success", False):
            st.success("Workspace purged. 5-Sigma sterile environment achieved.")
            st.session_state.purge_success = False
        
        st.markdown("### 1. The Entropy Purge")
        st.write("Wipe existing tensors and figures to prepare the physical container for new data ingestion.")
        
        if st.button("Execute 'make clean' / Purge Workspace", type="primary"):
            # 1. Annihilate the physical files on disk
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
            
            # 2. Annihilate the Streamlit RAM Cache (Flush the loaded .h5ad tensors)
            st.cache_resource.clear()
            st.cache_data.clear()
            
            # 3. Purge the Session State Dictionary (Eradicate ghost variables)
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.session_state.purge_success = True    
            # 4. Reboot the UI to reflect the sterile state
            st.rerun()
            
        st.divider()

        st.markdown("### 2. The Ingestion Protocol")
        st.write("Upload the 3 standard 10X Genomics files: `matrix.mtx`, `barcodes.tsv`, `genes.tsv`")
        
        uploaded_files = st.file_uploader("Drop 10X files here", accept_multiple_files=True)
        
        if uploaded_files:
            if st.button("Anchor Matter to Container Disk"):
                target_dir = "data/raw/pbmc3k_filtered_gene_bc_matrices/hg19"
                os.makedirs(target_dir, exist_ok=True)
                for f in uploaded_files:
                    file_path = os.path.join(target_dir, f.name)
                    with open(file_path, "wb") as disk_file:
                        disk_file.write(f.getbuffer())
                st.success(f"Successfully anchored {len(uploaded_files)} files to {target_dir}")

        st.divider()

        st.markdown("### 3. The Execution Engine")
        st.write("Trigger the analytical modules. WARNING: Observe the 16GB RAM boundary.")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Run Phase I (QC & Filter)"):
                target_dir = "data/raw/pbmc3k_filtered_gene_bc_matrices/hg19"
                required_files = ["matrix.mtx", "barcodes.tsv", "genes.tsv"]
                if not os.path.exists(target_dir):
                    st.error("⛔ **Execution Blocked:** The workspace is purged. Upload files first.")
                else:
                    missing_files = [f for f in required_files if not os.path.exists(os.path.join(target_dir, f))]
                    if missing_files:
                        st.error(f"⛔ **Execution Blocked:** Missing files: {missing_files}.")
                    else:
                        with st.spinner("Executing 5-MAD thermodynamic purge..."):
                            try:
                                P03_qc_filtering.main()
                                st.success("Phase I Complete: pbmc3k_qc.h5ad anchored.")
                            except Exception as e:
                                st.error(f"Phase I Failed: {e}")

        with col2:
            if not st.session_state.phase2_macro_swept:
                if st.button("Run Phase II (Start: Macro Sweep)"):
                    if not os.path.exists("data/objects/pbmc3k_qc.h5ad"):
                        st.error("⛔ **Execution Blocked:** Phase I output not found.")
                    else:
                        with st.spinner("Forging Macro Background Field..."):
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
                    st.success("Phase II Complete: All Topologies established.")
                else:
                    st.warning("Pipeline Active: See Human Overrides below.")
        with col3:
            if st.button("Run Phase III (Markers)"):
                if not os.path.exists("data/objects/Dictionary_of_returns_from_orch_A.json"):
                    st.error("⛔ **Execution Blocked:** Topological dictionaries not found.")
                else:
                    with st.spinner("Extracting Biological Markers..."):
                        try:
                            P05_top_markers.main()
                            st.success("Phase III Complete: Telemetry Ready.")
                        except Exception as e:
                            st.error(f"Phase III Failed: {e}")

        # =====================================================================
        # THE TEMPORAL AIRLOCKS (MACRO AND MICRO ITERATIONS)
        # =====================================================================
        
        # AIRLOCK 1: THE MACRO LOCK
        if st.session_state.phase2_macro_swept and not st.session_state.phase2_macro_locked:
            st.divider()
            st.markdown("### 🛑 Phase II: Human-in-the-Loop [MACRO-STATE]")
            
            svg_file = "./results/figures/p04_clustering/macro_thermodynamic_surface.svg"
            if os.path.exists(svg_file):
                with open(svg_file, "r", encoding="utf-8") as f:
                    svg_code = f.read()
                svg_code = re.sub(r'width="[^"]+"', 'width="100%"', svg_code, count=1)
                svg_code = re.sub(r'height="[^"]+"', 'height="auto"', svg_code, count=1)
                b64_svg = base64.b64encode(svg_code.encode("utf-8")).decode("utf-8")
                img_src = f"data:image/svg+xml;base64,{b64_svg}"
                st.markdown(f"<div style='border: 1px solid #444; padding: 10px; background: white;'><img src='{img_src}' style='width: 100%;'></div>", unsafe_allow_html=True)

            with st.form("macro_override_form"):
                c1, c2 = st.columns(2)
                with c1:
                    chosen_k = st.number_input("Optimal k (Neighbors)", value=int(st.session_state.p04_suggested_k), min_value=10, max_value=100, step=5)
                with c2:
                    chosen_r = st.number_input("Optimal r (Resolution)", value=float(st.session_state.p04_suggested_r), min_value=0.01, max_value=2.0, step=0.01)
                    
                if st.form_submit_button("Lock Macro Coordinates & Initiate Micro-Queue", type="primary"):
                    with st.spinner("Locking Macro-State and Extracting Micro-Continents..."):
                        macro_state = P04_clustering.lock_macro_and_extract_micro_queue(
                            st.session_state.p04_training_file_path, chosen_k, chosen_r, './data/regev_lab_cell_cycle_genes.txt'
                        )
                        st.session_state.macro_leiden_key = macro_state['macro_leiden_key']
                        st.session_state.macro_neighbors_key = macro_state['macro_neighbors_key']
                        st.session_state.micro_filepaths_dict = macro_state['micro_filepaths_dict']
                        
                        # Build the processing queue (Ignoring Terminal States automatically)
                        st.session_state.micro_queue = [k for k in macro_state['micro_filepaths_dict'].keys() if 'Terminal_State' not in k]
                        
                        st.session_state.phase2_macro_locked = True
                        st.rerun()

        # AIRLOCK 2: THE MICRO LOOP
        if st.session_state.phase2_macro_locked and not st.session_state.phase2_complete:
            st.divider()
            
            # If we have items in the queue, process the next one
            if len(st.session_state.micro_queue) > 0:
                current_micro = st.session_state.micro_queue[0]
                filepath = st.session_state.micro_filepaths_dict[current_micro]
                
                st.markdown(f"### 🛑 Phase II: Human-in-the-Loop [MICRO-STATE: `{current_micro}`]")
                st.info(f"{len(st.session_state.micro_queue)} Micro-states remaining in queue.")
                
                # Step 2A: Automatically Sweep the current micro-state if not done yet
                if not st.session_state.current_micro_swept:
                    with st.spinner(f"Sweeping thermodynamic surface for {current_micro}..."):
                        micro_sweep = P04_clustering.execute_micro_sweep(filepath, current_micro, "./results/figures/p04_clustering")
                        st.session_state.current_micro_k = micro_sweep['suggested_k']
                        st.session_state.current_micro_r = micro_sweep['suggested_r']
                        st.session_state.current_micro_swept = True
                        st.rerun()
                
                # Step 2B: Display the Map and Wait for Human Input
                if st.session_state.current_micro_swept:
                    svg_file = f"./results/figures/p04_clustering/{current_micro}_thermodynamic_surface.svg"
                    if os.path.exists(svg_file):
                        with open(svg_file, "r", encoding="utf-8") as f:
                            svg_code = f.read()
                        svg_code = re.sub(r'width="[^"]+"', 'width="100%"', svg_code, count=1)
                        svg_code = re.sub(r'height="[^"]+"', 'height="auto"', svg_code, count=1)
                        b64_svg = base64.b64encode(svg_code.encode("utf-8")).decode("utf-8")
                        img_src = f"data:image/svg+xml;base64,{b64_svg}"
                        st.markdown(f"<div style='border: 1px solid #444; padding: 10px; background: white;'><img src='{img_src}' style='width: 100%;'></div>", unsafe_allow_html=True)
                    
                    with st.form(f"micro_override_{current_micro}"):
                        c1, c2 = st.columns(2)
                        with c1:
                            chosen_k = st.number_input("Optimal k (Neighbors)", value=int(st.session_state.current_micro_k), min_value=5, max_value=60, step=5)
                        with c2:
                            chosen_r = st.number_input("Optimal r (Resolution)", value=float(st.session_state.current_micro_r), min_value=0.1, max_value=2.0, step=0.1)
                            
                        if st.form_submit_button(f"Lock `{current_micro}` & Proceed to Next", type="primary"):
                            with st.spinner(f"Locking {current_micro} and calculating Jaccard Stability..."):
                                micro_result = P04_clustering.lock_micro_state(
                                    filepath, current_micro, chosen_k, chosen_r, './data/regev_lab_cell_cycle_genes.txt'
                                )
                                # Save the dictionaries
                                if micro_result['m_leiden']:
                                    st.session_state.final_micro_leiden_dict[current_micro] = micro_result['m_leiden']
                                    st.session_state.final_micro_neighbors_dict[current_micro] = micro_result['m_neighbors']
                                
                                # Pop the completed item from the queue and reset the sweep state for the next one
                                st.session_state.micro_queue.pop(0)
                                st.session_state.current_micro_swept = False
                                st.rerun()
            
            # Step 3: If queue is empty, Seal the Pipeline
            else:
                st.success("Micro-Queue Exhausted. All states locked.")
                if st.button("Seal Pipeline and Execute Orchestrator B Projection", type="primary"):
                    with st.spinner("Sealing final JSON ledgers and casting projections..."):
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

        st.markdown("### 4. Artifact Extraction")
        st.write("Extract intermediate tensors before container hibernation.")
        
        qc_path = "data/objects/pbmc3k_qc.h5ad"
        if os.path.exists(qc_path):
            with open(qc_path, "rb") as f:
                st.download_button(
                    label="Download P03/P04 State (pbmc3k_qc.h5ad)",
                    data=f,
                    file_name="pbmc3k_qc_custom.h5ad",
                    mime="application/octet-stream"
                )

    # --- TAB 3: VISUAL TELEMETRY ---
    pipeline_state_file = "data/objects/Dictionary_of_returns_from_orch_B.json"
    with tab_telemetry:
        if not master_map:
            st.info("📡 **Visual Telemetry Offline:** The workspace is currently sterile. Please ingest 10X matrices in the Control Room and execute Phases I through III to generate telemetry.")
        else:
            st.header("📊 Visual Telemetry Vault")
            st.markdown("Select an analytical sector to project physical evidence onto the dashboard.")
 
            SECTOR_MAP = {
                "Phase I: Quality Control (P03)": "p03_qc_filtering",
                "Phase II: Topological Audits (P04)": "p04_clustering",
                "Phase III: Marker Extractions (P05)": "p05_top_markers"
            }

            selection = st.selectbox(
                "Select Analytical Sector", 
                options=list(SECTOR_MAP.keys()),
                index=0,
                key="telemetry_sector_selector" # Unique key forces DOM isolation
            )

            target_sub_dir = SECTOR_MAP[selection]
            
            st.caption(f"Scanning Physical Path: `./results/figures/{target_sub_dir}/`")
            st.divider()
            with st.container():
                render_visual_telemetry(target_sub_dir, selection)
            
    # --- TAB 2: ANNOTATION ENGINE ---
    with tab_annotate:
        if not master_map:
            st.info("✍️ **Annotation Engine Offline:** Awaiting structural topology. Execute the pipeline in the Control Room to unlock biological annotation.")
        else:
            st.markdown("### Human-in-the-Loop Topology Verification & Mapping")
            if active_path and active_label_key:
                st.subheader(f"Interrogating Topology: `{active_label_key}`")
                adata = load_tensor(active_path)
                if adata is not None:
                    if 'final_top_genes_per_cluster' in adata.uns:
                        df_markers = adata.uns['final_top_genes_per_cluster']
                        st.markdown("**Top Extracted Thermodynamic Markers (Wilcoxon Rank-Sum)**")
                        st.dataframe(
                            df_markers[['group', 'names', 'pvals_adj', 'logfoldchanges', 'violin_delta']], 
                            use_container_width=True
                        )
                    else:
                        if "Terminal_State" in active_path:
                            st.info(
                                "**Terminal State Confirmed.**\n"
                                "This micro-topology is mathematically homogeneous. Differential marker "
                                "extraction requires at least two sub-clusters to calculate variance. "
                                "Please refer to the parent Macro topology (`macro_leiden_2`) to view the "
                                "defining thermodynamic markers for this lineage."
                            )
                        else:
                            st.warning("Marker dictionary missing from active tensor. Execute Phase III `P05_top_markers.py`.")

                    st.divider()
                    st.markdown("### 🧬 Dual-Ledger Annotation Injection")
                    
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
                        
                        st.markdown("Double-click a cell to edit. Press **Enter** to lock the value into transient memory.")
                        edited_df = st.data_editor(
                            df_ui, 
                            use_container_width=True, 
                            hide_index=True,
                            disabled=["Cluster ID"]
                        )

                        if st.button("💾 Seal Dual Ledgers to Disk", type="primary"):
                            for _, row in edited_df.iterrows():
                                c_id = row["Cluster ID"]
                                label = row["Biological Identity"]
                                cl_id = row["Cell Ontology (CL) ID"]
                                
                                st.session_state.annotations[active_label_key][c_id] = label
                                st.session_state.ontologies[active_label_key][c_id] = cl_id
                                    
                            save_json_ledger(ANNOTATION_PATH, st.session_state.annotations)
                            save_json_ledger(ONTOLOGY_PATH, st.session_state.ontologies)
                            
                            st.success("Ledgers mathematically fractured and written to disk. Ready for Phase IV Injection.")

                        st.divider()
                        st.markdown("### ⚙️ Phase IV: Artifact Recombination")
                        st.markdown("Execute this strictly *after* you have completely populated and sealed the ledgers above.")
                        
                        if st.button("🚀 Execute P06 & Generate ML Artifact", type="secondary"):
                            with st.spinner("Injecting topologies and recombining global matrix..."):
                                try:
                                    # Native Python Execution via the Module Wormhole
                                    P06_annotation.main()
                                    st.success("Matrix successfully recombined and sealed!")
                                        
                                    # Locate the generated artifact to offer a download
                                    ml_ready_path = "./data/objects/pbmc3k_qc_ML_Ready.h5ad"
                                    if op.exists(ml_ready_path):
                                        with open(ml_ready_path, "rb") as file:
                                            st.download_button(
                                                label="⬇️ Download Final ML-Ready Matrix",
                                                data=file,
                                                file_name="pbmc3k_ML_Ready.h5ad",
                                                mime="application/octet-stream"
                                            )
                                except Exception as e:
                                    st.error(f"P06 Execution Failed. Thermodynamic fracture detected:")
                                    st.code(str(e))
                    else:
                        st.error(f"[ERROR] Required observation column '{cluster_col}' missing from matrix.")
                else:
                    st.error(f"[ERROR] Failed to load matrix at physical path: {active_path}")

            st.markdown("---")
            st.markdown("### ⚠️ Engine Diagnostics & Memory Purge")
            st.markdown("Execute this if the UI caches phantom annotations or ghost topologies from a previous run.") 
            if st.button("🔥 PURGE ANNOTATION MEMORY (HARD RESET)", type="primary"):
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