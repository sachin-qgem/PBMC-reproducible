"""
Phase IV: Comprehensive Orchestration & UI Bridge (Streamlit Engine)

This module operates as the master control interface for human-in-the-loop 
topological mapping. It utilizes cached memory to pin massive .h5ad tensors 
in RAM, constructs an interactive 2D matrix for biological identity injection, 
and features a Visual Telemetry scanner to project physical pipeline artifacts 
(PNGs) into the browser. Finally, it commands a Subprocess engine to execute 
the Phase IV Recombination script entirely in the background.
"""

import gc
import json
import os
import os.path as op
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional

import anndata as ad
import pandas as pd
import scanpy as sc
import streamlit as st
from PIL import Image

# =============================================================================
# GLOBAL THERMODYNAMIC CONSTANTS & UI CONFIGURATION
# =============================================================================
# Must be the absolute first Streamlit command executed
st.set_page_config(page_title="PBMC3k Dual-Ledger Annotation", layout="wide")

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
    Scans the physical disk for PNG artifacts and renders them in a grid.
    
    Parameters
    ----------
    sub_dir : str
        The specific folder inside `results/figures/` to scan.
    title : str
        The header text to display above the rendered images.
    """
    target_dir = op.join("./results/figures/", sub_dir)
    st.markdown(f"#### {title}")
    
    if not op.exists(target_dir):
        st.info(f"Directory not found: {target_dir}")
        return
        
    # Dynamically extract absolute paths of all PNGs in the target sector
    png_files = list(Path(target_dir).rglob("*.svg"))
    
    if not png_files:
        st.info(f"No telemetry artifacts found in {sub_dir}.")
        return
        
    # Build a 2-column mathematical grid
    cols = st.columns(2)
    for idx, img_path in enumerate(png_files):
        with cols[idx % 2]:
            st.image(Image.open(img_path), caption=img_path.name, use_container_width=True)

# =============================================================================
# MAIN ORCHESTRATION ENGINE
# =============================================================================

def main() -> None:
    """
    The primary execution loop for the Streamlit UI. Handles state initialization, 
    matrix selection routing, transient dataframe rendering, Subprocess execution,
    and the final fracture protocol to seal data to disk.
    """
    st.title("🧬 PBMC3k Dual-Ledger Annotation Engine")
    
    # 1. Establish the RAM Vault
    initialize_session_vault()
    
    # 2. Ingest the Master Map (The Orchestrator B Output)
    master_map = load_json_ledger(DICT_B_PATH)
    if not master_map:
        st.error(f"[CRITICAL FAILURE] Master state dictionary missing at `{DICT_B_PATH}`. Execute Phase II.")
        st.stop()
        
    # Synchronize physical ledgers with RAM on initial load only
    if not st.session_state.annotations:
        st.session_state.annotations = load_json_ledger(ANNOTATION_PATH)
    if not st.session_state.ontologies:
        st.session_state.ontologies = load_json_ledger(ONTOLOGY_PATH)

    # 3. Sidebar Navigation Architecture
    st.sidebar.header("Navigation")
    macro_key = master_map.get('macro_leiden_key_training')
    micro_paths = master_map.get('projected_micro_file_path_dictionary', {})
    micro_leiden_dict = master_map.get('projected_micro_leiden_key_dictionary', {})
    
    view_mode = st.sidebar.radio("Topology Level", ["Macro Level", "Micro Level"])
    
    active_path = None
    active_leiden = None
    active_label_key = None
    
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

    # 4. The Tab Architecture
    tab_annotate, tab_telemetry = st.tabs(["🧬 Annotation Engine", "📊 Visual Telemetry"])

    # --- TAB 2: VISUAL TELEMETRY ---
    with tab_telemetry:
        st.header("📊 Visual Telemetry Vault")
        st.markdown("Select an analytical sector to project physical evidence onto the dashboard.")

        # 1. THE SECTOR MAP: Absolute 1-to-1 Mapping
        # Keys are UI Labels, Values are physical folder names
        SECTOR_MAP = {
            "Phase I: Quality Control (P03)": "p03_qc_filtering",
            "Phase II: Topological Audits (P04)": "p04_clustering",
            "Phase III: Marker Extractions (P05)": "p05_top_markers"
        }

        # 2. THE SELECTION ENGINE
        # This widget triggers a full script rerun on every change
        selection = st.selectbox(
            "Select Analytical Sector", 
            options=list(SECTOR_MAP.keys()),
            index=0,
            key="telemetry_sector_selector" # Unique key forces DOM isolation
        )

        # 3. PHYSICAL RESOLUTION
        target_sub_dir = SECTOR_MAP[selection]
        
        # Diagnostic Telemetry (Small text to verify the path in real-time)
        st.caption(f"Scanning Physical Path: `./results/figures/{target_sub_dir}/`")

        st.divider()

        # 4. THE EXECUTION
        # Using a container ensures the previous tab content is physically purged
        with st.container():
            render_visual_telemetry(target_sub_dir, selection)
            
    # --- TAB 1: ANNOTATION ENGINE ---
    with tab_annotate:
        st.markdown("### Human-in-the-Loop Topology Verification & Mapping")
        
        # Dynamic Tensor Interrogation & Fusion
        if active_path and active_label_key:
            st.subheader(f"Interrogating Topology: `{active_label_key}`")
            adata = load_tensor(active_path)
            
            if adata is not None:
                # Render Extracted Thermodynamic Markers (Read-Only)
                if 'final_top_genes_per_cluster' in adata.uns:
                    df_markers = adata.uns['final_top_genes_per_cluster']
                    st.markdown("**Top Extracted Thermodynamic Markers (Wilcoxon Rank-Sum)**")
                    st.dataframe(
                        df_markers[['group', 'names', 'pvals_adj', 'logfoldchanges', 'violin_delta']], 
                        use_container_width=True
                    )
                else:
                    # Check if the void is due to mathematical purity (Terminal State)
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

                # -----------------------------------------------------------------
                # 5. The Transient DataFrame Constructor
                # -----------------------------------------------------------------
                st.divider()
                st.markdown("### 🧬 Dual-Ledger Annotation Injection")
                
                # Extract rigid mathematical clusters from the physical matrix
                cluster_col = active_leiden if active_leiden else active_label_key
                
                if cluster_col in adata.obs.columns:
                    clusters = sorted(
                        adata.obs[cluster_col].dropna().unique().tolist(), 
                        key=lambda x: int(x) if str(x).isdigit() else x
                    )
                    
                    # Ensure the structural keys exist in the RAM vault
                    if active_label_key not in st.session_state.annotations:
                        st.session_state.annotations[active_label_key] = {str(c): "" for c in clusters}
                    if active_label_key not in st.session_state.ontologies:
                        st.session_state.ontologies[active_label_key] = {str(c): "" for c in clusters}
                        
                    # Zipping Protocol: Fuse parallel ledgers into a flat 2D grid
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
                        disabled=["Cluster ID"]  # Immutable barrier protecting the core topology
                    )

                    # -----------------------------------------------------------------
                    # 6. The Commit Protocol (The Fracture)
                    # -----------------------------------------------------------------
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

                    # -----------------------------------------------------------------
                    # 7. The Recombination Engine (Executing P06)
                    # -----------------------------------------------------------------
                    st.divider()
                    st.markdown("### ⚙️ Phase IV: Artifact Recombination")
                    st.markdown("Execute this strictly *after* you have completely populated and sealed the ledgers above.")
                    
                    if st.button("🚀 Execute P06 & Generate ML Artifact", type="secondary"):
                        with st.spinner("Injecting topologies and recombining global matrix..."):
                            # Command the Linux server to run your P06 script
                            result = subprocess.run(
                                ["python", "src/02_analysis_scripts/P06_annotation.py"], 
                                capture_output=True, 
                                text=True
                            )
                            
                            if result.returncode == 0:
                                st.success("Matrix successfully recombined and sealed!")
                                with st.expander("View P06 Execution Logs"):
                                    st.code(result.stdout)
                                    
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
                            else:
                                st.error("P06 Execution Failed. Review the thermodynamic fracture logs below.")
                                st.code(result.stderr)
                else:
                    st.error(f"[ERROR] Required observation column '{cluster_col}' missing from matrix.")
            else:
                st.error(f"[ERROR] Failed to load matrix at physical path: {active_path}")

if __name__ == "__main__":
    main()