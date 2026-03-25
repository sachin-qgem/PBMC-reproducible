---
title: PBMC3k Single Donor Healthy
emoji: 🧬
colorFrom: indigo
colorTo: blue
sdk: streamlit
python_version: 3.11
sdk_version: 1.42.0
app_file: app.py
pinned: false
---

# PBMC3k-reproducible: 5-Sigma Cybernetic Clustering Engine

**Status:** EXECUTION MODE  
**Live Deployment:** [Access the Cybernetic Engine on Hugging Face Spaces](https://huggingface.co/spaces/qgem/your-space-name) *(Update link to your specific space)* **Objective:** Reproduce the PBMC3k dataset analysis from First Principles using a Human-in-the-Loop MLOps architecture.

**⚠️⚠️⚠️THIS IS EXPLICITLY FOR SINGLE DONOR ONLY AT THIS MOMENT. DEVELOPMENT GOING ON FOR MULTI_DONOR, BATCH INTEGRATION AND CORRECTION⚠️⚠️⚠️**

This is NOT a tutorial. This is a **Forensic Reconstruction**.  
We are auditing the standard pipeline to validate our Theory of Variance. We assume default automated pipelines are mathematically flawed and require rigorous computational proof at every structural node.
---

### The 5-Sigma Architectural Upgrades

This engine abandons the "blind execution" of standard pipelines (e.g., default Seurat/Scanpy). It introduces three strict cybernetic fail-safes:
1. **The Topographical Mesa Audit:** Phase II does not accept hardcoded Leiden resolutions. It generates a dual-pane thermodynamic contour map of Modularity ($Q$) across $k$-neighbors and $r$-resolution, forcing the human to anchor the algorithm strictly on flat biological continuous gradients (Mesas) rather than volatile mathematical phase-transitions (Cliffs).
2. **Jaccard Sub-Sampling:** Every human coordinate lock is forensically audited by randomly deleting 20% of the cells across 20 iterations to prove cluster stability.
3. **UUID Multi-Tenant Isolation:** The Streamlit orchestrator generates cryptographic UUIDs for every concurrent web visitor, forging isolated `/tmp/` workspaces to prevent RAM collision and global state bleed during simultaneous cloud analysis.
---

### Execution Constraints

1. **The Physical Object:** Explicitly tracking the transformation (e.g., Light Signal Probability → Count). Matrix orientation is strictly maintained as `Cells x Genes`.
2. **The Assumptions:** Stating mathematical simplifications and thermodynamic floors explicitly.
3. **The Bridge Axiom:** Justifying steps with derived truth (e.g., Axiom A1: Poisson Limit).
4. **The Failure Mode:** Analyzing exactly what breaks if a step is bypassed or abstracted.
5. **The Modernity Audit:** Comparing foundational methods against stringent industrial standards.

---

### ⚙️ Local Ignition & Environment Setup

If you wish to bypass the Hugging Face live deployment and run the cybernetic engine locally, execute this strict sequence:

**1. Clone the repository:**
```bash
git clone <your-repo-url>
cd PBMC3k-reproducible
```

**2. Acquire the Genesis Data:**
Raw sequencing matrices are heavy matter and excluded from version control. 
* Download the **Filtered gene-barcode matrices** from [10x Genomics](https://www.10xgenomics.com/resources/datasets/3-k-pbm-cs-from-a-healthy-donor-1-standard-1-1-0).
* Extract the `hg19/` folder into the physical data lake exactly here:
```bash
mkdir -p data/raw/pbmc3k_filtered_gene_bc_matrices/
# Place matrix.mtx, barcodes.tsv, and genes.tsv inside this folder
```

**3. Forge the Isolated Background Field:**
We utilize standard Python virtual environments and strict pip dependency ledgers.
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -e .           # Wires the internal logic via the pyproject.toml blueprint
```

**4. Ignite the Streamlit Orchestrator:**
Do not run the backend scripts manually. Boot the visual interface.
```bash
streamlit run app.py
```

---

### Global Architecture

* **`src/`**: The immutable Python logic core. Divided into upstream processing and downstream analysis scripts.
* **`data/`**: The physical data lake containing unadulterated raw inputs, checkpointed `.h5ad` state vectors, and absolute biological reference dictionaries.
* **`results/`**: The output staging ground. Houses the generated JSON ledgers, CSV topologies, and all cross-validation visual evidence.
* **`notebooks/`**: The computational workshop for initial audits, visual derivations, and parameter testing.

---

### Repository Structure

```text
/PBMC3k-reproducible
│
├── .github/workflows/                  # CI/CD Autonomous Bridge to Hugging Face
├── .streamlit/                         # Server configuration limits (e.g., 1GB upload max)
├── .venv/                              # Local isolated Python environment (Git ignored)
├── cache/                              # Temporary execution buffers
│
├── app.py                              <-- The Entry Point (Cybernetic Interface)
├── data/                               # The Data Lake
│   ├── celltypist_models/              # Automated reference-based annotation models
│   ├── objects/                        # Checkpointed AnnData (.h5ad) state vectors
│   ├── raw/                            # Immutable 10x Genomics inputs
│   ├── regev_lab_cell_cycle_genes.txt  # Biological reference for cell cycle scoring
│   ├── Teichlab_curated_markers.json   # Canonical marker validation dictionary
│   └── universal_ontology_id_dict.json # Standardized Cell Ontology (CL) mapping
│
├── docs/                               # Project documentation and phase roadmaps
├── notebooks/                          # Audits and experimental derivations
│
├── results/                            # Output staging and visual telemetry
│   └── figures/                        # The Visual Proofs (QC, Clustering, Markers, Annotation)
│
├── src/                                # The Python Logic Core
│   ├── 01_upstream_pipeline/           # The Tombstone (Reference for FASTQ/BAM -> Matrix)
│   └── 02_analysis_scripts/            # The 5-Sigma Pipeline Engines
│       ├── P02_matrix_construction.py  # Data ingestion and tensor formatting
│       ├── P03_qc_filtering.py         # Phase I: 5-MAD outlier detection and matrix purge
│       ├── P04_latets.py               # Phase II: Latent geometry, Jaccard validation, Topographical Sweep
│       ├── P05_top_markers.py          # Phase III: Wilcoxon rank-sum extraction and lineage validation
│       └── P06_annotation.py           # Phase IV: Ledger injection, ontology mapping, and final ML Tensor
│
├── .gitattributes                      # Git LFS and line-ending configurations
├── .gitignore                          # Exclusion rules (ignores large *.h5ad files, tracks code)
├── pyproject.toml                      # The Architectural Blueprint (for pip install -e .)
├── requirements.txt                    # Strict pinned dependencies for Hugging Face deployment
├── LICENSE                             # MIT License
└── README.md                           # The Forensic Log: Project Mission and Constraints
```