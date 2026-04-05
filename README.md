---
title: PBMC Single Donor Healthy
emoji: 🧬
colorFrom: indigo
colorTo: blue
sdk: streamlit
python_version: 3.11
sdk_version: 1.42.0
app_file: app.py
pinned: false
---

- # PBMC-reproducible: Cybernetic Clustering and markers Engine
- [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19335670.svg)](https://doi.org/10.5281/zenodo.19335670)
- [![ORCID](https://img.shields.io/badge/ORCID-0009--0000--2744--6131-A6CE39?logo=orcid&logoColor=white)](https://orcid.org/0009-0000-2744-6131)
- **Status:** EXECUTION MODE  
- **Live Deployment:** [Access the Cybernetic Engine on Hugging Face Spaces](https://huggingface.co/spaces/sachin-qgemai-alpha/pbmc_single_healthy_donor)
- **Objective:** Reproduce the PBMC dataset analysis from First Principles using a Human-in-the-Loop architecture.

**⚠️⚠️⚠️THIS IS EXPLICITLY FOR SINGLE DONOR ONLY AT THIS MOMENT. DEVELOPMENT GOING ON FOR MULTI_DONOR, BATCH INTEGRATION AND CORRECTION⚠️⚠️⚠️**

This is NOT a tutorial. This is a **Forensic Reconstruction**.  
We are auditing the standard pipeline to validate our learnt theory. We assume default automated pipelines are mathematically flawed and require rigorous computational proof at every structural node.
---
### The Phases:
- #### Phase I(P03_qc_filtering): 
    - 5-MAD outlier detection of `Mito %`
    - Doublets Scrubs
    - Dormant genes removal (genes expressed in less than 3 cells)
    - Cells expressed in less than 200 genes removal
- #### Phase II(P04_clustering):
    - Cell Cycle check
    - Double Dipping (random split main into training and projected 50-50) and data leakage addressed 
    - Pearson Residuals, PCA, HVG , recal after every split of dataset in new space (of Training dataset only).
    - KNN, UMAP, Leiden clustering
    - Topological Mesa audit for a dynamic and informed decision for choosing optimal K-neighbors and Leiden resolution
    - divide and save dataset based on macro or micro clusters
    - Audits the PCA variance geometry to determine if a cluster is a homogenous biological state (an arc) or contains structural subpopulations (an elbow).
    - casting projectable dataset on the clustered training dataset
- #### Phase III(P05_top_markers):
    - wilcoxon rank sum test stat method.
    - filter the `pvals_adj< 0.05` and `logfoldchange < 10.0` (a gene that was only expressed in one single cell out of a thousand. It is statistical ghost data.)
    - calculate the `neg_log10_pvals_adj`
    - local 93 percentile of every cluster for neg_log10_pvals_adj
    - `['violin_delta'] = ['pct_nz_group'] - ['pct_nz_reference']`
    - sort values based on voilen_delta
    - take top 3 or 5 for every cluster
    - Extracts top markers from all foreign matrices and projects them onto the target matrix to prove lineage isolation (Epigenetic Silencing).
    - Executes automated reference-based annotation using CellTypist.
    - Validates the matrix against a pre-curated JSON dictionary of canonical markers from Theis Lab.
- #### Phase IV(P06_annotation):
    - Injects human-verified biological annotations and standard Cell Ontology (CL) IDs into the localized Macro and Micro matrices.
    - Calculates alignment scores against automated reference models.
    - Extracts cell barcodes and injected labels from all isolated matrices and concatenates them into a master CSV ledger. Resolves barcode collisions by prioritizing the most recent execution state.
    - Ingests the master CSV ledger and maps the final biological identities onto the raw, un-split global matrix. Exports the final ML-Ready artifact.

---

### The Architectural Upgrades

This engine abandons the "blind execution" of standard pipelines (e.g., default Seurat/Scanpy). It introduces three strict cybernetic fail-safes:
1. **The Topographical Mesa Audit:** Phase II does not accept hardcoded Leiden resolutions. It generates a dual-pane thermodynamic contour map of Modularity ($\text Q$) across $\text k$-neighbors and $r$-resolution, forcing the human to anchor the algorithm strictly on flat biological continuous gradients (Mesas) rather than volatile mathematical phase-transitions (Cliffs).
2. **Jaccard Sub-Sampling:** Every human coordinate lock is forensically audited by randomly deleting 20% of the cells across 20 iterations to prove cluster stability.
3. **Thermodynamic Terminal State Check**: Audits the PCA variance geometry to determine if a cluster is a homogenous biological state (an arc) or contains structural subpopulations (an elbow).

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
gh repo clone sachin-qgem/PBMC-reproducible
cd PBMC-reproducible
```


**3. Forge the Isolated Background Field:**
We utilize standard Python virtual environments and strict pip dependency ledgers from makefile
```bash
make setup
```

**4. Ignite the Streamlit Orchestrator:**
Do not run the backend scripts manually. Boot the visual interface.
```bash
make run
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
/PBMC-reproducible
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
├── notebooks/                          # Audits and experimental derivations
│
├── results/                            # Output staging and visual telemetry
│   └── figures/                        # The Visual Proofs (QC, Clustering, Markers, Annotation)
│
├── src/                                # The Python Logic Core
│   ├── 01_upstream_pipeline/           # The Tombstone (Reference for FASTQ/BAM -> Matrix) But in pipeline , we use the filtered genes matix as input as I had MacOS only
│   └── 02_analysis_scripts/            # The 5-Sigma Pipeline Engines
│       ├── P02_matrix_construction.py  # Data ingestion and tensor formatting (But in pipeline , we use the filtered genes matix as input as I had MacOS only)
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
