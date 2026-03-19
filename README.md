# PBMC3k-reproducible

**Status:** EXECUTION MODE  
**Objective:** Reproduce the PBMC3k dataset analysis from First Principles.  

This is NOT a tutorial. This is a **Forensic Reconstruction**.  
We are auditing the pipeline to validate our Theory of Variance. We assume the standard pipeline might be flawed and requires rigorous computational proof at every structural node.

---

### Execution Constraints

1. **The Physical Object:** Explicitly tracking the transformation (e.g., Light Signal Probability → Count). Matrix orientation is strictly maintained as `Cells x Genes`.
2. **The Assumptions:** Stating mathematical simplifications and thermodynamic floors explicitly.
3. **The Bridge Axiom:** Justifying steps with derived truth (e.g., Axiom A1: Poisson Limit).
4. **The Failure Mode:** Analyzing exactly what breaks if a step is bypassed or abstracted.
5. **The Modernity Audit:** Comparing foundational 2018 methods against stringent 2024/2025 industrial standards.

---

### Global Architecture

* **`src/`**: The immutable Python logic core. Divided into upstream processing and downstream analysis scripts.
* **`data/`**: The physical data lake containing unadulterated raw inputs, checkpointed `.h5ad` state vectors, and absolute biological reference dictionaries.
* **`results/`**: The output staging ground. Houses the generated JSON ledgers, CSV topologies, and all cross-validation visual evidence.
* **`notebooks/`**: The computational workshop for initial audits, visual derivations, and parameter testing.

---

### Repository Structure

```
/PBMC3k-reproducible
│
├── .conda/                             # Isolated thermodynamic environment (Python runtime)(Hidden Not uploaded)
├── cache/                              # Temporary execution buffers (Hidden Not uploaded)
│
├── data/                               # The Data Lake (Hidden Not uploaded)
│   ├── celltypist_models/              # Automated reference-based annotation models
│   ├── objects/                        # Checkpointed AnnData (.h5ad) state vectors
│   ├── raw/                            # Immutable 10x Genomics inputs
│   │   ├── pbmc3k_filtered_gene_bc_matrices/
│   │   ├── pbmc3k_raw_gene_bc_matrices/
│   │   └── pbmc3k_molecule_info.h5
│   ├── reconstructed_matrices_final/   # Post-CellBender/Upstream corrected matrices
│   │   └── raw_gene_bc_matrices/
│   │       ├── barcodes.tsv
│   │       ├── genes.tsv
│   │       ├── matrix.mtx
│   │       └── matrix.mtx.gz
│   ├── regev_lab_cell_cycle_genes.txt  # Biological reference for cell cycle scoring
│   ├── Teichlab_curated_markers.json   # Canonical marker validation dictionary
│   └── universal_ontology_id_dict.json # Standardized Cell Ontology (CL) mapping
│
├── docs/                               # Project documentation and phase roadmaps
│
├── results/                            # Output staging and visual telemetry
│   └── figures/                        # The Visual Proofs
│       ├── p03_qc_filtering/           # MAD boundaries, cell cycle scoring, and dropout audits
│       ├── p04_clustering/             # Subsampling stability sweeps and thermodynamic overlap 
│       ├── p05_top_markers/            # Canonical dotplots and absence-audit cross-validation
│       └── p06_annotation/             # Reference mapping and final topology overlays
│
├── src/                                # The Python Logic Core
│   ├── 01_upstream_pipeline/           # The Tombstone (Reference for FASTQ/BAM -> Matrix)
│   └── 02_analysis_scripts/            # The 5-Sigma Pipeline Engines
│       ├── P02_matrix_construction.py  # Data ingestion and tensor formatting
│       ├── P03_qc_filtering.py         # Phase I: 5-MAD outlier detection and matrix purge
│       ├── P04_clustering.py           # Phase II: Latent geometry, KNN, and Leiden resolution arrays
│       ├── P05_top_markers.py          # Phase III: Wilcoxon rank-sum extraction and lineage validation
│       └── P06_annotation.py           # Phase IV: Ledger injection, ontology mapping, and final ML Tensor
│
├── .gitattributes                      # Git LFS and line-ending configurations
├── .gitignore                          # Exclusion rules (ignores large *.h5ad files, tracks code)
├── environment.yml                     # The "Laws of Physics": Conda dependencies
├── LICENSE                             # MIT License
└── README.md                           # The Forensic Log: Project Mission and Constraints