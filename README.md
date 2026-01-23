# PBMC3k-reproducible : 

**Status:** EXECUTION MODE.
**Objective:** Reproduce the PBMC3k dataset analysis from First Principles.

This is NOT a tutorial. This is a **Forensic Reconstruction**. We are auditing the pipeline to validate our **Theory of Variance**. We assume the standard pipeline might be flawed and requires rigorous proof at every step.

## Execution Constraints
1. **The Physical Object:** Explicitly tracking the transformation (e.g., Light Signal → Probability → Count).
2. **The Assumptions:** Stating mathematical simplifications.
3. **The Bridge Axiom:** Justifying steps with derived truth (e.g., Axiom A1: Poisson Limit).
4. **The Failure Mode:** Analyzing what breaks if a step is skipped.
5. **The Modernity Audit:** Comparing 2018 methods against 2024/2025 standards.

## Architecture
- **src/**: Modular logic corresponding to the 9 Phases.
- **data/**:
    - \raw: Immutable inputs (BAM/FASTQ or Matrices).
    - \processed: Canonical AnnData objects.
- **notebooks/**: Audits and derivations.


-------
### The Repo Structure:

/PBMC3k-reproducible
│
├── README.md                      # The Forensic Log: Project Mission, Assumptions, and 5-Sigma Status.
├── environment.yml                # The "Laws of Physics": Conda environment (Scanpy, PyTorch, scvi-tools).
├── .gitignore                     # Exclusion rules (e.g., ignore large *.h5ad files, keep code).
├── License                        # MIT License
│
│── docs/
│   └── roadmap.md                 # How the phases 1 to 9 are covered
│       
│── data/
│   |── raw/                       # IMMUTABLE INPUTS (Read-Only)
│   │   │── pbmc3k_raw_gene_bc_matrices/ # Starting Point (if skipping Phase I)
│   │   │   └── hg19/
│   │   │      ├── matrix.mtx
│   │   │      ├── barcodes.tsv
│   │   │      └── genes.tsv
│   │   │
│   │   └── pbmc3k_filtered_gene_bc_matrices/  # SP (if skipping raw matrices)   
│   │       └── hg19/
│   │          ├── matrix.mtx
│   │          ├── barcodes.tsv
│   │          └── genes.tsv
│   │
│   └── objects/                   # The State Vectors (H5AD Checkpoints)
│       ├── pbmc3k_raw.h5ad        # Output of P02
│       ├── pbmc3k_qc.h5ad         # Output of P03
│       ├── pbmc3k_norm.h5ad       # Output of P04
│       └── pbmc3k_final.h5ad      # The Grand Unification Object (Phase IX)
│
│── results/
│   │
│   ├── figures/                   # The Visual Proofs
│   │   │── phase3_qc/
│   │   │   └── qc_violin_plots.png
│   │   │── phase4_variance/
│   │   │   └── mean_variance_trend.png
│   │   │── phase5_geometry/
│   │   │   └── pca_elbow_plot.png
│   │   │   └── umap_leiden_clusters.png
│   │   │── phase7_markers/
│   │   │   └── dotplot_top_markers.png
│   │   └── phase8_function/
│   │       └── gsea_bcell_pathway.png
│   │
│   ├── tables/                    # The Digital Proofs
│   │   ├── markers_wilcoxon.csv   # Differential Expression Stats
│   │   ├── cell_type_mapping.csv  # Cluster ID -> Biological Name
│   │   └── gsea_results.csv       # Functional Enrichment Scores
│   └── report/                    # Final Certificate
│       └── forensic_audit_pbmc3k.pdf
│
└── src/
    ├── 01_upstream_pipeline/      # THE TOMBSTONE (Reference Only)    
    │   └── Readme.md              # "Provenance Log: Data sourced from 10x Genomics v1.1.0..."
    │
    └── 02_analysis_scripts/           # THE PYTHON LOGIC CORE (Phases II-X)
        ├── __init__.py                # Makes this a package
        ├── utils.py                   # Shared physics (Plotting styles, Helper functions)
        │
        ├── P02_matrix_construction.py # Phase II: Load 10x -> AnnData (N x p enforcement)
        ├── P03_qc_filtering.py        # Phase III: MAD-based outlier detection
        ├── P04_normalization.py       # Phase IV: SCTransform vs LogNorm (Variance Decoupling)
        ├── P05_latent_geometry.py     # Phase V: PCA (Eigenstructure) & Neighbor Graph
        ├── P06_clustering.py          # Phase V(B): Leiden Community Detection
        ├── P07_markers_identity.py    # Phase VI/VII: Marker Discovery & Annotation
        ├── P08_functional_gsea.py     # Phase VIII: Pathway Enrichment (Random Walk)
        ├── P09_final_synthesis.py     # Phase IX: The Final Report Generation
        └── P10_causal_inference.py    # Phase X: The Causal DAG (Regulon Inference)
