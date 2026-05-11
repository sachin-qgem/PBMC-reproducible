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
- [![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.19335670-blue)](https://doi.org/10.5281/zenodo.19335670)
- [![ORCID](https://img.shields.io/badge/ORCID-0009--0000--2744--6131-A6CE39?logo=orcid&logoColor=white)](https://orcid.org/0009-0000-2744-6131)
- [![Report](https://img.shields.io/badge/Report-PDF-darkred?logo=adobeacrobatreader&logoColor=white)](report.pdf) [View the Full Analytical Thesis](report.pdf) 
- [![Hugging Face Space](https://img.shields.io/badge/🤗_Hugging_Face-Live_Deployment-FFD21E)](https://huggingface.co/spaces/sachin-qgemai-alpha/pbmc_single_healthy_donor) [Interact with the Live Pipeline](https://huggingface.co/spaces/sachin-qgemai-alpha/pbmc_single_healthy_donor)
- **Status:** EXECUTION MODE 
- **Objective:** Reproduce the PBMC dataset analysis from First Principles using a Human-in-the-Loop architecture.

**⚠️⚠️⚠️THIS IS EXPLICITLY FOR SINGLE DONOR ONLY AT THIS MOMENT. DEVELOPMENT GOING ON FOR MULTI_DONOR, BATCH INTEGRATION AND CORRECTION⚠️⚠️⚠️**

This is NOT a tutorial. This is a **Forensic Reconstruction**.  
We are auditing the standard pipeline to validate our learnt theory. We assume default automated pipelines are mathematically flawed and require rigorous computational proof at every structural node.
---
### The Phases of Execution:

#### ▶ Phase I: Quality Control & Preprocessing

- **Adaptive MAD Filtering**: Removal of low-quality cells and technical artifacts by applying a strict Median Absolute Deviation (MAD) of 5 to Mito % and total expressed genes. Ribosomal fractions are calculated but not filtered out at this stage. Removing them early would artificially reduce total cellular counts, skewing the expected variance baseline required for Pearson residuals in downstream steps.
    
- **Doublet Removal**: Identification and filtering of synthetic multi-cell droplets using Scrublet to prevent artificial clustering between distinct cell types.
    
- **Gene Sparsity Filtering**: Removal of uninformative or sparsely expressed genes (enforcing a minimum threshold of detection in ≥ 3 cells).
    
- **Library Size Normalization:** Log1p-transformation and target-sum scaling (1e4) of the raw count matrix to normalize cellular sequencing depth.
    

#### ▶ Phase II: Structural Topology &  Clustering

- **Data Splitting (Train/Project)**: A strict 50/50 split of the dataset into training and holdout (projection) sets to prevent data leakage and circular inference during downstream validation.
    
- **Iterative Dimensionality Reduction & HVG Selection**: Pearson residuals, highly variable genes (HVGs), and PCA are recalculated exclusively on the sub-matrices. Crucially, HVGs are computed on the full gene set to establish an accurate variance baseline. Only after this calculation are mitochondrial and ribosomal genes explicitly masked from the HVG pool prior to PCA, preventing housekeeping genes from driving spatial clustering.
    
- **Cell Cycle Scoring & Orthogonal Projection:** Auditing S-phase and G2M-phase genes to ensure topological clustering is driven by core phenotypic identity, not transient mitotic states.
    
- **Hyperparameter Grid Search (Stability Audit**): Evaluating cluster stability using Jaccard survival scores (20 iterations, 0.8 subsample) and structural Modularity (Q) across a combinatorial grid of KNN and Leiden resolutions to deterministically lock boundaries without human bias.
    
- **PCA Variance Analysis**: Examining the PCA variance ratio to distinguish continuous developmental gradients from discrete sub-populations, determining if further sub-clustering is necessary.
    
- **Holdout Projection**: Projecting the unseen 50% holdout data onto the established training reference (using Scanpy Ingest) to validate cluster boundary generalization.
    

#### ▶ Phase III: Differential Gene Expression (DGE) & Marker Extraction

- **Wilcoxon Rank-Sum Test**: Computing cluster-specific marker genes. Clusters with fewer than 10 cells are excluded from DGE to prevent statistically unreliable results.
    
- **Log-Fold Boundaries:** Isolating significant markers by strictly enforcing `logfoldchanges > 1.0` while dynamically capping extreme dropout artifacts (`logfoldchanges < 10.0`), alongside a baseline significance of `pvals_adj < 0.05`.
    
- **Adaptive P-value Thresholding**: Applying a local 93.75th-percentile cutoff to p-values to isolate the most significant markers per cluster, with a fallback minimum of 3 genes to ensure small clusters retain defining markers.
    
- **Spatial Exclusivity Scoring (`violin_delta`):** Mathematically isolating the purest marker genes by calculating the expression differential (`pct_nz_group` - `pct_nz_reference`).
    
- **Cross-Cluster Marker Auditing**: Comparing top markers against neighboring micro-clusters to verify lineage separation and distinct expression profiles.
    
- **Canonical Ledger Validation:** Auditing machine-derived markers against a pre-curated JSON dictionary of established biological truths (e.g., Theis Lab signatures).
    

#### ▶ Phase IV: Annotation & Data Recombination

- **Manual Ontology Injection:** Mapping mathematically validated gene signatures to standard Cell Ontology (CL) IDs within the isolated Macro and Micro matrices.
    
- **Automated Annotation Validation**: Comparing manual annotations against predictions from pre-trained, supervised Neural Networks (`CellTypist` Immune models).
    
- **Concordance Scoring:** Calculating the Adjusted Rand Index (ARI) between the human labels and the machine's majority voting logic to prove structural agreement.
    
- **Metadata Aggregation:** Consolidating cell barcodes and annotations from all sub-matrices into a master CSV, using a 'latest-execution-wins' logic to resolve duplicate barcodes.
    
- **Global Tensor Recombination:** Ingesting the master FAIR-compliant CSV ledger and mapping the final biological identities back onto the raw, un-split global matrix. Unannotated or failed cells are programmatically standardized to `Unknown/Filtered` before exporting the final Machine-Learning-Ready (`.h5ad`) artifact.

---

### Architectural Enhancements

This pipeline improves upon the default workflows of standard single-cell tools (e.g., Seurat/Scanpy) by introducing three rigorous computational validation steps:
1. **Hyperparameter Grid Search (Modularity Audit):** Instead of relying on arbitrary Leiden resolutions, Phase II generates a grid-search contour map of Modularity (Q) across varying k-neighbors and resolutions. This guides parameter selection toward stable modularity plateaus rather than volatile transition zones, ensuring mathematically robust cluster boundaries.
2. **Jaccard Bootstrapping:** Chosen clustering parameters are empirically validated by randomly subsampling 80% of the cells across 20 iterations to quantify and prove cluster stability.
3. **Sub-Clustering Stopping Criterion:** Evaluates the PCA variance ratio to dynamically determine whether a cluster represents a homogeneous biological population (terminal state) or contains further substructure requiring additional micro-clustering.

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

## ⚠️ Note on High-Fidelity Visual Rendering
This repository contains high-density single-cell transcriptomic visualizations (UMAPs, PCA manifolds, and topological surfaces) exported as scalable vector graphics (`.svg`). 

Due to GitHub's aggressive Content Security Policy (Camo proxy) restricting complex XML/SVG execution in the browser, these figures may occasionally fail to render or appear as blank spaces in the online view of `report.md`.

**To view the full analytical report with all structural telemetry intact:**
1. Clone or download this repository to your local machine.
2. Open the project folder using **Obsidian**, **Visual Studio Code**, or any standard local Markdown engine. 
3. All relative paths and high-resolution figures will render natively.
4. OR a report.pdf File is also there in the repo


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
├── report.md                           # The Methodology and Results (IMRAD)
└── README.md                           # The Forensic Log: Project Mission and Constraints
```
