## 1 Introduction

### 1.1 The Biological Manifold and High-Dimensional Transcription

The human immune system operates as a highly dynamic, distributed network of specialized cellular phenotypes. Peripheral Blood Mononuclear Cells (PBMCs) provide a direct, accessible window into this circulating network, comprising lymphocytes (T cells, B cells, NK cells) and myeloid lineages (monocytes, dendritic cells). Historically, the resolution of these populations relied on restricted surface-protein markers. The advent of single-cell RNA-sequencing (scRNA-seq) shattered this limitation, allowing for the unbiased quantification of the entire transcriptome within individual droplets.

Mathematically, scRNA-seq projects biological identity into a high-dimensional tensor, $X \in \mathbb{R}^{C \times G}$, where each viable cell ($C$) is defined by its expression variance across thousands of genomic features ($G$). Within this vast probability space, distinct biological phenotypes exist as dense, localized geometric manifolds. The fundamental objective of downstream bioinformatics is to identify the precise mathematical boundaries of these manifolds.

### 1.2 The Entropic Crisis of Standard Methodologies

While the hardware for transcript capture has reached extraordinary fidelity, the computational methodologies used to resolve the resulting matrices remain critically flawed. Single-cell data is inherently corrupted by physical entropy: ambient RNA leakage from lysed cells, stochastic droplet multiplets (doublets), and extreme variance in sequencing capture efficiency.

Rather than addressing this entropy with strict mathematical laws, conventional downstream pipelines frequently operate as heuristic "black boxes." They rely on arbitrary parameter selection—such as guessing the number of nearest neighbors ($k$) or community resolution ($r$)—until the output visually aligns with human expectation. This "hit and trial" approach introduces massive subjective bias.

Furthermore, standard workflows routinely violate foundational rules of empirical prediction:

- **Data Leakage:** Computing global variance (Highly Variable Genes) and dimensional reduction (PCA) on the entire dataset simultaneously, thereby allowing the training space to illegally "see" the geometric coordinates of the validation space.
    
- **Double Dipping (Circular Inference):** Utilizing the exact same cellular vectors to dynamically define cluster boundaries and subsequently compute the statistical significance of the marker genes defining those boundaries. This mathematical tautology guarantees the hallucination of false-positive phenotypes, even in purely random noise.
    

### 1.3 Architectural Thesis and Protocol Intersection

To resolve the crisis of subjective variance partitioning, this report details the engineering and execution of a fully deterministic, mathematically constrained analytical engine. The architecture abandons heuristic guessing in favor of rigorous thermodynamic mapping, topological bootstrapping, and strict out-of-sample projection.

This pipeline is constructed at the exact intersection of three rigorous protocols:

1. **IMRAD (Introduction, Methods, Results, and Discussion):** Ensuring the logical flow from empirical observation to physical deduction.
    
2. **CRISP-DM (Cross-Industry Standard Process for Data Mining):** Enforcing a strict separation of training and projection data to guarantee the model's predictive validity on unseen reality.
    
3. **FAIR (Findable, Accessible, Interoperable, Reusable):** Prohibiting undocumented, active-memory data mutation. All biological annotations are governed by external, immutable JSON ledgers mapped to standardized Cell Ontology (CL) IDs, ensuring the final artifact is a universally comprehensible, machine-learning-ready asset.
    

By establishing absolute computational provenance and replacing human assumption with structural physics, this pipeline aims to resolve the PBMC manifold.

---------
## 2 Methods

### 2.1 Pre-Phase 1: Upstream Provenance and System Ingestion Boundaries

#### 2.1.1 Data Provenance and Upstream Processing

The physical isolation of Peripheral Blood Mononuclear Cells (PBMCs) was achieved via density gradient centrifugation, isolating the viable mononuclear fraction. Single-cell encapsulation and barcoding were executed utilizing the 10x Genomics Chromium droplet-based microfluidic platform. Within the generated Gel Bead-in-emulsions (GEMs), cellular lysis and bead dissolution facilitated the capture of polyadenylated mRNA via Poly(dT) primers linked to unique Cell Barcodes and Unique Molecular Identifiers (UMIs). Following reverse transcription and Template Switch Oligo (TSO)-mediated extension, the resulting complementary DNA (cDNA) libraries were amplified and sequenced via Illumina Sequencing by Synthesis (SBS).

Primary upstream analysis was conducted by transposing binary BCL outputs into FASTQ format. Alignment to the GRCh38/hg19 human reference genome was performed utilizing the STAR splice-aware alignment algorithm, producing structurally mapped BAM files. Subsequent feature quantification intersected genomic coordinates with the corresponding Gene Transfer Format (GTF) annotation, collapsing redundant UMIs to eliminate exponential amplification bias. This generated the raw transcriptomic count matrices, structured natively in a sparse Coordinate List (COO) format with a strict `Cells x Genes` orientation.

#### 2.1.2 Local Hardware Constraints and Decontamination Bottlenecks

Standard methodologies for the computational removal of ambient RNA contamination (e.g., CellBender) rely heavily on deep convolutional neural networks and generative modeling. These architectures are mathematically optimized for execution on discrete NVIDIA CUDA tensor cores. The local computational environment for this downstream architecture is strictly constrained to an ARM64 Apple Silicon framework, which leverages unified Metal Performance Shaders (MPS). Compiling and executing CUDA-dependent epoch-training loops natively on MPS introduces significant memory bottlenecks and extended processing times, making local generative ambient RNA modeling computationally impractical.

#### 2.1.3 The Structural Ingestion Boundary

To accommodate these hardware constraints, this pipeline initiates downstream of deep-learning-based ambient RNA decontamination methods. The architecture initiates strictly with the 10x Genomics Filtered Feature-Barcode Matrix, which has undergone heuristic, rank-based UMI thresholding (the conventional "Knee Plot" inflection) to isolate high-confidence cellular barcodes from ambient background partitions.

To account for residual ambient RNA, the downstream Phase I Quality Control steps employ robust variance filters. The pipeline utilizes Median Absolute Deviation (MAD) as a robust, non-parametric distance metric, deploying an absolute 5-MAD boundary to isolate structural anomalies such as mitochondrial stress events, ribosomal collapse, and doublets. This rigorous filtering protocol ensures the statistical integrity of the count matrix prior to Phase II normalization.

The physical sample consists of PBMCs extracted from a healthy human male donor, aged 18-35 were obtained by 10x Genomics from Cellular Technologies Limited.

#### 2.1.4 Sequencing Architecture and Chemistry

Single-cell capture and library preparation were performed using the Gene Expression libraries which were generated as described in the Chromium GEM-X Single Cell 3' Reagent Kits v4 User Guide (CG000731) and sequenced on an Illumina NovaSeq 6000 with approximately 39,000 read pairs per cell for the Gene Expression library.

Paired-end, dual indexing libraries were sequenced with this configuration:

- 28 cycles Read 1
- 10 cycles i7
- 10 cycles i5
- 90 cycles Read 2

Libraries were analyzed using the `cellranger multi` pipeline.

#### 2.1.5 The Computational Anchor (Reference Genome)

The translation of raw sequencing reads into the count matrix $X \in \mathbb{R}^{C \times G}$ was performed utilizing the Cell Ranger pipeline (version 9.0.0). Crucially, the reads were aligned against the GRCh38  human reference genome. Consistent use of this reference genome is critical, as utilizing a different genomic assembly would alter the gene feature space and hinder downstream reproducibility. The resulting filtered feature-barcode matrices (`matrix.mtx`, `barcodes.tsv`, `genes.tsv`) serve as the primary input for the downstream analytical pipeline.

### 2.2 Phase 1: Transcriptomic Quality Control and Matrix Preprocessing

#### 2.2.1 Matrix Initialization and Quality Control Metrics

The foundational stage of the downstream architecture requires establishing the baseline viability and transcriptomic integrity of every individual sequenced droplet. Upon ingesting the Coordinate List (COO) matrix, the pipeline computes fundamental quality control metrics—termed biological vital signs. For each cellular vector, the engine calculates the total library size (total_counts) and the number of detected features (n_genes_by_counts).

Crucially, the architecture quantifies the proportion of reads mapping to mitochondrial and ribosomal genomes. These fractions serve as primary proxies for cellular stress and active apoptosis. During the microfluidic encapsulation process, a mechanically ruptured cell membrane will leak diffuse cytoplasmic mRNA into the ambient medium, whilst the rigid mitochondria remain trapped within the oil partition. Consequently, an artificially inflated mitochondrial read fraction (`pct_counts_mt`) indicates a compromised or non-viable cell that must be excluded prior to downstream modeling.

#### 2.2.2 Limitations of Parametric Thresholding and Stochastic Multiplets

Conventional data science pipelines frequently employ Standard Deviation (SD) to define threshold boundaries. However, SD fundamentally assumes the underlying data follows a Gaussian (normal) distribution.

Single-cell RNA sequencing data is strictly non-Gaussian; it represents a highly skewed, continuous biological gradient. Employing a mean-dependent metric like SD on skewed biological tissue allows extreme outliers (such as ambient RNA pools or cell clumps) to disproportionately skew the mean. This artificially expands the threshold bounds, resulting in the erroneous retention of dead cells and debris.

Furthermore, droplet encapsulation is a stochastic Poisson process. There is a non-zero probability that two distinct cells are co-encapsulated within a single Gel Bead-in-emulsion (GEM). These "doublets" exhibit artificially inflated complexity vectors and hybrid transcriptomic profiles that, if undetected, generate spurious continuous differentiation trajectories where none biologically exist.

#### 2.2.3 5-MAD QC Filtering and Doublet Purging

The isolation of the core biological manifold via Median Absolute Deviation (MAD), computationally simulated doublet purging, and layer architecture generation.

To robustly filter the matrix for Quality Control (QC), the architecture completely bypasses mean-centric statistics like standard deviation. The pipeline implements a strict non-parametric outlier identification algorithm utilizing Median Absolute Deviation (MAD). By calculating absolute distances from the spatial median rather than the mean, the MAD metric remains highly resistant to the skewing effects of extreme outliers. The engine enforces an absolute **5-MAD threshold**, dynamically establishing highly restrictive variance bounds for `total_counts`, `n_genes_by_counts`, and `pct_counts_mt`. Any cellular profile exceeding this 5-MAD boundary is subsequently filtered from the dataset. Features (genes) detected in fewer than three viable cells are simultaneously dropped to eliminate extreme sparsity and reduce tensor dimensionality.

To resolve the multiplet anomaly, the pipeline deploys a doublet-detection algorithm (`Scrublet`). This engine computationally simulates synthetic doublets by randomly sampling and fusing observed transcriptomes, constructs a k-Nearest Neighbors (kNN) graph, and calculates a doublet density score for every cell. High-scoring multiplets are subsequently removed from the matrix.

Finally, to prepare the tensor for topological extraction while preserving the unedited baseline data, the architecture establishes a multi-layer framework. The raw, discrete integer counts are locked immutably into a base `counts` layer. The copy of primary `.X` tensor is then subjected to a depth-normalization algorithm (scaling total counts to a target sum of $10^4$ to correct for stochastic sequencing depth) and a log-plus-one (`log1p`) transformation to stabilize variance. This engineered state is preserved in a dedicated `log1p_norm` layer of the primary anndata. The fully decontaminated, variance-stabilized matrix is then saved to disk, ensuring the subsequent dimensionality reduction engine operates exclusively on high-confidence, viable cellular profiles.

### 2.3 Phase 2: Dimensionality Reduction and Subpopulation Clustering

The fundamental objective of this analytical phase is to mathematically separate distinct biological phenotypes from a continuous, high-dimensional probability space. Operating on the quality-controlled master tensor $X \in \mathbb{R}^{C \times G}$, where the row space $C$ represents viable cellular droplets and the column space $G$ represents genomic features, the system must deduce definitive cell populations. to ensure the defined clusters represent robust biological states rather than algorithm-specific artifacts, this pipeline enforces a strict physical segregation of data, followed by iterative variance stabilization, hyperparameter optimization, and out-of-sample validation.

#### 2.3.1 Dataset Segregation and the Prevention of Data Leakage

A foundational requirement of any empirical model is its ability to predict unseen reality. If a dataset is utilized in its entirety to establish the mathematical boundaries of a clustering model, those boundaries are inherently biased by the specific topological noise of that exact dataset. In the standard analysis of single-cell transcriptomics, this principle is frequently violated by two distinct, yet mathematically linked, entropic errors: **Data Leakage** and **Double Dipping**.

**Data Leakage** is an error of dimensional foresight. It occurs when global variance structures—such as Highly Variable Genes (HVGs) or the eigenvectors of a Principal Component Analysis (PCA)—are computed on the entire dataset. If the training manifold is built upon global variance, the geometric coordinates of the validation set have already been mathematically incorporated into the training state, rendering independent validation impossible.

**Double Dipping** is a circular inference error. It occurs when an algorithm uses a specific dataset to define physical cluster boundaries, and then subsequently uses that exact same data to run Differential Gene Expression (DGE) tests between those boundaries. Because the transcripts driving the clustering are inherently the ones being tested for statistical significance, the mathematics will automatically force false-positive marker genes, yielding false-positive marker genes and driving spurious cluster formation.

To mathematically prevent this violation, the data partitioning protocol strictly splits the dataset. Upon loading the standardized, quality-controlled expression matrix, the cellular barcodes are bisected into two independent, non-overlapping subsets: $X_{train}$ and $X_{project}$. This separation is executed using a stratified random split (allocating 50% of the data to each tensor) locked by a global pseudo-random seed to guarantee exact computational reproducibility. All subsequent identification of highly variable features, calculation of dimensional manifolds, and definitions of physical cluster boundaries are restricted entirely to the $X_{train}$ tensor. The $X_{project}$ matrix is completely withheld from analysis until the final validation phase

#### 2.3.2 Mathematical Instruments of Topological Resolution

Before executing the analytical pipeline, we must explicitly define the mathematical instruments utilized to process the $X_{train}$ tensor. Each module is engineered to mitigate specific forms of technical noise, substituting heuristic approaches with robust quantitative frameworks.

**I. Cell Cycle Confounder Evaluation**

Cellular division is a massive driver of transcriptional variance. If unmonitored, the clustering algorithm will erroneously partition a single, homogenous immune population into distinct clusters based purely on whether the cells are in the resting (G0/G1), DNA synthesis (S), or mitotic (G2/M) phases. To prevent this spatial distortion, physical expression scores for canonical S-phase and G2M-phase genes are calculated for each cell. By projecting these scores onto the foundational coordinate space, we visually and quantitatively audit the manifold. If the primary topological axes align precisely with cell cycle progression rather than immune phenotype, mathematical regression of these vectors is strictly mandated.

**II. Analytic Variance Stabilization (NPR, HVG, and PCA Recalibration)**

Raw transcript counts are heavily confounded by the arbitrary total sequencing depth of each individual droplet. Traditional logarithmic normalization often fails to stabilize variance in highly sparse datasets, artificially inflating the weight of lowly expressed, noisy transcripts. To isolate the true biological signal, this pipeline utilizes Analytic Pearson Residuals. This transformation models the expected expression of a transcript based strictly on the physical sequencing depth of the cell and the global relative abundance of the gene across the population. The residual $r_{i,j}$ for cell $i$ and gene $j$ is calculated as:

$$r_{i,j} = \frac{x_{i,j} - \mu_{i,j}}{\sqrt{\mu_{i,j} + \frac{\mu_{i,j}^2}{\theta}}}$$

Here, $x_{i,j}$ is the observed count, $\mu_{i,j}$ is the expected count under a null model of uniform distribution, and $\theta$ is the overdispersion parameter. In single-cell RNA-sequencing, biological variance scales quadratically with the mean expression. Rather than estimating this arbitrarily, $\theta$ is strictly locked to 100, adhering to the empirical consensus established by the Seurat `SCTransform` framework (Hafemeister & Satija). This precise threshold prevents ultra-abundant transcripts from exhibiting infinite variance and distorting the Euclidean space.

Furthermore, to ensure the resulting low-dimensional geometry maps functional biological states rather than cellular mortality, transcripts corresponding to mitochondrial and ribosomal structures are algorithmically exiled from the Highly Variable Gene pool. While apoptosis generates a massive variance signal, it represents a dying state, not a stable immune phenotype. The remaining non-apoptotic, highly variable features are then projected into a lower-dimensional space via an ARPACK-solved Principal Component Analysis (PCA).

**III. Terminal State Assessment (Automated Over-Clustering Prevention)**

A critical, often ignored flaw in recursive clustering algorithms is over-clustering—their inability to naturally halt, continuing to partition homogenous noise if forced by the user. We must mathematically define the boundary between a true biological sub-population and an amorphous cloud of technical variance.

In Principal Component Analysis (PCA), random thermal noise generates variance eigenvalues up to a theoretical limit described by the Marchenko-Pastur distribution. An eigenvalue that is only 1.8 or 2.0 times larger than the baseline noise often represents Tracy-Widom fluctuations—random mathematical chance rather than physical reality. To autonomously detect true structures, we calculate the Structural Energy Ratio: the variance of the primary axis of separation (PC1) divided by the median baseline thermal noise of the matrix. We enforce a strict 5-sigma eigenvalue threshold of 3.5. If the structural ratio is $\ge 3.5$, an angular structural divergence (an "elbow") exists, confirming a true sub-population. If the ratio falls below 3.5, the geometry is classified as isotropic, and the module permanently locks the sub-population as a "Terminal State," halting further algorithmic subdivision.

**IV. Hyperparameter Optimization for Cluster Stability**

Graph-based community detection algorithms (like Leiden) rely heavily on two highly sensitive parameters: the scaffolding limit ($k$-nearest neighbors) and the thermodynamic resolution ($r$). The standard industry practice of selecting these scalars via iterative "hit and trial" introduces severe human bias. To eradicate this flaw, we execute a comprehensive Hyperparameter Grid Search to map Cluster Stability.

To eradicate this flaw, we map the stability of the matrix by computing the structural Modularity ($Q$) across an extensive Cartesian grid of $k$ and $r$ coordinates. This generates a definitive matrix of resulting cluster counts. Instead of guessing, the algorithm searches the parameter space for an optimal stability plateau—a spatial block (e.g., $2 \times 2$ or larger) where the number of discrete clusters remains entirely static despite aggressive perturbations in both $k$ and $r$. By generating contour and heatmaps of this surface, we identify the physical centroid of this stable plateau. This mathematically collapses an infinite plane of "trial and error" guesswork down to 2 or 3 highly calculated, deterministic parameter pairs.

**V. Cluster Stability Bootstrapping (Jaccard Overlap)**

Even optimized boundaries must survive structural destruction to be considered biologically real. Upon generating a baseline K-Nearest Neighbors graph, the pipeline executes a rigorous bootstrap diagnostic. 20% of the cells are randomly removed, and the community detection process is recalculated from scratch across multiple iterations. A Jaccard Overlap score is calculated between the original boundary and the newly generated boundary. We strictly enforce that a mean Jaccard score $>0.85$ indicates high topological stability, while scores between $0.62$ and $0.85$ indicate moderate stability requiring extreme biological caution during annotation.

#### 2.3.3 Global Lineage Resolution (Macro-State Execution Sweep)

With the mathematical instruments defined, the active execution of the $X_{train}$ tensor begins. The objective of the Macro-State Sweep is to resolve the primary, overarching lineages within the PBMC population (e.g., distinguishing the entire T-cell compartment from the Myeloid compartment).

The global matrix undergoes variance stabilization via Analytic Pearson residuals, and the top Highly Variable Genes are isolated to construct the primary PCA manifold. A hyperparameter grid search evaluates a broad parameter space ($k \in [5, 105]$, $r \in [0.01, 0.21]$) to lock the optimal macro-resolution. Once these primary boundaries are verified by the Jaccard bootstrapping diagnostic, the global matrix is partitioned. The system subsets the original $X_{train}$ object into multiple independent sub-tensors, each representing a single, distinct macro-cluster, and saves these isolated matrices to disk for independent processing.

#### 2.3.4 Recursive Subclustering

The variance required to differentiate subtle micro-phenotypes (e.g., a naive CD4+ T-cell versus a memory CD4+ T-cell) is mathematically orthogonal to, and vastly smaller than, the variance that separates major macro-lineages. In a global PCA, this critical micro-variance is crushed into the discarded lower dimensions.

Therefore, each isolated macro-cluster matrix is ingested individually to reconstruct its internal geometry from the ground up. To prevent over-fracturing the data and degrading the geometric density, this pipeline strictly bounds the recursion depth to a two-tier architecture: one Macro-level separation, followed by exactly one Micro-level separation. The global highly variable genes are discarded. The Analytic Pearson Residuals are recalculated specifically on the isolated sub-population, exposing a new, highly specialized subset of local HVGs. A completely new, localized PCA space is computed.

Before further subdivision occurs, the Terminal State Assessment is executed. If the Structural Energy Ratio is $< 3.5$, the compartment is recognized as a homogenous state and safely archived. If the ratio is $\ge 3.5$, a high-resolution micro-Mesa audit is executed to find the exact boundaries of the sub-populations. This loop recurs until every physical compartment of the data reaches an isotropic terminal state, leaving no biological stone unturned.

#### 2.3.5 The Out-of-Sample Projection Audit

The final phase seals the methodology by proving the universal generalizability of the learned topological boundaries. The $X_{project}$ matrix, held in stasis since ingestion, is introduced.

A standard, critical flaw in validation pipelines is allowing the test set to normalize itself. If $X_{project}$ is permitted to calculate its own internal mean and variance for normalization, the testing space has mathematically influenced itself, resulting in data leakage. To strictly enforce FAIR and CRISP-DM validation protocols, $X_{project}$ is fundamentally barred from defining its own parameters. It is normalized using the global gene probabilities ($p_j$) derived strictly from the $X_{train}$ tensor, multiplied by its own local cell depths to calculate the expected variance.

These rigidly normalized validation vectors are then projected onto the pre-computed PCA eigenvectors of the training space. Using the finalized K-Nearest Neighbor architecture, the validation cells are classified based entirely on their geometric proximity to the training boundaries. By demonstrating that the mathematical boundaries constructed on the training universe seamlessly organize the strictly segregated data of the testing universe, the methodology demonstrates robust quantitative generalizability.

### 2.4 Phase 3: Differential Marker Identification and Lineage Validation

Following the mathematical resolution of topological boundaries, the system must assign biological identities to the identified clusters. This phase dictates the extraction of defining genomic features (marker genes) for each isolated cluster. To ensure absolute empirical validity, the pipeline utilizes a combination of statistical filtering, adaptive thresholding, and negative marker assessment to ensure cluster purity.

#### 2.4.1 Statistical Framework for Feature Identification

Before orchestrating the extraction across the data hierarchy, we establish the specific mathematical instruments designed to isolate the true biological signal from background transcriptional noise.

**I. Statistical Filtering and Cluster Size Constraints**

The fundamental axiom of phenotype identification is that a cellular state is defined by transcripts that are differentially upregulated relative to the surrounding environment. However, executing statistical tests on microscopic, unstable clusters mathematically guarantees the amplification of noise. To resolve this, the system first executes a spatial prune: any topological state containing fewer than 10 cellular vectors is excluded from differential expression analysis to avoid statistical bias associated with low-n clusters. For the remaining viable states, a Wilcoxon Rank-Sum test is executed across the log-normalized layer. To prevent the capture of biologically irrelevant noise, the resulting output is filtered using standardized thresholds: only genes exhibiting an adjusted p-value $< 0.05$ and a strict $\text{Log}_2\text{FoldChange} > 1.0$ (representing a minimum two-fold absolute biological up-regulation) are permitted to proceed.

**II. Adaptive Thresholding for Differential Gene Expression (DGE)**

Standard Differential Gene Expression (DGE) tests often return thousands of statistically significant genes for massive macro-clusters, complicating the identification of primary lineage-defining markers. Conversely, highly specific micro-lineages may return almost zero genes under universal thresholds. To resolve this parameter collapse, the system applies an Elastic Distillation algorithm.

For robust clusters, the system isolates only the genes existing in the 93.75th percentile ($Q_{93.75}$) of statistical significance (calculated via negative log-adjusted p-values). However, if a fragile micro-state breaches the "Starvation Limit" (yielding fewer than 3 surviving genes under the $Q_{93.75}$ cut), the system autonomously lowers the local threshold to capture a Minimum Viable Payload (MVP) of exactly 3 markers. Once distilled, these markers are rigorously sorted not just by magnitude, but by their _Violin Delta_—the mathematical difference between the percentage of cells expressing the gene in the target cluster versus the background reference. This guarantees the selection of highly specific, cluster-defining markers (highly prevalent in the target, absent elsewhere) rather than universally expressed genes that simply experienced a slight up-regulation.

**III. Negative Marker Assessment and Purity Validation**

A critical flaw in standard single-cell annotation is the reliance on positive identification alone, which masks doublet contamination. To enforce Negative Marker Validation, we recognize that identifying T-cell markers in a cluster does not prove it is a pure T-cell population; it could be a doublet population physically contaminated by Macrophages. A true biological cell is defined just as much by the epigenetic programs it has silenced as by the ones it expresses.

To definitively prove cluster purity, the pipeline evaluates the absence of markers characteristic of non-target lineages. It systematically assesses markers from alternative lineages to verify transcriptomic purity and the absence of cross-contamination. If foreign lineage markers manifest strongly within the target state, topological contamination is flagged.

**IV. Algorithmic Reference Mapping**

Manual annotation can be influenced by the subjective interpretation of existing literature. To establish an unbiased empirical baseline, the system incorporates CellTypist, an automated supervised machine learning framework. Using pre-trained, high-fidelity immune models (`Immune_All_High` and `Immune_All_Low`), the system executes a majority-voting algorithm across the Leiden boundaries. This provides an algorithmic hypothesis for every cluster, providing an independent reference to validate the statistically derived markers.

#### 2.4.2 Implementation of the Marker Identification Pipeline (Phase III Execution)

The execution engine ingests the rigorous mathematical state dictionary (JSON) generated during Phase II, ensuring data traceability. The analysis is performed hierarchically as follows:

1. **Macro-State Marker Identification:** The global $X_{train}$ tensor is ingested, and its anchored macro-boundaries are subjected to the Wilcoxon extraction engine. Ghost states are pruned, elastic thresholds are applied, and the top definitive markers are isolated and rendered. The global structure is concurrently mapped via the automated CellTypist `Immune_All_High` model.
    
2. **Micro-State Resolution:** The orchestrator iterates through the physical file paths of every isolated micro-tensor. For each validated micro-state, the extraction process is re-initialized within its local geometric space, mapping the high-resolution features against the CellTypist `Immune_All_Low` model. If a subset was previously locked as an isotropic Terminal State in Phase II, extraction is correctly bypassed, and it inherits the parent topology's markers.
    
3. **Cross-Validation and Wide-Span Auditing:** Every extracted micro-state is systematically subjected to the Epigenetic Absence Cross-Validation to prove spatial purity. Concurrently, the states are validated against a pre-curated canonical JSON ledger of established immune markers to align the empirical findings with accepted biological literature.
    
4. **Metadata Ledger Generation:** The final step involves generating standardized JSON templates to facilitate the manual assignment of biological labels and Cell Ontology IDs (`annotation_manual.json` and `ontology_cl_id_manual.json`). These ledgers map the exact mathematical boundaries discovered in the data, strictly formatting the environment for human annotation and guaranteeing that the final biological conclusions remain immutably linked to the underlying tensor coordinates.

### 2.5 Phase 4: Metadata Annotation, Ontological Alignment, and Dataset Integration

The final phase of the analytical pipeline bridges the gap between cluster coordinates and biological identities. It is at this stage that the rigorous quantitative evidence generated in previous phases is synthesized into definitive cellular identities. To ensure absolute FAIR (Findable, Accessible, Interoperable, Reusable) compliance, the system avoids the direct modification of core data structures. Instead, a programmatic metadata integration protocol, guided by external JSON ledgers, generates a unified dataset optimized for machine learning.

#### 2.5.1 Framework for Metadata Integration

Before detailing the orchestration, the specific mechanical safeguards enforcing annotation validity must be defined.

**I. Independent Metadata Mapping via JSON Ledgers**

A primary flaw in standard analytical pipelines is the "black-box" assignment of biological labels directly within active memory. When an analyst assigns a label based on subjective visual inspection of markers, the reasoning is lost as soon as the code terminates, violating the Reusability and Interoperability standards of the FAIR protocol.

To resolve this, the system separates the metadata mapping from the primary data processing. The system strictly ingests two external JSON ledgers: `annotation_manual.json` and `ontology_cl_id_manual.json`. Crucially, these are the exact structured, but functionally empty, taxonomic ledgers generated during the terminal step of the previous marker extraction phase. They ensure consistent data traceability. The system maps the string values populated within these ledgers strictly to the numeric Leiden keys of the isolated physical matrices. Furthermore, to satisfy rigorous Interoperability, the system enforces the mapping of standardized Cell Ontology (CL) IDs alongside colloquial cell names, ensuring the resulting data can be ingested by any global database without semantic ambiguity.

**II. Hierarchical Annotation Inheritance**

The physical fracturing of the matrix generated a nested taxonomy of both active states (micro-lineages undergoing active clustering) and locked states (Terminal States deemed isotropic). The annotation engine must navigate this spatial hierarchy seamlessly. For active states, labels are mapped directly via their localized topological keys. However, for clusters previously locked as Thermodynamic Terminal States, the local key is mathematically non-existent. The algorithm is engineered to autonomously identify terminal clusters and assign labels based on parent-level lineage definitions. It traces the lineage back up the Directed Acyclic Graph (DAG) and injects the phenotype defined at the macro-level, ensuring all cells are correctly assigned within the hierarchical structure.

#### 2.5.2 Data Integration and Final Dataset Generation (Phase IV Execution)

The execution of the annotation engine operates in a strict three-stage protocol designed to prevent barcode duplication and ensure consistent label assignment.

1. **Training Set Integration and Ledger Assembly (The $X_{train}$ Matrix):** The system loads the state dictionary for the primary training matrix ($X_{train}$). It iterates through every isolated macro and micro tensor currently on disk. For each physical file, the algorithm queries the human-populated JSON ledgers and injects the precise biological identities and CL IDs. Once localized mapping is complete, the engine generates a mapping of cell barcodes to biological identities and CL IDs `[cell_barcode, manual_label, human_CL_ID]` from each sub-matrix. These vectors are strictly aggregated and concatenated into a singular, central CSV ledger (`master_labels_df.csv`).
    
2. **Validation Set Integration (The $X_{project}$ Matrix):** The orchestrator then ingests the state dictionary for the out-of-sample projection matrix ($X_{project}$). It repeats the identical localized injection process. When appending the resulting validation vectors to the central CSV ledger, the system performs a deduplication check. If a barcode collision is detected (a mathematically impossible event under the current architecture, but a mandatory CRISP-DM robustness safeguard), the system isolates the conflict and prioritizes the most recent RAM state, permanently sealing the universal ledger of cell identities.
    
3. **Global Dataset Integration:** The ultimate goal of the data mining lifecycle is the creation of a definitive, deployment-ready asset. The orchestrator ingests the raw, un-split, quality-controlled global matrix and the finalized central CSV ledger. Utilizing a highly optimized, vectorized `.map()` operation, the biological identities and CL IDs are projected from the CSV ledger directly onto the coordinates of the global tensor. Cells without definitive labels (due to prior quality control filtering) are assigned a status of 'Unknown/Filtered'. The fully integrated, universally labeled object is then serialized to disk as the final ML-Ready artifact (`_ML_Ready.h5ad`), completing the analytical pipeline.

### 2.6 Streamlit API:  Development of an Interactive Interface for Human-in-the-Loop Analysis

The ultimate objective of the CRISP-DM life-cycle is the deployment of a validated, usable model. In high-resolution bioinformatics, automated pipelines can be limited by their inability to resolve biological edge cases during the deployment phase. To resolve this, the system culminates in a graphical Orchestration Engine (built upon the Streamlit framework). The interface serves as a bridge between automated computational outputs and the expertise of the human analyst. It is engineered to manage massive memory loads, halt execution at critical decision nodes, and guarantee FAIR-compliant data ingestion and extraction.

#### 2.6.1 The Reactive Memory Constraint and Tensor Pinning

A fundamental physical limitation of reactive UI frameworks like Streamlit is their stateless architecture; the application resets its internal state and re-executes the script from top to bottom upon every user interaction (e.g., a button click). In single-cell transcriptomics, reading a massive `.h5ad` expression tensor from the physical disk takes significant time and RAM. If the application attempts to reload the matrix from disk on every reactive cycle, it risks exceeding memory capacity and causing application instability.

To address these resource constraints, the orchestration engine utilizes two distinct memory-preservation instruments:

1. **Resource Caching (Tensor Pinning):** The system isolates the `load_tensor` function from the reactive execution loop using a static cache decorator (`@st.cache_resource`). When the matrix is loaded once, it is permanently pinned in the server's RAM. Subsequent UI interactions simply reference this existing memory block rather than triggering redundant, heavy disk I/O operations.
    
2. **Session State Management:** Transient states—such as the human-approved $k$ and $r$ coordinates, or the active queue of micro-lineages waiting to be processed—are strictly registered into a protected dictionary (`st.session_state`). This facilitates a persistent state that is maintained across subsequent UI interactions of the UI, allowing the user to progress sequentially through the Directed Acyclic Graph without losing their previous structural decisions.
    

#### 2.6.2 Integrated Iterative Validation Breakpoints

Standard analytical pipelines operate as impenetrable black boxes, ingesting raw data and blindly outputting final clusters without intermediate validation. This explicitly violates the requirement for iterative evaluation.

This engine is engineered with "Temporal Airlocks." Instead of running Phase II (Clustering) as a single continuous script, the orchestrator divides the execution into a dynamic queue.

1. **Macro-Level Optimization:** The interface executes the initial sweep and pauses to allow the analyst to review the stability metrics before finalizing parameters.
    
2. **The Micro-Queue:** Once the Macro-State is locked and the matrix is fractured, the resulting sub-matrices are pushed into an active Python list (the `micro_queue`). The engine loops through this queue one lineage at a time, evaluate the localized stability for each micro-state and allow for manual parameter input. By forcing the system to pause at these critical nodes, the engine transforms an automated script into a fully auditable, step-wise scientific instrument.
    

#### 2.6.3 Visual Telemetry and Spatial DOM Isolation

To allow the human analyst to make informed parameter decisions, the geometric evidence (scatter plots, dendrograms, stability surfaces) must be visualized within the interface. However, rendering mathematical Scalable Vector Graphics (SVGs) directly into a web interface can lead to layout inconsistencies, unconstrained SVGs will overflow their containers and overlap with the UI controls.

The visualization module addresses this by standardizing the image rendering protocol. Instead of merely linking to the image, the engine opens the physical `.svg` file and utilizes regular expressions to modify the XML attributes, ensuring the height and width are responsive within the interface. to force `width="100%"` and `height="auto"`. The rewritten XML is encoded into a Base64 string and injected directly into a rigidly styled CSS flexbox container. This ensures that regardless of the physical dimensions of the manifold generated by the backend engine, the visual evidence is perfectly constrained and isolated within the browser's Document Object Model (DOM).

#### 2.6.4 Workspace Initialization and Environment Reset

The integrity of a scientific workspace is compromised if residual data from previous experiments contaminates the current matrix. To guarantee a completely sterile execution environment, the Control Room features an "Entropy Purge" mechanism. Upon activation, the application resets the workspace by utilizing the `shutil.rmtree` command to recursively delete all physical data directories, topological artifacts, and temporary RAM caches. It then mathematically reconstructs the empty directory scaffolding, ensuring a consistent baseline for subsequent data processing.

Finally, the Annotation Engine tab provides the interface for Phase IV. Instead of forcing the analyst to write code to label cells, the system parses the empty JSON taxonomies generated in Phase III and renders them as an interactive, two-dimensional dataframe. The user inputs their biological deductions and standardized Cell Ontology (CL) IDs directly into the matrix. Upon validation, the system records these labels to disk as standardized JSON files as JSON files, immediately triggering the Recombination engine to weave these human labels into the final ML-Ready tensor. This strict separation between human UI input and backend tensor mutation guarantees absolute computational provenance.

------------
## 3 Results

The downstream PBMC analysis pipeline was applied to the `5k_Human_Donor1_PBMC_3p_gem-x` dataset. The workflow was executed using automated parameter selection for dimensionality reduction and clustering. The results of the analysis are detailed below.

### 3.1 Data Ingestion and Quality Control

Initial quality control (QC) and data filtering (`p03`) were performed to remove technical artifacts prior to downstream clustering. The raw transcriptomic dataset was filtered to remove technical noise and retain high-quality cells.

**Initial State:**

The raw sequencing data generated an initial matrix consisting of `[5710]` cells and `[38606]`features (genes).

**Technical Filtering:**

Standard QC metrics were used to filter out low-quality cells:

- **Library and Feature Complexity:** Cells exhibiting extreme deviations in total transcript abundance or unique gene counts were filtered using a threshold of `5-MAD` (Median Absolute Deviation) and  of genes whose expressions were in less than 3 cells.
	
- **Mitochondrial Fraction:** To exclude dead, dying, or stressed cells that contribute to ambient RNA contamination, cells with a mitochondrial transcript fraction exceeding `5-MAD` were removed.
	
- Removing total of 377 cells and 13741 genes

**Doublet Removal:**

Following initial QC, computational doublet detection (`doublet purge`) was executed to identify and remove `[11]` predicted multiplets. This step is critical to prevent the clustering algorithm from generating false hybrid-states between distinct biological lineages.

**Final QC State and Data Split:**

Following filtering and doublet removal, the final dataset comprised `[5322]` high-quality cells and `[24865]` genes. To prevent data leakage and circular inference (double dipping) during downstream marker extraction, the filtered matrix was randomly bisected (`random_split_data`, seed 42), allocating 2,661 cells to the training set ($X_{train}$) and 2,661 cells to the projection holdout set ($X_{project}$).

![[results/figures/p03_qc_filtering/scatter_pre_filter.svg]]
![[results/figures/p03_qc_filtering/violin_pre_filter.svg]]

![[results/figures/p03_qc_filtering/scatter_post_filter.svg]]
![[results/figures/p03_qc_filtering/violin_post_filter.svg]]
![[results/figures/p03_qc_filtering/scrublet_score_distribution_doublet_distribution.svg]]


### 3.2 Dimensionality Reduction and Macro-Clustering

Following initial quality control, the dataset was processed to define the primary biological lineages. The workflow executed a sequential dimensionality reduction followed by graph-based community detection.

**Feature Selection and Dimensionality Reduction:**

Prior to Principal Component Analysis (PCA), cell cycle scoring (`cell_cycle_check`) was performed to assess whether cell cycle phases were confounding transcriptomic variation. The maximum observed scores (S-phase max = `[0.6]`, G2M-phase max = `[0.4]`) indicated minimal phase-driven bias. To ensure clustering was driven by cell identity rather than technical or metabolic variance, `[1]` mitochondrial and ribosomal genes were explicitly excluded from the Highly Variable Gene (HVG) pool before it was subjected to PCA, which was computed on the remaining `[2499]` HVGs. An evaluation of the variance explained by each principal component (`evaluation of the PCA variance ratio / elbow plot`) determined that `[10]` principal components were optimal for capturing the biological signal while minimizing computational noise.

![[results/figures/p04_clustering/umap_training_cell_cycle.svg]]
![[results/figures/p04_clustering/pca_variance_ratio_training_file_.svg]]

**Graph-Based Clustering:**

A k-nearest neighbor (kNN) graph and Uniform Manifold Approximation and Projection (UMAP) embeddings were generated (`knn_umap_leiden`). To define robust macro-cluster boundaries objectively, a systematic grid search of the nearest neighbors ($k$) and Leiden resolution ($r$) parameters was executed (`macro sweep`). By evaluating clustering stability across iterations (`mesa_audit`) during visual inspection of the parameter space via heat-map and contour plots confirmed this region as highly stable, consistently yielding `[5]` distinct macro-clusters, the optimal parameters were identified as $k=$ `[42]` and $r=$ `[0.16]`. This partitioned the dataset into `[5]` distinct macro-clusters.

![[results/figures/p04_clustering/macro_thermodynamic_surface.svg]]
![[results/figures/p04_clustering/macro_umap.svg]]


**Macro-Cluster Stability Evaluation:**

To assess the robustness of the resulting cluster boundaries, a bootstrapping protocol (`jaccard scores`) was applied using 20% data subsampling across repeated iterations. All macro-clusters exceeded the minimum structural stability thresholds. The mean Jaccard survival indices were:

- **Cluster 0:** `[0.779]` and 1228 cells
    
- **Cluster 1:** `[0.955]` and 484 cells
    
- **Cluster 2:** `[0.697]` and 399 cells
    
- **Cluster 3:** `[0.890]` and 294 cells
    
- **Cluster 4:** `[0.948]` and 256 cells
   
Visual inspection of the resulting projection overlap diagnostic plots confirmed a high degree of structural alignment between the reference ($N=2,661$) and projected ($N=2,661$) coordinates. This exact overlap physically demonstrates that the defined micro-cluster boundaries are highly stable and accurately represent the underlying biological topology.
![[results/figures/p04_clustering/macro_leiden_Projection_Overlap.svg]]
### 3.3 High-Resolution Micro-Clustering

The `[5]` isolated macro-clusters were independently sub-clustered to identify finer cellular subpopulations (`micro sweep`). Parameter optimization (grid searches for $k$ and $r$) was performed iteratively for each macro-cluster, yielding a total of `[27]` distinct sub-clusters across the training dataset.

Bootstrapping (`jaccard scores`) was re-applied at the micro-level to evaluate boundary stability. The variance in Jaccard scores successfully differentiated between discrete terminal cell types (high stability, $>0.85$) and continuous developmental trajectories (moderate stability, $0.65 - 0.80$):

- **Macro-State 0:** Locked at $k=$ `[33]`, $r=$ `[0.46]`. Yielded `[8]` sub-clusters. Minimum observed Jaccard stability: `[0.658]`.
    ![[results/figures/p04_clustering/pca_variance_ratio_macro_leiden_0_.svg]]
    ![[results/figures/p04_clustering/macro_leiden_0_thermodynamic_surface.svg]]
    ![[results/figures/p04_clustering/macro_leiden_0_micro_umap.svg]]
    ![[results/figures/p04_clustering/macro_leiden_0_micro_leiden_Projection_Overlap.svg]]
- **Macro-State 1:** Locked at $k=$ `[32]`, $r=$ `[0.68]`. Yielded `[5]` sub-clusters. Minimum observed Jaccard stability: `[0.797]`.
    ![[results/figures/p04_clustering/pca_variance_ratio_macro_leiden_1_.svg]]
    ![[results/figures/p04_clustering/macro_leiden_1_thermodynamic_surface.svg]]
    ![[results/figures/p04_clustering/macro_leiden_1_micro_umap.svg]]
    ![[results/figures/p04_clustering/macro_leiden_1_micro_leiden_Projection_Overlap.svg]]
- **Macro-State 2:** Locked at $k=$ `[35]`, $r=$ `[0.44]`. Yielded `[3]` sub-clusters. Minimum observed Jaccard stability: `[0.743]`.
    ![[results/figures/p04_clustering/pca_variance_ratio_macro_leiden_2_.svg]]
    ![[results/figures/p04_clustering/macro_leiden_2_thermodynamic_surface.svg]]
    ![[results/figures/p04_clustering/macro_leiden_2_micro_umap.svg]]
    ![[results/figures/p04_clustering/macro_leiden_2_micro_leiden_Projection_Overlap.svg]]
- **Macro-State 3:** Locked at $k=$ `[50]`, $r=$ `[1.2]`. Yielded `[7]` sub-clusters. Minimum observed Jaccard stability: `[0.660]`.
    ![[results/figures/p04_clustering/pca_variance_ratio_macro_leiden_3_.svg]]
    ![[results/figures/p04_clustering/macro_leiden_3_thermodynamic_surface.svg]]
    ![[results/figures/p04_clustering/macro_leiden_3_micro_umap.svg]]
    ![[results/figures/p04_clustering/macro_leiden_3_micro_leiden_Projection_Overlap.svg]]
    
- **Macro-State 4:** Locked at $k=$ `[25]`, $r=$ `[0.5]`. Yielded `[4]` sub-clusters. Minimum observed Jaccard stability: `[0.756]`.
	![[results/figures/p04_clustering/pca_variance_ratio_macro_leiden_4_.svg]]
	![[results/figures/p04_clustering/macro_leiden_4_thermodynamic_surface.svg]]
	![[results/figures/p04_clustering/macro_leiden_4_micro_umap.svg]]
	![[results/figures/p04_clustering/macro_leiden_4_micro_leiden_Projection_Overlap.svg]]

### 3.4 Differential Gene Expression and Marker Extraction

Following clustering, differential gene expression analysis was performed to identify cluster-specific marker genes for cell type annotation.

**Differential Expression Analysis:** Differential gene expression (DGE) profiling (`rank_genes_group`) was executed across the validated topology using the Wilcoxon rank-sum test. For each defined cluster, candidate marker genes were evaluated using a dual-metric approach: statistical significance via FDR-adjusted p-values (`pvals_adj`) and spatial expression exclusivity (`violin_delta`). To be considered as a candidate marker, a gene was required to meet a baseline significance threshold of an adjusted $p$-value < `[0.05]` and a minimum $\log_{2}$ fold-change of `[1.0]`. The most discriminative features (`top_genes`) were extracted to define the transcriptomic signature of each cluster.

**Cluster Filtering for Statistical Robustness:** To prevent false-positive annotations resulting from low cell counts or extreme data sparsity, an adaptive filtering threshold (`elastic threshold`) was applied to the DGE outputs. Clusters failing to meet the minimum threshold for cellular representation ($N \ge$ `[10]` cells) or lacking sufficient statistical confidence in their marker profiles were flagged as non-viable. This filtering retained  total of `[24]` high-confidence (`viable_states`) sub-clusters for downstream annotation.

**Marker Extraction Metrics:**

- **Major Cell Lineages:** Distinct macro-clusters exhibited strong transcriptional divergence. Marker genes defining these major lineages recorded adjusted p-values approaching the computational minimum ( `[1.0e324]`) and high spatial exclusivity, with `violin_delta` scores exceeding :

	- **Cluster 0:** `[0.544]`
	    
	- **Cluster 1:** `[0.915]`
	    
	- **Cluster 2:** `[0.135]` 
	    
	- **Cluster 3:** `[0.560]` 
	    
	- **Cluster 4:** `[0.681]`
    
- **Sub-cluster Resolution:** Within closely related sub-clusters, highly specific markers demonstrated reduced `violin_delta` scores due to shared baseline expression. Marker genes distinguishing these related sub-types recorded `violin_delta` values ranging from: 
    
	- **Cluster 0:**  `[0.144]` to `[0.799]`
	    
	- **Cluster 1:** `[0.235]` to `[0.877]`
	    
	- **Cluster 2:** `[0.388]` to `[0.807]`
	    
	- **Cluster 3:** `[0.277]` to `[0.954]`
	    
	- **Cluster 4:** `[0.225]` to `[0.527]`
    

**Independent Marker Validation:** Because standard DGE statistics rely on relative contrast, which can obscure shared functional markers in continuous sub-lineages, candidate signatures were subjected to an independent validation protocol (`absence_cross_validation`). The absolute expression profiles of canonical lineage markers, curated from `[ Teichmann Lab. (2023). _Basic_celltype_information.xlsx_ [Data file]. CellTypist Wiki: Pan-Immune CellTypist Atlas v2. Available at: https://github.com/Teichlab/celltypist_wiki/blob/main/atlases/Pan_Immune_CellTypist/v2/tables/Basic_celltype_information.xlsx.]`, were quantitatively and visually evaluated across all clusters (`wide_span_plots`). This orthogonal cross-validation confirmed the true biological presence or absence of specific genes, successfully resolving ambiguous cell states where continuous background expression locally suppressed primary DGE significance.

### 3.5 Phenotype Annotation and Tensor Recombination

Following marker gene extraction, biological identities were assigned to the identified sub-clusters. These annotations were then mapped to standardized ontologies, and the partitioned datasets were recombined into a single, fully annotated object.

#### 3.5.1 Phenotype Annotation and Cell Ontology Mapping

Human-derived biological annotations, guided by the statistical marker profiles (Section 3.4), were systematically injected into the dataset metadata (`orc_annotation`). To ensure interoperability and standardization, all micro-state annotations were strictly mapped to the formal Cell Ontology (CL) reference structure.
![[results/figures/p05_top_markers/matrixplot__macro_leiden_top_genes.svg]]
![[results/figures/p05_top_markers/dotplot__macro_leiden_top_genes.svg]]

The hierarchical annotation of the dataset is detailed below:

**Macro-Cluster 0: `[T-Cell Lineage (CL:0000084)]**`

- **Macro Decision Factor:** `[Universal expression of the canonical T-cell transcription factor BCL11B, coupled with high baseline expression of IL7R and CAMK4. The cluster was topologically partitioned from Macro-Cluster 2 despite shared baseline transcription, evidenced by a moderate (0.4 mean) expression of CCR4, SNED1, and NECTIN3.]`.
	    ![[results/figures/p05_top_markers/matrixplot__macro_leiden_0_micro_leiden_top_genes.svg]]
	    ![[results/figures/p05_top_markers/dotplot__macro_leiden_0_micro_leiden_top_genes.svg]]
	    ![[results/figures/p05_top_markers/matrixplot__absence_audit_macro_0.svg]]
	    ![[results/figures/p05_top_markers/matrixplot__curated_genes_audit_widespan_macro_leiden_0_micro_leiden.svg]]
	    
    - **Micro-Cluster 0.0:** `[Naive T-Cell (CL:0000898)]`
        
        - **Decision Factor:** `[Retention of the foundational T-cell manifold signature (high CAMK4, moderate BCL11B) establishes structural continuity with the parent macro-cluster. The discrete mathematical boundary of this sub-state is driven by the exclusive, high-level co-expression of TRABD2A and FHIT, tightly coupled with NELL2—a canonical marker of immunologically uncommitted lymphocytes. Supported by high absolute expression of the homeostatic survival receptor IL7R, this signature definitively isolates the resting, Naive T-Cell compartment.]`.
            
    - **Micro-Cluster 0.1:** `[Early Activated / Proliferating T-Cell (CL:0000899)]`
        
        - **Decision Factor:** `[Anchored to the parent T-cell manifold via high CAMK4 and moderate BCL11B, this sub-state maintains strict topological adjacency to the naive pool (Micro-Cluster 0.0) through sustained homeostatic IL7R expression and residual NELL2 transcription. However, the discrete mathematical boundary isolating this cluster is driven entirely by a functional metabolic shift. The high, exclusive upregulation of DHFR—an enzyme physically required for de novo nucleotide biosynthesis and clonal expansion—coupled with the cytoskeletal remodeling factor ARHGEF38 and the transcript ELOA-AS1, definitively isolates a state of early cellular activation and active proliferation diverging from the resting naive baseline.]`.
            
	- **Micro-Cluster 0.2:** `[Transitional / Memory T-Cell (CL:0000813)]`
        
		-  **Decision Factor:** `[Topological divergence from the naive state is mathematically evidenced by the transcriptomic attenuation of the foundational homeostatic anchors, with IL7R and CAMK4 dropping to moderate and low-moderate expression tiers, respectively. The discrete boundaries of this sub-state are established by a signature of cellular priming and organelle remodeling. The cluster is uniquely defined by the positive expression of SLC9A7—indicating targeted pH regulation of the Golgi apparatus for altered secretory trafficking—alongside CCSER1 and a low-amplitude, high-variance expression of CDK14. This profile isolates a structurally poised, non-proliferative transitional or memory T-cell compartment that has exited strict naive homeostasis but maintains developmental continuity with the BCL11B-positive parent manifold.]`.
            
    - **Micro-Cluster 0.3:** `[Metabolically Poised Transitional T-Cell (CL:0000813)]`
        
        - **Decision Factor:** `[Sustained high expression of IL7R dictates an ongoing reliance on peripheral homeostatic survival networks. However, the explicit transcriptomic silencing of the foundational CAMK4 and BCL11B signatures physically decouples this population from the baseline naive state. The mathematical boundary isolating this sub-cluster is defined by the unique, high-variance expression of GRAMD1B. Despite a low mean expression tier (0.5), this sterol-transport transcript indicates targeted biophysical remodeling of plasma membrane lipid rafts. This signature identifies a distinct, metabolically poised transitional T-cell pool undergoing structural membrane reorganization, indicative of memory lineage commitment or immune synapse priming.]`.
            
	- **Micro-Cluster 0.4:** `[Naive T-Cell (CL:0000898) — Isomorphic Sub-partition]`
        
		- **Decision Factor:** `[This sub-cluster exhibits a complete absence of unique differential or explicit absence markers, indicating it does not represent a discrete biological phase transition. Instead, it maintains robust expression of the foundational T-cell manifold anchors (high CAMK4, moderate BCL11B) and the peripheral survival receptor IL7R. Critically, its identity is established by a shared, albeit slightly attenuated, expression profile of the canonical naive markers defining Micro-Cluster 0.0, notably NELL2 (0.8 mean expression), TRABD2A (0.5), and FHIT (0.3). The mathematical isolation of this state by the Leiden algorithm reflects topological density variations—specifically transcriptomic amplitude gradients within the massive naive T-cell pool—rather than a true biological divergence, confirming its terminal identity as an isomorphic sub-partition of the resting Naive T-Cell compartment.]`.
			
    - **Micro-Cluster 0.5:** `[Effector T-Cell (CL:0000911)]`
        
        - **Decision Factor:** `[Topological derivation of this state is defined by a definitive transition from a poised memory phenotype into an active effector phase. The shared high expression of the lipid-remodeling transcript GRAMD1B serves as a continuous transcriptomic bridge linking this population to the transitional memory pool (Micro-Cluster 0.3). However, the mathematical boundary isolating this sub-cluster is driven by the activation of the terminal differentiation transcription factor ZEB2. The subsequent downstream weaponization of the cell is evidenced by the massive, exclusive upregulation of the cytotoxic granule stabilizer NKG7 and the inflammatory chemokine CCL5. This profile definitively isolates a structurally armed, terminal Effector T-Cell compartment within the broader lymphoid manifold.]`.
            
	- **Micro-Cluster 0.6:** `[Terminally Differentiated / Exocytic Effector T-Cell (CL:0000911)]`
		
		-  **Decision Factor:** `[Topological continuity with the primary effector pool (Micro-Cluster 0.5) is established through the sustained, high-level expression of the terminal differentiation transcription factor ZEB2 and the cytotoxic granule component NKG7. However, the mathematical isolation of this discrete sub-state is driven by a profound cytoskeletal and signaling shift required for target engagement. The exclusive upregulation of PLEK—a key mediator of actin reorganization and granular exocytosis—coupled with the synaptic signaling kinase LYN, identifies a functionally distinct, terminally differentiated effector population mechanically poised for direct cytolytic execution and active degranulation.]`.
            
    - **Micro-Cluster 0.7:** `[Hyper-Quiescent / Homeostatic T-Cell (CL:0000898)]`
        
        - **Decision Factor:** `[Spatial origin and developmental continuity with the primary resting pool (Micro-Cluster 0.0) are strictly established by the continuous, high-level expression of the transcriptomic bridge FHIT, coupled with maximal expression of the homeostatic survival receptor IL7R and the foundational CAMK4/BCL11B anchors. The mathematical partition of this discrete sub-state is driven by a thermodynamic shift from passive resting to active transcriptional suppression. The discrete boundary is defined by the high, exclusive expression of ID3—a master repressor that competitively inhibits effector differentiation pathways—alongside the basal translational regulators TSR1 and TEX9. This coordinate signature defines a hyper-quiescent sub-population actively enforcing strict homeostatic longevity and preventing spontaneous activation within the resting manifold.]`.
            

**Macro-Cluster 1: `[Myeloid Lineage (CL:0000763)]**`

- **Macro Decision Factor:** `[Universal expression of the myeloid master transcription factor SPI1, coupled with high baseline expression of the canonical myeloid markers LYZ and CLEC7A. The absolute transcriptomic silencing of all foreign macro-lineage markers confirms a definitive, discrete boundary isolating the mononuclear phagocyte compartment.]`.
    ![[results/figures/p05_top_markers/matrixplot__macro_leiden_1_micro_leiden_top_genes.svg]]
    ![[results/figures/p05_top_markers/dotplot__macro_leiden_1_micro_leiden_top_genes.svg]]
    ![[results/figures/p05_top_markers/matrixplot__absence_audit_macro_1.svg]]
    ![[results/figures/p05_top_markers/matrixplot__curated_genes_audit_widespan_macro_leiden_1_micro_leiden.svg]]
	- **Micro-Cluster 1.0:** `[Non-Classical (CD16+) Patrolling Monocyte (CL:0002057)]`
        
		- **Decision Factor:** `[Developmental continuity with the mature myeloid manifold is strictly maintained through the robust, retained expression of the terminal differentiation factor ZEB2 and the ubiquitous protease inhibitor CST3, anchored by basal canonical pathways (high FTL, moderate CD74). However, the mathematical isolation of this specific sub-state is driven by the primary spatial variance of FCGR3A (CD16), definitively partitioning it from the classical monocyte baseline. This receptor transition is coupled with the distinct upregulation of the trafficking mediator FAM117B and a low-amplitude, high-variance expression of the cytoskeletal driver RHOC. This specific coordinate configuration isolates a mature, non-classical monocyte compartment actively structurally poised for endothelial patrolling and cellular motility.]`.
            
    - **Micro-Cluster 1.1:** `[Classical (CD14+) Monocyte (CL:0000860)]`
        
        - **Decision Factor:** `[Anchored fundamentally to the mature myeloid manifold via maximal expression of ZEB2, CST3, and FTL, this sub-state establishes its primary classical lineage identity through the dominant expression of the canonical calprotectin subunit S100A9 and massive baseline transcription of Lysozyme (LYZ). The discrete mathematical boundary is defined by the high-variance expression of the leukotriene-synthesis anchor ALOX5AP and the classical-monocyte specific regulator MCEMP1, alongside a poised, low-amplitude transcription of the antimicrobial peptide RNASE6. Basal metabolic continuity is verified by the retained topological bridges SLC16A10 and SMOX. Crucially, the absolute transcriptomic silencing of TMEM106A structurally enforces the boundary of this cluster, mathematically proving these cells remain locked in the circulating monocyte phase and have not initiated tissue-resident macrophage differentiation.]`.
            
	- **Micro-Cluster 1.2:** `[Tissue-Differentiating Macrophage (CL:0000235)]`
        
		-  **Decision Factor:** `[Lineage continuity with the mature circulating myeloid pool is preserved via the persistent topological bridges ZEB2 and CST3. However, the mathematical derivation of this discrete state is defined by a profound biophysical phase transition: the attenuation of primary circulating anchors (LYZ and FTL dropping to moderate log-means near 2.0) coupled with the explosive, maximal upregulation of TMEM106A. The high expression of TMEM106A structurally dictates advanced endolysosomal maturation, mathematically proving the breach of the circulating monocyte threshold and the initiation of tissue-resident macrophage differentiation. This physical structural rewiring is thermodynamically supported by the high-variance expression of SMOX (driving polyamine-mediated chromatin remodeling) and the amino acid transporter SLC16A10, isolating a distinct population undergoing active tissue-permeation and terminal macrophage commitment.]`.
            
    - **Micro-Cluster 1.3:** `[Activated / Hyper-Inflammatory Classical Monocyte (CL:0000860)]`
        
        - **Decision Factor:** `[Topological derivation proves this state is a direct, hyper-activated evolutionary derivative of the classical monocyte baseline (Micro-Cluster 1.1), evidenced by the retained, high-expression bridges of the leukotriene-driver ALOX5AP and the antimicrobial peptide RNASE6. The mathematical isolation of this discrete phase is governed by a systemic activation signature. The absolute boundaries are established by the high-variance upregulation of the high-affinity targeting receptor FCGR1A (CD64)—indicating an acute response to systemic inflammatory signaling (e.g., IFN-gamma)—coupled with the matrix-degrading enzyme MMP19, which structurally primes the cell for endothelial extravasation. This extreme metabolic and translational surge (further evidenced by maximal LYZ and FTL amplitudes) is thermodynamically stabilized by the high expression of the E3 ubiquitin ligase STUB1, mathematically defining an activated classical monocyte locked in a hyper-inflammatory, pre-extravasation state.]`.
            
	- **Micro-Cluster 1.4:** `[Heterotypic Doublet (CD16+ Monocyte x B-Cell Artifact)]`
        
		- **Decision Factor:** `[Topological derivation of this cluster reveals a mathematically isolated coordinate space defined by the simultaneous co-expression of mutually exclusive lineage programs. The robust, retained expression of FCGR3A, FAM117B, and RHOC tethers this profile strictly to the Non-Classical Monocyte manifold (Micro-Cluster 1.0). However, the discrete mathematical boundary driving this cluster is forged by the maximal, orthogonal expression of CD79B—a canonical and absolute structural component of the B-Cell Receptor complex—alongside LINC00540 and LINC02432. Because the simultaneous thermodynamic maintenance of a patrolling myeloid cytoskeletal program and a resting lymphoid receptor complex is biologically impossible within a single diploid nucleus, this sub-cluster mathematically represents a heterotypic doublet. It is a microfluidic artifact capturing the physical co-encapsulation of a Non-Classical Monocyte and a B-Cell within a single partition.]`.
        

**Macro-Cluster 2: `[T-Cell Sub-Lineage (CL:0000084)]**`

- **Macro Decision Factor:** `[High expression of IL7R, coupled with moderate baseline expression of CAMK4 and BCL11B, establishes a shared lymphoid origin with Macro-Cluster 0. However, this population was mathematically partitioned into a discrete macro-state due to the exclusive, high-level upregulation of the homing receptor CCR4, alongside SNED1 and NECTIN3. The complete absence of myeloid or other foreign lineage markers confirms this boundary represents a distinct, specialized T-cell sub-compartment (indicative of a polarized or regulatory state) rather than a technical doublet.]`.
    ![[results/figures/p05_top_markers/matrixplot__macro_leiden_2_micro_leiden_top_genes.svg]]
    ![[results/figures/p05_top_markers/dotplot__macro_leiden_2_micro_leiden_top_genes.svg]]
    ![[results/figures/p05_top_markers/matrixplot__absence_audit_macro_2.svg]]
    ![[results/figures/p05_top_markers/matrixplot__curated_genes_audit_widespan_macro_leiden_2_micro_leiden.svg]]
    - **Micro-Cluster 2.0:** `[Central Memory CD4+ T-Cell (CL:0000904)]`
        
        - **Decision Factor:** `[The structural foundation of this state is defined by the maximal expression of the lineage-defining co-receptor CD4, mathematically isolating it within the helper/regulatory T-cell manifold. Topological derivation proves the exit from the pristine naive phase via the targeted attenuation of the homeostatic anchors IL7R (dropping to ~2.0) and CAMK4 (dropping to 1.0), confirming an antigen-experienced, memory timeline. The discrete mathematical boundary of this sub-cluster is governed by the simultaneous high expression of the primary co-stimulatory receptor CD28—priming the cell for secondary expansion—and the immunophilin FKBP5. The massive upregulation of FKBP5 dictates active, endogenous regulation of glucocorticoid receptor sensitivity, structurally defining a polarized, tissue-homing memory cell that is thermodynamically enforcing strict threshold checks against premature inflammatory activation.]`.
            
    - **Micro-Cluster 2.1:** `[Cytotoxic / Effector Memory CD4+ T-Cell (CL:0000905)]`
        
        - **Decision Factor:** `[Developmental exit from the resting Central Memory pool (Micro-Cluster 2.0) is mathematically dictated by the continued attenuation of the homeostatic anchor IL7R (dropping to a log-mean of 1.8). The discrete isolation of this sub-state is governed by a profound phenotypic polarization: the explicit weaponization of the CD4+ lineage. The coordinate boundary is defined by the massive upregulation of the cytotoxic granule stabilizer NKG7 and the inflammatory chemokine CCL5, establishing a direct topological bridge to the broader cytotoxic effector manifold. This terminal arming phase is mechanistically supported by the high expression of the calcium-sensor NCALD, structurally hypersensitizing the cell's degranulation threshold and defining a mature, tissue-infiltrating Effector Memory CD4+ T-Cell with potent direct cytolytic capabilities.]`.
	        
	  - **Micro-Cluster 2.2:** `[Sub-Threshold Artifact (CD4+ T-Cell x B-Cell Doublet / Noise Floor)]`
        
        - **Decision Factor:** `[Mathematical elimination of this sub-state is dictated by a complete failure to meet the minimum viable statistical mass (N < 10) required to define a stable thermodynamic or biological phase. Furthermore, the localized coordinate space is structurally corrupted by the orthogonal presence of absolute B-cell lineage determinants (MS4A1/CD20, CD79A, BANK1) within the strictly defined CD4+ T-cell macro-manifold. The physical reality of this coordinate is a rare, low-frequency microfluidic collision—a heterotypic doublet—representing technical static rather than a true biological entity. This space is deemed non-viable and designated for structural purging to protect downstream pipeline integrity.]`.
        

**Macro-Cluster 3: `[B-Cell Lineage (CL:0000236)]**`

- **Macro Decision Factor:** `[High, universal expression of the canonical B-cell markers BANK1 and IGKC, alongside COBLL1, dictates a definitive commitment to the B-cell lineage. The cluster exhibits complete transcriptomic silencing of myeloid and foundational T-cell drivers. However, a moderate baseline expression of SNED1 (shared with Macro-Cluster 2) was preserved, indicating that while the algorithm mathematically isolated this independent B-cell macro-state, it accurately retained the continuous transcriptomic edges between the broader lymphoid sub-populations.]`.
    ![[results/figures/p05_top_markers/matrixplot__macro_leiden_3_micro_leiden_top_genes.svg]]
    ![[results/figures/p05_top_markers/dotplot__macro_leiden_3_micro_leiden_top_genes.svg]]
    ![[results/figures/p05_top_markers/matrixplot__absence_audit_macro_3.svg]]
    ![[results/figures/p05_top_markers/matrixplot__curated_genes_audit_widespan_macro_leiden_3_micro_leiden.svg]]
    - **Micro-Cluster 3.0:** `[Resting / Naive B-Cell (CL:0000788)]`
        
        - **Decision Factor:** `[Structural anchorage to the mature B-cell manifold is definitively established by the retained, high-expression bridges of the canonical lineage markers MS4A1 (CD20) and CD79A, alongside the critical peripheral survival receptor TNFRSF13C (BAFF-R). The basal signaling readiness of this cell is maintained by moderate expression of the receptor-associated kinase LYN and massive accumulation of the MHC Class II invariant chain (CD74 at 3.6). The mathematical isolation of this discrete phase is governed by the high-variance transcription of SOX5—a master regulator that acts as a thermodynamic padlock to actively repress terminal differentiation and enforce strict cellular quiescence. Supported by an atypical paracrine IL7 signature, this coordinate space mathematically defines a pristine, immunologically uncommitted Resting Naive B-Cell poised for initial antigen encounter.]`.
		
	- **Micro-Cluster 3.1:** `[Heterotypic Doublet / Synaptic Artifact (T-Cell x B-Cell)]`
        
        - **Decision Factor:** `[Topological derivation of this cluster reveals a severe thermodynamic contradiction and a violation of lineage physics. Despite localizing near the B-cell manifold, the primary variance of this sub-state is driven by the maximal expression of absolute T-cell lineage determinants, specifically the master transcription factor BCL11B and the T-Cell Receptor structural adaptor TRAT1 (alongside the receptor CD96). Conversely, the foundational B-cell anchors (CD79A, MS4A1, TNFRSF13C) are recorded at heavily attenuated, faded amplitudes (mean ~0.18). Because a single nucleus cannot sustain both T-cell lineage commitment and a B-cell receptor complex, this coordinate space mathematically defines a heterotypic doublet. The asymmetric marker amplitude strongly suggests the physical co-encapsulation of a highly active T-cell physically bound to a resting B-cell via an immunological synapse. This cluster is a microfluidic artifact and must be quarantined from biological downstream analysis.]`.
	        
	- **Micro-Cluster 3.2:** `[Mature Naive B-Cell / Core Manifold (CL:0000788)]`
        
        - **Decision Factor:** `[Topological derivation reveals this coordinate space as the mathematical center of gravity for the B-cell lineage. The discrete isolation of this cluster is driven exclusively by the maximal, high-variance expression of the foundational lineage anchors: the mature ion channel MS4A1 (CD20), the receptor signaling tail CD79A, and the homeostatic survival receptor TNFRSF13C (BAFF-R). The thermodynamic physics of this state indicate a mature equilibrium; it lacks the active transcriptional padlock (SOX5) observed in transitional states, relying instead on structural stability. Basal signaling readiness is verified by the retained expression of the docked kinase LYN (1.2), while a massive accumulation of the MHC Class II invariant chain (CD74 at 3.9) structurally primes the cell for immediate antigen processing and T-cell presentation. This signature mathematically isolates the unadulterated, prototypical baseline of the circulating Naive B-Cell pool.]`.
	        
	- **Micro-Cluster 3.3:** `[Cytotoxic Natural Killer (NK) Cell (CL:0000623)]`
        
        - **Decision Factor:** `[Topological deconstruction reveals a severe algorithmic spatial artifact: this sub-cluster physically resides within the B-cell macro-manifold but transcribes an entirely orthogonal, innate cytolytic lineage. The mathematical isolation and identity of this state are driven by the absolute foundational expression of the early T/NK-lineage anchor CD7, explicitly divorcing it from humoral biology. The thermodynamic reality of the cell is defined by a massive weaponization profile: the maximal expression of the cytotoxic granule stabilizer NKG7, coupled perfectly with the extreme upregulation of Cystatin F (CST7), which acts as an intracellular thermodynamic shield against autolytic protease damage. Target acquisition is independent of antigen presentation, driven instead by the high, retained expression of the innate stress-receptor CD96 (TACTILE). This coordinate geometry flawlessly defines a mature, fully armed Natural Killer (NK) cell, requiring mathematical re-assignment away from the B-cell ontology.]`.
	        
	- **Micro-Cluster 3.4:** `[Mature Dendritic Cell / Professional APC (CL:0001056)]`
        
        - **Decision Factor:** `[Structural deconstruction reveals this coordinate space to be computationally misplaced within the B-cell manifold due to functional transcriptomic convergence. The spatial proximity to Macro 3 is driven exclusively by the massive, shared expression of the antigen-presentation engine CD74 (MHC Class II invariant chain at 3.6). However, the true embryological lineage of this cluster is mathematically proven by the maximal expression of CST3 and the robust, retained topological bridge ZEB2 (2.4), which firmly tether this cell to the Myeloid developmental trajectory. Completely lacking B-cell receptor complex machinery, this cell utilizes the calcium-binding transmembrane protein MCTP1 to manage vesicular trafficking, defining a mature, fully differentiated Dendritic Cell perfectly optimized for professional antigen presentation.]`.
	        
	- **Micro-Cluster 3.5 and 3.6:** `[Sub-Threshold Artifact / Noise Floor (N < 10)]`
        
        - **Decision Factor:** `[Mathematical elimination of this coordinate space is dictated by a complete failure to meet the minimum viable statistical mass (N < 10) required to define a stable thermodynamic or biological phase. Residing at the extreme spatial periphery of the B-cell macro-manifold, this sub-state lacks the transcriptomic gravity to represent a true lineage or functional transition. It is classified as technical static—likely representing apoptotic debris or fragmented microfluidic capture—and is mathematically quarantined and designated for purging to protect downstream pipeline integrity.]`.
   

**Macro-Cluster 4: `[Natural Killer (NK) Cell Lineage (CL:0000623)]**`

- **Macro Decision Factor:** `[High, universal expression of the cytotoxic effector molecules GZMB and NKG7, alongside the chemokine CCL4, establishes a definitive cytotoxic profile. Crucially, the absolute transcriptomic silencing of foundational T-cell lineage drivers (e.g., BCL11B) mathematically and biologically isolates this population as the innate Natural Killer (NK) cell lineage, successfully partitioning it from adaptive cytotoxic T-cell states]`.
    ![[results/figures/p05_top_markers/matrixplot__macro_leiden_4_micro_leiden_top_genes.svg]]
    ![[results/figures/p05_top_markers/dotplot__macro_leiden_4_micro_leiden_top_genes.svg]]
    ![[results/figures/p05_top_markers/matrixplot__absence_audit_macro_4.svg]]
    ![[results/figures/p05_top_markers/matrixplot__curated_genes_audit_widespan_macro_leiden_4_micro_leiden.svg]]
    - **Micro-Cluster 4.0:** `[Terminally Differentiated Cytotoxic Effector / Core Manifold (CL:4030002)]`
        
        - **Decision Factor:** `[Topological derivation reveals this coordinate space as the mathematical centroid and unadulterated baseline of the cytotoxic macro-manifold. The discrete isolation of this cluster is characterized by an absence of novel primary variance drivers, indicating it serves as the foundational operating system for the surrounding continent. The physical reality of this state is defined exclusively by its high-amplitude topological anchors: the master transcription factor ZEB2 (3.0) acts as a thermodynamic padlock enforcing terminal differentiation and preventing memory reversion. This locked state is structurally coupled with an apex cytolytic arsenal, defined by the co-expression of the granule stabilizer NKG7 (2.5) and the membrane-breaching protein Granulysin (GNLY at 2.0), alongside the inflammatory chemokine CCL5. This exact thermodynamic signature mathematically defines the core circulating pool of terminally differentiated, fully armed cytotoxic effector cells.]`.
            
    - **Micro-Cluster 4.1:** `[Actively Engaged / Synaptic Cytotoxic Effector (CL:4030002)]`
        
        - **Decision Factor:** `[Topological derivation confirms this sub-state is firmly anchored to the terminal cytotoxic manifold via the maximal expression of the lineage padlock ZEB2 (3.5) and the sustained, high-amplitude cytolytic arsenal (GNLY at 3.5, NKG7 at ~3.0). However, the mathematical isolation of this cluster is driven by the physics of active target execution. The primary spatial variance is defined by the explosive upregulation of the acute inflammatory chemokines CCL3 and CCL4L2—acting as a localized transcriptomic flare to recruit phagocytic clearance machinery. Concurrently, the discrete, high-variance transcription of the actin-binding protein Supervillin (SVIL) mathematically proves active cytoskeletal remodeling. SVIL physically braces the plasma membrane to the actin network, providing the rigid structural scaffolding necessary for immunological synapse formation and the mechanical exocytosis of cytotoxic granules. This coordinate space exactly defines a terminally differentiated effector actively executing a target.]`.
	        
	- **Micro-Cluster 4.2:** `[Resting / Central Memory Cytotoxic T-Cell (CL:0000909)]`
        
        - **Decision Factor:** `[Topological derivation reveals a profound thermodynamic reversion within the cytotoxic macro-manifold, fundamentally separating this sub-state from the terminally differentiated baseline (Micro-Cluster 4.0). The discrete isolation of this cluster is driven by the maximal expression of CAMK4, a master kinase that physically enforces genomic quiescence, memory stemness, and mathematically prevents terminal effector differentiation. This long-lived structural state is metabolically sustained by a shift toward lipid degradation and fatty acid oxidation, proven by the high-variance transcription of FAAH2. The cell maintains a highly poised, resting architecture through the upregulation of the nuclear import structural protein KPNA5 (Importin alpha 6), ensuring rapid nucleocytoplasmic transport capacity upon secondary antigen encounter. This coordinate signature exactly defines a stable, circulating Central Memory CD8+ T-Cell.]`.
	        
	- **Micro-Cluster 4.3:** `[Exhausted / Senescent Cytotoxic Effector (CL:4030002)]`
        
        - **Decision Factor:** `[Topological derivation confirms this sub-state is structurally tethered to the terminal cytotoxic manifold via the retained expression of the differentiation padlock ZEB2 (3.0) and the baseline chemokine CCL5 (1.9). However, the mathematical isolation of this cluster is driven by the physical transcriptomic signature of terminal exhaustion and impending Activation-Induced Cell Death (AICD). The active weaponized payload is observed attenuating (GNLY dropping to 1.8), replaced by the high-variance transcription of CAPN15 (Calpain 15)—a calcium-dependent protease indicative of severe intracellular calcium stress and active cytoskeletal degradation. Concurrently, the massive upregulation of the metabolic stress enzyme CBR4 mathematically proves the cell is managing severe oxidative damage and mitochondrial lipid dysregulation. This coordinate space exactly defines a terminally differentiated effector that has exhausted its cytolytic capacity and entered a senescent or apoptotic trajectory.]`.
        

#### 3.5.2 Validation of Cluster Labels on the Holdout Dataset

To evaluate the reliability of the defined cluster boundaries, the pipeline utilized a split-sample validation approach. The structural manifold was established using the training subset ($X_{train}$), and the independent holdout dataset ($X_{project}$) was subsequently projected into this pre-defined coordinate space.

This projection demonstrated a high degree of structural stability, as the cells from the holdout set aligned consistently with the established transcriptomic boundaries. These results indicate that the identified populations are not the result of overfitting or technical noise, but represent stable biological states within the PBMC population. While external supervised benchmarking (e.g., against CellTypist models) provides an additional layer of validation, the internal consistency demonstrated through the train-test projection provides evidence for the robustness of the final integrated dataset.

#### 3.5.3 Final Tensor Recombination

In the final step, the annotation metadata (`master_labels_df.csv`) was merged back into the original, un-split dataset (`recombine_topology`). This step matched the cell barcodes with their assigned CL IDs across the global dataset.

This operation exported the final deployment asset: `[ pbmc3k_qc_ML_Ready.h5ad]`. This unified tensor represents a fully verified dataset containing exactly zero barcode collisions and `[82]` unannotated cells, generating a finalized dataset ready for downstream analysis.

### 3.6 Technical Verification of Pipeline Integrity

> [!IMPORTANT] **Forensic Audit Notice:** This section provides the transcriptomic and computational proof that the biological results described in Sections 3.1–3.5 were derived through the validated analytical pipeline.

#### 3.6.1 Phase I & II: Ingestion and Topological Scaffolding

The following log verifies the successful ingestion of the filtered feature-barcode matrix and the subsequent generation of the Leiden-based lineage maps.

- **Evidence:** 
    ![[results/logs/execution_evidence/EVT_1.png]]
    ![[results/logs/execution_evidence/EVT_2.png]]
    ![[results/logs/execution_evidence/EVT_3.png]]
    ![[results/logs/execution_evidence/EVT_4.png]]

#### 3.6.2 Phase III: Marker Extraction and JSON Ledger Sealing

Verification of the Wilcoxon Rank-Sum execution and the atomic sealing of the annotation and ontology ledgers to the physical disk.

- **Evidence:** 
    ![[results/logs/execution_evidence/EVT_5.png]]
    ![[results/logs/execution_evidence/EVT_6.png]]
    ![[results/logs/execution_evidence/EVT_7.png]]
    ![[results/logs/execution_evidence/EVT_8.png]]


------------
## 4 Discussion

The primary objective of this workflow was to identify discrete immune cell populations from single-cell transcriptomic data while systematically preventing data leakage and circular inference. By utilizing automated parameter selection and independent marker validation, the pipeline generated a rigorously annotated and standardized dataset.

### 4.1 Resolution of PBMC Subpopulations

The iterative clustering approach successfully mirrored the known biological hierarchy of human PBMCs. The initial global clustering effectively established the primary biological lineages, separating the `[T-Cell Lineage,Myeloid Lineage,B-Cell Lineage,Natural Killer (NK) Cell Lineage]`.

Crucially, the recursive Micro-State execution proved capable of resolving high-resolution biological variance that is typically lost in global PCA manifolds. For instance, the pipeline successfully segregated `[Naive T-Cell]` from `[Effector T-Cell]`. This confirms that recalculating Highly Variable Genes (HVGs) and Pearson residuals specifically within each isolated macro-cluster is necessary to capture the subtle transcriptomic variance driving micro-phenotypes.

### 4.2 Transcriptomic Fidelity and Marker Gene Selection

The application of adaptive expression thresholds and targeted independent validation mitigated the limitations of relying solely on standard differential expression tests.

The extracted defining markers, such as `[FCGR3A,FAM117B,RHOC]` for the `[Non-Classical (CD16+) Patrolling Monocyte]` population, demonstrated not only extreme positive up-regulation (Log2FoldChange > 1.0) but also near-absolute silencing in foreign macro-lineages. In instances where strict statistical cutoffs yielded insufficient markers (e.g., within the fragile `[Metabolically Poised Transitional T-Cell]` compartment), dynamic thresholding successfully captured a core set of defining genes, preventing the cluster from remaining unannotated. This proves that the pipeline's mathematical boundaries align strictly with the biological reality of transcriptomic regulation.

### 4.3 Methodological Limitations

Despite stringent filtering, this computational approach possesses inherent limitations::

- **Rigid Variance Thresholds:** Enforcing a strict PCA variance ratio threshold prevents over-clustering, but inherently limits the algorithm's ability to map continuous developmental trajectories (e.g., hematopoiesis), which may not exhibit sharp structural divergences.
    
- **Cluster Size Constraints**: By enforcing a strict minimum cell count threshold ($N \ge 10$) for differential expression testing, the pipeline sacrifices the ability to identify ultra-rare cell types (e.g., circulating dendritic cell subsets) in order to preserve the statistical integrity of the broader dataset.
    

### 4.4 Deployment and FAIR Compliance

The final output of this pipeline is not merely a descriptive report, but an interoperable computational asset. By strictly segregating the dataset into training ($X_{train}$) and projection ($X_{project}$) subsets, we demonstrated that the defined cluster boundaries are robust and reproducible when applied to holdout data, effectively mitigating the risk of algorithmic overfitting. The final output, `[pbmc3k_qc_ML_Ready.h5ad]`, is a fully annotated, FAIR-compliant dataset. It has been rigorously filtered for technical noise, mapped to standardized Cell Ontology IDs, and is prepared for downstream applications such as predictive modeling or spatial data integration.

