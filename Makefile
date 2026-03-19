# ==============================================================================
# PBMC3k Forensic Reconstruction - Master Execution Console
# ==============================================================================

# --- Variables ---
PYTHON = .conda/bin/python
SRC_DIR = src/02_analysis_scripts
RESULTS_DIR = results

# --- Phony Targets ---
.PHONY: help setup clean pipeline matrix qc cluster markers annotate

# --- Master Execution ---
help:
	@echo "PBMC3k-Reproducible Pipeline Execution Console"
	@echo "--------------------------------------------"
	@echo "------This Pipeline is only for Single Donor Dataset, Development for multi_donor,batch integration and correction going on at this moment----------------"
	@echo "Available commands:"
	@echo "  make setup      - Forges the isolated Conda environment from environment.yml"
	@echo "  make pipeline   - Executes the entire forensic pipeline end-to-end"
	@echo "  make matrix     - Executes Phase 0 (Matrix Construction) Not included,Only if you can run cellbender on linux, i have macos"
	@echo "  make qc         - Executes Phase I (QC & Filtering)"
	@echo "  make cluster    - Executes Phase II (Clustering & Projection)"
	@echo "  make markers    - Executes Phase III (Marker Extraction)"
	@echo "  make annotate   - Executes Phase IV (Annotation Injection)"
	@echo "  make clean      - Wipes generated artifacts, cache, and logs to reset workspace"

# --- Environment Genesis ---
setup:
	@echo "\n[INIT] Forging isolated thermodynamic environment..."
	conda env create --prefix ./.conda -f stable_environment.yml
	@echo "\n[SUCCESS] Environment forged. Run 'conda activate ./.conda' before executing the pipeline."

# --- Pipeline Sequence ---
pipeline: clean matrix qc cluster markers annotate
	@echo "\n[TERMINAL] Full Pipeline Execution Complete. ML Tensor ready."

# --- Individual Engines ---
qc:
	@echo "\n[1/5] INITIATING QUALITY CONTROL..."
	$(PYTHON) $(SRC_DIR)/P03_qc_filtering.py

cluster:
	@echo "\n[2/5] INITIATING CLUSTERING AND LATENT GEOMETRY..."
	$(PYTHON) $(SRC_DIR)/P04_clustering.py

markers:
	@echo "\n[3/5] INITIATING THERMODYNAMIC MARKER EXTRACTION..."
	$(PYTHON) $(SRC_DIR)/P05_top_markers.py

annotate:
	@echo "\n[4/5] INITIATING ANNOTATION INJECTION..."
	$(PYTHON) $(SRC_DIR)/P06_annotation.py

# --- Workspace Reset ---
clean:
	@echo "\n[WARNING] Purging generated matrices, ledgers, and temporary OS noise..."
	rm -rf $(RESULTS_DIR)/figures/*/*
	rm -rf $(RESULTS_DIR)/*.json
	rm -rf $(RESULTS_DIR)/*.csv
	rm -rf data/objects/*
	rm -rf cache/*
	rm -rf logs/*
	find . -type d -name "__pycache__" -exec rm -rf {} +
	@echo "[CLEARED] Workspace reset to genesis state."
