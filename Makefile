# =============================================================================
# PBMC3k REPRODUCIBLE PIPELINE: CHRONOLOGICAL CONTROL LEDGER
# =============================================================================
# WORKFLOW ORDER: setup -> evidence -> ui -> p06
# =============================================================================

# PHYSICAL BINARY RESOLUTION
ENV_PATH = ./.conda
PYTHON = $(ENV_PATH)/bin/python
STREAMLIT = $(ENV_PATH)/bin/streamlit

# --- PHASE 0: THE FORGE ---
.PHONY: setup
setup:
	@echo "INITIATING HYBRID FORGE: Building Python 3.11 Bedrock..."
	conda env create --prefix $(ENV_PATH) -f streamlit_based_env.yml
	@echo "SUCCESS: Environment forged at $(ENV_PATH)"

# --- PHASE I-III: THE EVIDENCE (AUTOMATED) ---
.PHONY: evidence
evidence:
	@echo "GENERATING ANALYTICAL EVIDENCE..."
	@echo "-> Running P03 (QC & Filtration)"
	$(PYTHON) src/02_analysis_scripts/P03_qc_filtering.py
	@echo "-> Running P04 (Clustering)"
	$(PYTHON) src/02_analysis_scripts/P04_clustering.py
	@echo "-> Running P05 (Marker Extraction)"
	$(PYTHON) src/02_analysis_scripts/P05_top_markers.py
	@echo "SUCCESS: Evidence generated in ./results/figures/ and ./data/objects/"

# --- PHASE IV-A: THE INTERVENTION (HUMAN-IN-THE-LOOP) ---
.PHONY: ui
ui:
	@echo "IGNITING UI BRIDGE: app.py"
	@echo "ACTION REQUIRED: Enter cluster labels in the browser and click 'Seal Ledgers'."
	$(STREAMLIT) run app.py

# --- PHASE IV-B: THE FINAL SEAL (RECOMBINATION) ---
.PHONY: p06
p06:
	@echo "EXECUTING PHASE IV RECOMBINATION..."
	@echo "CAUTION: This assumes you have already populated the JSONs via 'make ui'."
	$(PYTHON) src/02_analysis_scripts/P06_annotation.py
	@echo "SUCCESS: Final ML-Ready matrix forged."

# --- CLOUD PRE-FLIGHT ---
.PHONY: prepare_hf
prepare_hf:
	@echo "STAGING FOR HUGGING FACE ASCENT..."
	git add app.py requirements.txt environment.yml .gitattributes Makefile
	git lfs ls-files
	@echo "READY. Execute: git commit -m 'Final Seal' && git push hf main"

.PHONY: clean
clean:
	@echo "ANNIHILATING CACHE AND ENVIRONMENT..."
	rm -rf $(ENV_PATH)
	find . -type d -name "__pycache__" -exec rm -rf {} +
	@echo "CLEAN COMPLETE."