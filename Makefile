.PHONY: setup clean run

setup:
	@echo "Forging 3.11 Bedrock..."
	conda env create -f environment.yml -p ./.conda || conda env update -f environment.yml -p ./.conda
	@echo "Installing 5-Sigma Mathematical Engines..."
	./.conda/bin/pip install -e .

clean:
	@echo "Vaporizing Exhaust and Artifacts..."
	rm -rf cache/*
	rm -rf .conda
	rm -rf data/objects/*
	rm -rf results/figures/*
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +

run:
	@echo "Igniting Visual Telemetry Dashboard..."
	./.conda/bin/streamlit run app.py