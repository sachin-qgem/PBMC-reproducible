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
