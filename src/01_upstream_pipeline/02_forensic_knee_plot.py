import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
#---------------
#config
#----------------
H5_PATH = "data/raw/pbmc3k_molecule_info.h5"
FILTERED_MATRIX_PATH = "data/raw/pbmc3k_filtered_gene_bc_matrices/hg19/barcodes.tsv"

def generate_knee_plot():
    print("   °°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°")
    print ("Reconstructing the knee")
    print("   ------------------------------------------------")
    #Loading h5 File
    #AVAILABLE KEYS: ['barcode', 'barcode_corrected_reads', 'conf_mapped_uniq_read_pos',
    # 'gem_group', 'gene', 'nonconf_mapped_reads', 'reads', 'umi',
    # 'umi_corrected_reads', 'unmapped_reads'],
    with h5py.File(H5_PATH,'r') as fh5:
        barcode_indices = np.array(fh5['barcode'][:])

    print(f"    -> Loaded {len(barcode_indices):,} molecular events.")

    print("   ------------------------------------------------")

    #Calculate (UMIs per barcode) by simply counting numbers each barcode appeared
    print(f"    -> Summing UMIS per barcode")
    print("   ------------------------------------------------")
    # The V1 file might map indices to a whitelist. We just count observed indices.
    umi_counts = pd.Series(barcode_indices).value_counts()
    umi_counts = umi_counts.sort_values(ascending=False)

    #Ranking the barcodes
    ranks = np.arange(len(umi_counts))
    counts = umi_counts.values

    # How many cells did official 10x Genomics actually keep? (For Comparison)
    if os.path.exists(FILTERED_MATRIX_PATH):
        with open(FILTERED_MATRIX_PATH, 'r') as fm:
            official_cell_count = len(fm.readlines())
        print(f"    -> Official 10x Verdict: {official_cell_count} Cells Kept")
    else:
        official_cell_count = 2700 # Fallback known value for PBMC3k
        print("   -> Official verdict file not found (using default 2700).")
    
    #Knee plot
    plt.figure(figsize=(10,6))

    #plot the barcodes(Blue Line)
    plt.loglog(ranks,counts,color = 'darkgrey', linewidth = 2, label ='All barcodes(Raw)')
 
    # Highlight the "Real Cells" based on the official count
    # We draw a line exactly where 10x made the cut
    plt.axvline(x=official_cell_count, color='red', linestyle='--', label=f'10x Cutoff (~{official_cell_count})')
    
    # Aesthetics
    plt.xlabel('Barcode Rank (sorted by UMI count)', fontsize=12)
    plt.ylabel('UMI Count (Total Molecules)', fontsize=12)
    plt.title('Forensic Reconstruction: The "Knee" Plot (Cell vs Empty)', fontsize=14)
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)

    # Zoom in on the transition zone relevant to PBMC3k
    plt.xlim(1, 100000) 
    plt.ylim(1, 100000)
    
    output_plot = "results/figures/forensic_knee_plot.png"
    plt.savefig(output_plot, dpi=150)
    print("   ------------------------------------------------")
    print(f"    -> PLOT GENERATED: {output_plot}")
    
    # FORENSIC VERDICT
    # Let's see the UMI count of the last "real" cell vs the first "empty" one
    last_real_cell_umis = counts[official_cell_count-1]
    first_empty_drop_umis = counts[official_cell_count]

    print("   ------------------------------------------------")
    print("\n THE DECISION BOUNDARY:")
    print(f"   Rank #{official_cell_count} (Last Cell):  {last_real_cell_umis} UMIs")
    print(f"   Rank #{official_cell_count+1} (First Trash): {first_empty_drop_umis} UMIs")
    print("   ------------------------------------------------")
    print("   Interpretation: The software drew the line here.")
    print("   Any droplet with fewer than ~1000 UMIs was condemned as 'Background'.")
    print("   ------------------------------------------------")
if __name__ == "__main__":
    generate_knee_plot()

