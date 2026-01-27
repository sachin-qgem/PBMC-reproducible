import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.io as sio
import os
import gzip
import shutil

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
H5_PATH = "data/raw/pbmc3k_molecule_info.h5"
# We still need the gene names from the file you have, because H5 lost them
GENES_TSV_PATH = "data/raw/pbmc3k_filtered_gene_bc_matrices/hg19/genes.tsv"
OUTPUT_DIR = "data/reconstructed_matrices_final"

def decode_barcode(n, length=16):
    """
    Decodes a 2-bit integer back to ACGT string.
    10x Mapping: A=0, C=1, G=2, T=3
    Note: The length depends on Chemistry (V2=16bp, V1=14bp).
    We will auto-detect or default to 16.
    """
    # 10x uses a specific bit-packing. 
    # Usually: A=00, C=01, G=10, T=11 (or similar map).
    # Standard: {0:'A', 1:'C', 2:'G', 3:'T'}
    bases = ['A', 'C', 'G', 'T']
    
    # We construct the string in reverse order of division
    seq = []
    for _ in range(length):
        remainder = n % 4
        seq.append(bases[remainder])
        n = n // 4
    
    return "".join(reversed(seq))

def reconstruct():
    print(" SATURN PROTOCOL: FINAL RECONSTRUCTION (DECODING MODE)")

    # 1. LOAD GENE NAMES (The Dictionary)
    print("   -> Loading Gene definitions...")
    genes_df = pd.read_csv(GENES_TSV_PATH, sep='\t', header=None, names=['id', 'name'])
    gene_ids = genes_df['id'].values
    gene_names = genes_df['name'].values
    n_genes_ref = len(genes_df)
    
    # 2. LOAD H5 DATA
    print("   -> Reading H5 Events...")
    with h5py.File(H5_PATH, 'r') as f:
        # Load Raw Data
        bc_indices = f['barcode'][:]  # The huge integers
        gene_indices = f['gene'][:]
        
        # Handle 'count' if it exists (for pre-aggregated H5)
        if 'count' in f:
            counts = f['count'][:]
        else:
            # If no count, every row is 1 molecule
            counts = np.ones(len(bc_indices), dtype=np.uint16)

    print(f"   -> Loaded {len(bc_indices):,} events.")
    print(f"   -> Max Gene Index found: {gene_indices.max()} (Ref Size: {n_genes_ref})")

    # 3. FILTER THE GARBAGE BIN (Gene Index 32738)
    # The H5 often has an extra index for "None".
    # We strictly keep only indices that exist in our Gene TSV (0 to 32737).
    valid_mask = gene_indices < n_genes_ref
    
    if not valid_mask.all():
        n_dropped = (~valid_mask).sum()
        print(f"    REMOVING {n_dropped} EVENTS (Unmapped/Garbage Genes).")
        bc_indices = bc_indices[valid_mask]
        gene_indices = gene_indices[valid_mask]
        counts = counts[valid_mask]

    # 4. COMPRESS BARCODES (Integer -> 0..N Index)
    # We have 15 million events, but many share the same barcode integer.
    # We need to find the UNIQUE barcodes and assign them new indices (0, 1, 2...)
    print("   -> Finding Unique Barcodes...")
    
    # specialized function 'unique' sorts and returns indices
    unique_bc_ints, inverse_indices = np.unique(bc_indices, return_inverse=True)
    n_barcodes_unique = len(unique_bc_ints)
    
    print(f"   -> Found {n_barcodes_unique:,} Unique Barcodes (The Raw Count).")

    # 5. DECODE BARCODES (Int -> String)
    # This is the magic step. We convert the unique integers to strings.
    print("   -> Decoding Barcode Sequences (This takes CPU)...")
    # PBMC3k is V1/V2 chemistry. Let's assume 16bp (V2) or 14bp (V1).
    # Based on max val 266M, it fits in 14 bits? 4^14 = 268M.
    # So it is likely 14 base pairs.
    BARCODE_LEN = 14 
    
    # Vectorize or list comp
    decoded_barcodes = [decode_barcode(i, BARCODE_LEN) + "-1" for i in unique_bc_ints]
    # Note: "-1" is the Gem Group suffix standard in 10x files.

    # 6. BUILD THE RAW MATRIX
    # Rows = Unique Barcodes (0..N)
    # Cols = Genes
    # Data = Counts
    # Use 'inverse_indices' which maps every event to its new Barcode ID (0..N)
    print("   -> Constructing Sparse Matrix...")
    
    raw_matrix = sp.coo_matrix(
        (counts, (inverse_indices, gene_indices)), 
        shape=(n_barcodes_unique, n_genes_ref)
    ).tocsr() #compress the Coordinates

    # 7. SAVE RAW DATA
    raw_out = os.path.join(OUTPUT_DIR, "raw_gene_bc_matrices")
    print(f"   -> Saving Raw Matrix to {raw_out}...")
    write_10x(raw_matrix, decoded_barcodes, gene_ids, gene_names, raw_out)

    print(" RECONSTRUCTION COMPLETE.")

def write_10x(matrix, barcodes, g_ids, g_names, folder):
    os.makedirs(folder, exist_ok=True)
    # Matrix
    mat_path = os.path.join(folder, "matrix.mtx")
    sio.mmwrite(mat_path, matrix.transpose()) # Transpose to Genes x Cells
    with open(mat_path, 'rb') as f_in, gzip.open(mat_path + '.gz', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.remove(mat_path)
    
    # Barcodes
    with open(os.path.join(folder, "barcodes.tsv"), 'wt') as f:
        for b in barcodes: f.write(f"{b}\n")
        
    # Features
    with open(os.path.join(folder, "genes.tsv"), 'wt') as f:
        for i, n in zip(g_ids, g_names): f.write(f"{i}\t{n}\n")

if __name__ == "__main__":
    reconstruct()