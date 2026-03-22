import h5py
import pandas as pd
import numpy as np

# CONFIGURATION
FILE_PATH = "data/raw/pbmc3k_molecule_info.h5"

print(f" UNLOCKING LEGACY H5: {FILE_PATH}")

with h5py.File(FILE_PATH, 'r') as f:
    # 1. EXTRACT THE COLUMNS (Using your discovered keys)
    # We slice [:] to load the data into RAM.
    
    print("   -> Extracting 'barcode' (Cell Index)...")
    barcode_col = f['barcode'][:]
    
    print("   -> Extracting 'gene' (Gene Index)...")
    gene_col = f['gene'][:]
    
    print("   -> Extracting 'umi' (Molecular Tag)...")
    umi_col = f['umi'][:]
    
    print("   -> Extracting 'reads' (Count Depth)...")
    reads_col = f['reads'][:]

    print("   -> Extracting 'GEM_group' (GEM WELLS)...")
    gem_col = f['gem_group'][:]

    # 2. CHECK ALIGNMENT (Forensic Verification)
    # In a valid file, all these columns must be the same length.
    if not (len(barcode_col) == len(gene_col) == len(reads_col)):
        print(f" DATA CORRUPTION: Column lengths differ!")
        print(f"   Barcodes: {len(barcode_col)}")
        print(f"   Genes:    {len(gene_col)}")
        exit()

    # 3. BUILD THE DATAFRAME
    print(f" Alignment Confirmed (N={len(barcode_col):,} molecules). Building DataFrame...")
    
    df = pd.DataFrame({
        'cell_id_int': barcode_col,
        'gene_id_int': gene_col,
        'umi_encoded': umi_col,
        'read_count': reads_col,
        'gem_num': gem_col
    })

    # 4. INSPECTION
    print("\n MOLECULAR LOGS (HEAD):")
    print(df.head())
    
    print("\n STATISTICS:")
    print(f"   Total Reads Processed: {df['read_count'].sum():,}")
    print(f"   Unique Molecules:      {len(df):,}")

# Note: The 'cell_id_int' and 'gene_id_int' are integers (0, 1, 2...).
# To know that '0' = 'AAACATAC...', you would typically look for a 'barcodes' 
# lookup table. If it's missing from the keys you listed, this file 
# relies on the external 'barcodes.tsv' for the translation.