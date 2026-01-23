import h5py
import pandas as pd
import numpy as np

FILE_PATH = "/Users/qgem/GitHub/PBMC3k-reproducible/data/raw/pbmc3k_molecule_info.h5"

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
    
    # 2. CHECK ALIGNMENT (Forensic Verification)
    # In a valid file, all these columns must be the same length.
    if not (len(barcode_col) == len(gene_col) == len(reads_col)):
        print(f"❌ DATA CORRUPTION: Column lengths differ!")
        print(f"   Barcodes: {len(barcode_col)}")
        print(f"   Genes:    {len(gene_col)}")
        exit()

    # 3. BUILD THE DATAFRAME
    print(f"✅ Alignment Confirmed (N={len(barcode_col):,} molecules). Building DataFrame...")
    
    df = pd.DataFrame({
        'cell_id_int': barcode_col,
        'gene_id_int': gene_col,
        'umi_encoded': umi_col,
        'read_count': reads_col
    })

    # 4. INSPECTION
    print("\n🔹 MOLECULAR LOGS (HEAD):")
    print(df.head())
    print (df)