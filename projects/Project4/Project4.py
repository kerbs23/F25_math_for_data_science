# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown] id="ucxGNyGI_vs7"
# Math 5750/6880: Mathematics of Data Science \
# Project 4

# %% [markdown] id="nWcBGhqY_a1I"
# # 1. Exploratory Analysis
#
# Use the following code to download the dataset from
# [https://www.kaggle.com/code/mineshjethva/eda-pulsedb/notebook](https://www.kaggle.com/code/mineshjethva/eda-pulsedb/notebook). The dataset is described in the paper [https://doi.org/10.3389/fdgth.2022.1090854](https://doi.org/10.3389/fdgth.2022.1090854).
#
# I would recommend saving the data files to a google drive (or your local machine) so that you don't have to download them again. Note that the 5 data files correspond to the 5 columns in Table 4 of the paper.
#

# %% id="-4AP1y_pxclv"
# download the data from kagglehub
# The dataset is 17.3 G
# This took about 15min using university wifi and, if
# you save the data, you should only have to do it once
#/home/u1329310/.local/share/Trash/files/pulsedb-balanced-training-and-testing.zip

# %% id="pSQFg0mK8cgd"
# load the data
# the subject information is stored in a pandas df
# the Signals (ECG, PPG, ABP) are stored in numpy arrays
# this block takes 11 minutes to execute

# .parquet exists?! gonna turn it into that with polars, then just load those all nice
# That might also let me take a small one off cade which would be good for the membrane.
# Commented all this out bc one only really has to run it once.

"""
import polars as pl
import h5py
import os
import glob
import gc
import numpy as np

DATA_DIR = "/var/tmp/u1329310"


def convert_mat_to_parquet(file_path, output_path):
    '''Convert large .mat file to .parquet using memory-efficient h5py'''
    
    # Open the HDF5 file
    with h5py.File(file_path, 'r') as f:
        subset = f['Subset']
        
        # Get dimensions
        signals = subset['Signals']
        n_samples = signals.shape[0]
        signal_length = signals.shape[2]
        
        print(f"Processing {n_samples} samples with signal length {signal_length}")
        
        # Process metadata first (these are small)
        metadata = {}
        print("Processing metadata")
        for key in ['Age', 'BMI', 'DBP', 'Height', 'SBP', 'Weight']:
            data = subset[key][:]
            if data.dtype == np.object_:  # Handle string arrays
                metadata[key] = [x[0] if isinstance(x, np.ndarray) else x for x in data.flatten()]
            else:
                metadata[key] = data.flatten().tolist()
        
        # Handle Gender (string array) - may contain HDF5 references
        gender_data = subset['Gender'][:]
        gender_list = []
        for ref in gender_data.flatten():
            if hasattr(ref, '__getitem__') and len(ref) > 0:
                # Direct string access
                if isinstance(ref[0], bytes):
                    gender_list.append(1 if ref[0] == b'M' else 0)
                elif isinstance(ref[0], str):
                    gender_list.append(1 if ref[0] == 'M' else 0)
            else:
                # Handle HDF5 reference by dereferencing
                try:
                    deref = f[ref]
                    if isinstance(deref[()], bytes):
                        gender_list.append(1 if deref[()] == b'M' else 0)
                    elif isinstance(deref[()], str):
                        gender_list.append(1 if deref[()] == 'M' else 0)
                    else:
                        gender_list.append(0)  # Default
                except:
                    gender_list.append(0)  # Default on error
        metadata['Gender'] = gender_list
        
        # Handle Subject (string array) - may contain HDF5 references
        subject_data = subset['Subject'][:]
        subject_list = []
        for ref in subject_data.flatten():
            if hasattr(ref, '__getitem__') and len(ref) > 0:
                # Direct string access
                if isinstance(ref[0], bytes):
                    subject_list.append(ref[0].decode('utf-8'))
                elif isinstance(ref[0], str):
                    subject_list.append(ref[0])
            else:
                # Handle HDF5 reference by dereferencing
                try:
                    deref = f[ref]
                    if isinstance(deref[()], bytes):
                        subject_list.append(deref[()].decode('utf-8'))
                    elif isinstance(deref[()], str):
                        subject_list.append(deref[()])
                    else:
                        subject_list.append("unknown")  # Default
                except:
                    subject_list.append("unknown")  # Default on error
        metadata['Subject'] = subject_list
        
        # Process signals in chunks
        chunk_size = 500  # Adjust based on available memory
        all_chunks = []
        
        for i in range(0, n_samples, chunk_size):
            end = min(i + chunk_size, n_samples)
            
            # Read only current chunk of signals
            signals_chunk = signals[i:end, :, :]
            
            # Extract individual signals
            print("starting ECG chunk")
            ECG_chunk = signals_chunk[:, 0, :].tolist()
            print("starting PPG chunk")
            PPG_chunk = signals_chunk[:, 1, :].tolist()
            print("starting ABP chunk")
            ABP_chunk = signals_chunk[:, 2, :].tolist()
            
            # Create chunk DataFrame
            chunk_data = {
                'ECG': ECG_chunk,
                'PPG': PPG_chunk,
                'ABP': ABP_chunk,
                'Age': metadata['Age'][i:end],
                'BMI': metadata['BMI'][i:end],
                'DBP': metadata['DBP'][i:end],
                'Gender': metadata['Gender'][i:end],
                'Height': metadata['Height'][i:end],
                'SBP': metadata['SBP'][i:end],
                'Subject': metadata['Subject'][i:end],
                'Weight': metadata['Weight'][i:end]
            }
            
            chunk_df = pl.DataFrame(chunk_data)
            all_chunks.append(chunk_df)
            
            print(f"Processed samples {i} to {end-1}")
            
            # Clean up
            del signals_chunk, ECG_chunk, PPG_chunk, ABP_chunk, chunk_data, chunk_df
            gc.collect()
        
        # Combine all chunks
        df = pl.concat(all_chunks)
        
        print("\n=== PARQUET FILE ANALYSIS ===")
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {len(df.columns)}")
        print(f"Rows: {len(df)}")
        
        # Write to parquet
        df.write_parquet(output_path)
        print(f"Saved {output_path}")
        
        # Final cleanup
        del df, all_chunks, metadata
        gc.collect()

# Convert files
# Find all .mat files in DATA_DIR
mat_files = glob.glob(os.path.join(DATA_DIR, "*.mat"))
print(f"Found .mat files: {mat_files}")

# Create parquets directory if it doesn't exist
os.makedirs(f"{DATA_DIR}/parquets", exist_ok=True)

# Convert each .mat file to .parquet
for mat_file in mat_files:
    base_name = os.path.splitext(os.path.basename(mat_file))[0]
    parquet_file = f"{DATA_DIR}/parquets/{base_name}.parquet"
    print(f"\nConverting {mat_file} → {parquet_file}")
    convert_mat_to_parquet(mat_file, parquet_file)
    gc.collect()
"""


# %% id="jIc-z3zuJWlQ"
import polars as pl
import glob
# Let us begin by loading in a sample from the data and create some tools for veiwing the timeseries

DATA_DIR = "/var/tmp/pulsedb-balanced-training-and-testing"


# Function to examine the data
def examine_parquet_data(file):
    # Load first 1000 rows from the parquet file
    df = pl.read_parquet(file, n_rows=1000)

    print("\n=== PARQUET FILE ANALYSIS ===")
    
    # Column names and types
    print(f"\nColumn names and types:")
    for col in df.columns:
        dtype = df[col].dtype
        print(f"  {col}: {dtype}")
    
    # First row contents
    print(f"\nFirst row contents:")
    first_row = df.row(0)
    for i, (col, value) in enumerate(zip(df.columns, first_row)):
        print(f"  {col}: {value}")
    
    # Basic statistics
    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {len(df.columns)}")
    print(f"Rows: {len(df)}")
    
    return df
    

parquet_files = glob.glob("data/*.parquet")
for file in parquet_files:
    examine_parquet_data(file)
    



# %% [markdown] id="rhZ5XU2rIpLU"
# #2. Blood Pressure Prediction
#

# %% id="Bl2N87kvImVI"
# your code here

# %% [markdown] id="l0Qg1Sm8JP2p"
# #3. Generative Modeling

# %% id="dEoQzvV4JPFI"
# your code here
