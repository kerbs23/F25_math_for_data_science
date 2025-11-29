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
# Test edit - tool functionality verified

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
'''
# Transform the data from .mat to .parquet files
# load the data
# the subject information is stored in a pandas df
# the Signals (ECG, PPG, ABP) are stored in numpy arrays
# this block takes 11 minutes to execute

# .parquet exists?! gonna turn it into that with polars, then just load those all nice
# That might also let me take a small one off cade which would be good for the membrane.
# Commented all this out bc one only really has to run it once.

import polars as pl
import mat73
import os
import glob
import gc

DATA_DIR = "/var/tmp/u1329310"


def convert_mat_to_parquet(file_path, output_path):
    """
    Reads all of the .mat files in file_path, writes them to parquet files in output_path
    only works for pulsedb
    
    For memory reasons, for each .parquet Ill load the time signals one at a time and save them to parqet
    individually, then do the person-level metadata, then use the polars streaming stuff to put it all 
    back together into a single file.
    """
    
    # Write .parquets for all of the signals data
    def process_signal(index_num, name, output_path):
        # Load only the Signals array (not the entire Subset)
        data_dict = mat73.loadmat(mat_file, only_include='Subset/Signals')
        signals = data_dict['Subset']['Signals']
        signal = signals[:,index_num,:]  # Extract ECG slice

        # Bin the rest of the data. Slows things down a lot but keeps the memory happy
        del signals
        del data_dict
        gc.collect()

        df = pl.DataFrame({name: signal.tolist()})
        del signal
        df.write_parquet(output_path)

    signal_dictionary = {
            'EKG': 0,
            'PPG': 1,
            'ABP': 2,
            }
    for i, n in signal_dictionary.items():
        signal_parqet = f'{output_path}/{i}.parquet'
        os.makedirs(f'{output_path}', exist_ok=True)
        process_signal(n, i, signal_parqet)


    # Write a .parquet for the metadata
    data_dict = mat73.loadmat(file_path)['Subset']
    data_dict.pop('Signals', None)

    data_dict['Age'] = data_dict['Age'].tolist()
    data_dict['BMI'] = data_dict['BMI'].tolist()
    data_dict['DBP'] = data_dict['DBP'].tolist()
    data_dict['SBP'] = data_dict['SBP'].tolist()
    data_dict['Height'] = data_dict['Height'].tolist()
    data_dict['Weight'] = data_dict['Weight'].tolist()
    data_dict['Gender'] = [1 if x[0] == 'M' else 0 for x in data_dict['Gender']]
    data_dict['Subject'] = [x[0] for x in data_dict['Subject']]

    metadata_df = pl.DataFrame(data_dict)
    metadata_file = f"{output_path}/metadata.parquet"
    metadata_df.write_parquet(metadata_file)
    
    # Now, use the straming api from polars to combine the 3 dfs rowwise in a memory safe way
    
    # Combine signal files using lazy evaluation
    signal_files = [f"{output_path}/{signal}.parquet" for signal in ['EKG', 'PPG', 'ABP']]
    # Read all signal files lazily and concatenate horizontally
    combined_signals = pl.concat(
        [pl.scan_parquet(file) for file in signal_files],
        how="horizontal"
    )
    
    metadata_file = f"{output_path}/metadata.parquet"
    metadata_lazy = pl.scan_parquet(metadata_file)
    final_combined = pl.concat([combined_signals, metadata_lazy], how = "horizontal")

    # Execute and write combined signals
    final_combined.sink_parquet(f"{output_path}_final.parquet")



# Convert files
# Find all .mat files in DATA_DIR
mat_files = glob.glob(os.path.join(DATA_DIR, "*.mat"))
print(f"Found .mat files: {mat_files}")

# Create parquets directory if it doesn't exist
os.makedirs(f"{DATA_DIR}/parquets", exist_ok=True)

# Convert each .mat file to .parquet
for mat_file in mat_files:
    base_name = os.path.splitext(os.path.basename(mat_file))[0]
    parquet_file = f"{DATA_DIR}/parquets/{base_name}"

    convert_mat_to_parquet(mat_file, parquet_file)
'''

# %% id="jIc-z3zuJWlQ"
'''
# Sumarize the data and generate some histograms and such
import polars as pl
import glob
import os
import matplotlib.pyplot as plt
# Let us begin by loading in a sample from the data and create some tools for veiwing the timeseries

DATA_DIR = "/var/tmp/u1329310/"
PARQUETS_DIR = "/var/tmp/u1329310/parquets/"


# Function to examine the data
def examine_parquet_data(file):
    # Load first 1000 rows from the parquet file
    df = pl.read_parquet(file, n_rows=5)

    print("\n=== PARQUET FILE ANALYSIS ===")
    
    # Column names and types
    print("\nColumn names and types:")
    for col in df.columns:
        dtype = df[col].dtype
        print(f"  {col}: {dtype}")
    
    # First row contents
    print("\nFirst row contents:")
    first_row = df.row(0)
    for i, (col, value) in enumerate(zip(df.columns, first_row)):
        print(f"  {col}: {value}")
    
    # Basic statistics
    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {len(df.columns)}")
    print(f"Rows: {len(df)}")
    
    return df


# Function to calculate statistics for numeric columns and save as CSV and histograms
def parquet_analysis(file_path, output_path):
    """
    Calculate mean, standard deviation, and count for all numeric columns
    in a Parquet file and save results as CSV and histograms.
    
    Parameters:
    file_path (str): Path to input .parquet file
    output_path (str): Path to output files
    """

    # Get file size
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    print(f"File size: {file_size_mb:.2f} MB")

    # Read the parquet file
    df = pl.read_parquet(file_path)
    
    # Select only numeric columns
    numeric_cols = [col for col in df.columns if df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8]]
    
    if not numeric_cols:
        print(f"No numeric columns found in {file_path}")
        return
    
    print(f"Found {len(numeric_cols)} numeric columns: {numeric_cols}")

    # Check for list columns
    list_columns = [col for col in df.columns if str(df[col].dtype).startswith('List')]

    # Calculate statistics for each numeric column
    stats_data = {}
    for col in numeric_cols:
        stats_data[f"{col}_mean"] = df[col].mean()
        stats_data[f"{col}_std"] = df[col].std()
        stats_data[f"{col}_n"] = df[col].count()
    # Add lengths for the list columns
    if list_columns:
        print(f"\nList columns found: {list_columns}")
        for col in list_columns:
            sample_value = df[col][0]
            if hasattr(sample_value, '__len__'):
                print(f"  {col}: list length ~{len(sample_value)}")
                stats_data[f"{col}_len"] = len(sample_value)


    # Create statistics DataFrame
    stats_df = pl.DataFrame(stats_data)
    
    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)
    
    # Save to CSV
    csv_path = f"{output_path}/statistics.csv"
    stats_df.write_csv(csv_path)
    print(f"Statistics saved to {output_path}")
    
    # Generate histograms for numeric columns
    for col in numeric_cols:
        plt.figure(figsize=(8, 6))
        plt.hist(df[col].to_numpy(), bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel(col)
        plt.tight_layout()
        plt.savefig(f"{output_path}/{col}_density.png", dpi=150, bbox_inches='tight')
        plt.close()
    print(f"Histograms saved to {output_path}")
    
    # Generate time series plots for list columns
    plot_list_column_timeseries(df, output_path)
    
    return stats_df

# Function that takes a df that has some rows that are arrays, and plots the first row's first bit as a timeseries for each
def plot_list_column_timeseries(df, output_path):
    """
    Plot the first 500 elements of list columns as time series from the first row.
    Done seperately so I can just add the pred timeseries to the df and compare with this same functon

    Parameters:
    df (pl.DataFrame): Polars DataFrame with list columns
    output_path (str): Directory to save the plots
    """
    # Identify list columns
    list_columns = [col for col in df.columns if str(df[col].dtype).startswith('List')]
    
    if not list_columns:
        print("No list columns found for time series plotting")
        return
    
    print(f"Generating time series plots for list columns: {list_columns}")
    
    for col in list_columns:
        # Get the first row's list data
        first_row_data = df[col][0]
        
        if hasattr(first_row_data, '__len__') and len(first_row_data) > 0:
            # Take first 100 elements or all if less than 100
            plot_data = first_row_data[:500]
            
            plt.figure(figsize=(10, 4))
            plt.plot(plot_data, linewidth=1)
            plt.xlabel('Sample Index')
            plt.ylabel(col)
            plt.tight_layout()
            plt.savefig(f"{output_path}/{col}_timeseries.png", dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Saved {col}_timeseries.png ({len(plot_data)} samples)")
        else:
            print(f"  Skipping {col}: no valid list data in first row")

parquet_files = glob.glob(f"{PARQUETS_DIR}*.parquet")
print(parquet_files)

# Create balance_check directory if it doesn't exist
os.makedirs("balance_check", exist_ok=True)

for file in parquet_files:
    base_name = os.path.splitext(os.path.basename(file))[0]

    # This just confirms the integrity of the parquet files
    sample_df = examine_parquet_data(file)
    
    output_path = f"balance_check/{base_name}"
    parquet_analysis(file, output_path)

'''
# %% [markdown] id="rhZ5XU2rIpLU"
# #2. Blood Pressure Prediction
def load_parquets_to_tensors(data_path):
    """
    Take a .parqet at data_path and load it up as a pytorch tensor
    """
# Linear regression
#%%
"""
Not super sure where to go with this one.
I guess normally with panel data like this I would do a huge twfe thing, with all the covariates,
but since the time dimension is not actually time in like a date sense but actually interval from an arbitrary start,
the time dim is meaningless in that sense.

I guess I could unwind the data such that each time observation of the 3 vars. is its own row and just do a regular ols on that?
That is going to take FOREVER to run though
It looks like the play might be to do it on pytorch rather then some dedicated regression platform, so I can use the gpu.

"""

# %% id="Bl2N87kvImVI"
# your code here

# %% [markdown] id="l0Qg1Sm8JP2p"
# #3. Generative Modeling

# %% id="dEoQzvV4JPFI"
# your code here
