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

import kagglehub
path = kagglehub.dataset_download("weinanwangrutgers/pulsedb-balanced-training-and-testing")
print("Path to dataset files:", path)

# %% id="RnU6-XNKBpoz"
# run this block to move the data to a permanent directory in your drive

import os, glob
DATA_DIR = "/content/drive/MyDrive/pulsedb/"
# !mkdir -p $DATA_DIR
# !cp -r $path/* $DATA_DIR

# %% id="Cb3z3VM-xd1s"
# run this block after data is saved to your drive

import os, glob
from google.colab import drive
drive.mount('/content/drive')
DATA_DIR = "/content/drive/MyDrive/pulsedb/"

mat_files = sorted(glob.glob(os.path.join(DATA_DIR, "**", "*.mat"), recursive=True))
print(f"Found {len(mat_files)} .mat files")
for f in mat_files:
    print(" -", f)

# %% [markdown] id="fybIsP4zmCeV"
# Now we'll load the data. The data is about 20GB, which exceeds the colab basic RAM allocation. You can check your RAM using
#
# `!cat /proc/meminfo`
#
# You should upgrade to colab pro, which is free for students.
#
# [https://colab.research.google.com/signup](https://colab.research.google.com/signup)
#
# Then in 'change runtime type' click A100 GPU and high RAM.

# %% id="pSQFg0mK8cgd"
# load the data
# the subject information is stored in a pandas df
# the Signals (ECG, PPG, ABP) are stored in numpy arrays
# this block takes 11 minutes to execute

# !cat /proc/meminfo

# !pip install mat73
import mat73
import pandas as pd
import numpy as np

def load_mat_file(file_path):
    data_dict = mat73.loadmat(file_path)['Subset']
    print('finished loading'+file_path)
    # print(data_dict.keys())

    # first handle Signals
    ECG = data_dict['Signals'][:,0,:]
    PPG = data_dict['Signals'][:,1,:]
    ABP = data_dict['Signals'][:,2,:]
    data_dict.pop("Signals", None)

    data_dict['Age'] = data_dict['Age'].tolist()
    data_dict['BMI'] = data_dict['BMI'].tolist()
    data_dict['DBP'] = data_dict['DBP'].tolist()
    data_dict['Gender'] = [1 if x[0] == 'M' else 0 for x in data_dict['Gender']]
    data_dict['Height'] = data_dict['Height'].tolist()
    data_dict['SBP'] = data_dict['SBP'].tolist()
    data_dict['Subject'] = [x[0] for x in data_dict['Subject']]
    data_dict['Weight'] = data_dict['Weight'].tolist()

    data_df = pd.DataFrame(data_dict)
    print('constructed df')

    return data_df, ECG, PPG, ABP

df_CalBased_Test, ECG_CalBased_Test, PPG_CalBased_Test, ABP_CalBased_Test = load_mat_file(DATA_DIR+'VitalDB_CalBased_Test_Subset.mat')
df_Train, ECG_Train, PPG_Train, ABP_Train = load_mat_file(DATA_DIR+'VitalDB_Train_Subset.mat')

# %% id="GfBz5kcvEPRS"
# df_CalBased_Test has 51720 entries
print(df_CalBased_Test.keys())
print(df_CalBased_Test.info())
print(df_CalBased_Test.describe())
df_CalBased_Test

# %% id="xP89OfDQEqwH"
# 1293 subjects, 40 samples/ subject = 51720 samples
df_CalBased_Test['Subject'].value_counts()

# %% id="jIc-z3zuJWlQ"
# your code here

# %% [markdown] id="rhZ5XU2rIpLU"
# #2. Blood Pressure Prediction
#

# %% id="Bl2N87kvImVI"
# your code here

# %% [markdown] id="l0Qg1Sm8JP2p"
# #3. Generative Modeling

# %% id="dEoQzvV4JPFI"
# your code here
