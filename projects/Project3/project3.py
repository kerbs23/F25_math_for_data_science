# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: hydrogen
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown] id="0gdC70xxFyc4"
# Math 5750/6880: Mathematics of Data Science \
# Project 3

# %% [markdown] id="i9_7SnpMGKDJ"
# # 1. Fashion-MNIST image classification using sklearn

# %% id="AB136H0PGKq1"
# ruff: noqa: E402 F401

from cProfile import label
from tensorflow.keras.datasets import fashion_mnist #pyright: ignore
from sklearn.preprocessing import StandardScaler

# Load Fashion-MNIST
# Classes (0-9): T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
X_train = X_train.reshape(len(X_train), -1)
X_test  = X_test.reshape(len(X_test), -1)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# %% id="5GAsN-dmHjRM"
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time
import pandas as pd
import numpy as np
from datetime import datetime




print('Training the test model...')
start_time = time.time()
#classifier = MLPClassifier(verbose=True).fit(X_train, y_train)
training_time = time.time() - start_time
print(f'Took {training_time}')

def evaluate_model(model, X_test = X_test, y_test = y_test, training_time = training_time):
    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    target_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle-boot']

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred=y_pred, normalize=True)
    print(classification_report(y_test, y_pred, labels=labels, target_names=target_names))
    cm = confusion_matrix(y_test, y_pred)
    print(confusion_matrix)
    return training_time, accuracy, cm




# Cool that works well. Just going to define the test cases then write a series of loops over them,
# and save it all to one big array so that I can format that as a table for the reports

base_case = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='sgd',
                          learning_rate='constant', learning_rate_init=.001, early_stopping=False
                          )

hidden_layers = [2, 4, 8, 16]
neurons = [20, 50, 200, 500]
activations = ['logistic', 'tanh']
solvers = ['adam','lbfgs']
learning_rate = ['invscaling', 'adaptive']
learning_rate_init = [.1, .01, .0001]
early_stopping = [True]



# Initialize results list
results = []

# Test base case first
print("\n=== Testing Base Case ===")
start_time = time.time()
base_model = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='sgd',
                          learning_rate='constant', learning_rate_init=.001, early_stopping=False,
                          verbose=False)
base_model.fit(X_train, y_train)
training_time = time.time() - start_time
time_taken, accuracy, cm = evaluate_model(base_model, training_time=training_time)

results.append({
    'hidden_layers': 1,
    'neurons': 100,
    'activation': 'relu',
    'solver': 'sgd',
    'learning_rate': 'constant',
    'learning_rate_init': 0.001,
    'early_stopping': False,
    'training_time': time_taken,
    'accuracy': accuracy,
    'confusion_matrix': cm
})

# Test hidden layers
for n_layers in hidden_layers:
    print(f"\n=== Testing {n_layers} hidden layers ===")
    start_time = time.time()
    model = MLPClassifier(hidden_layer_sizes=tuple([100] * n_layers), activation='relu', solver='sgd',
                         learning_rate='constant', learning_rate_init=.001, early_stopping=False,
                         verbose=False)
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    time_taken, accuracy, cm = evaluate_model(model, training_time=training_time)
    
    results.append({
        'hidden_layers': n_layers,
        'neurons': 100,
        'activation': 'relu',
        'solver': 'sgd',
        'learning_rate': 'constant',
        'learning_rate_init': 0.001,
        'early_stopping': False,
        'training_time': time_taken,
        'accuracy': accuracy,
        'confusion_matrix': cm
    })

# Test neurons
for n_neurons in neurons:
    print(f"\n=== Testing {n_neurons} neurons ===")
    start_time = time.time()
    model = MLPClassifier(hidden_layer_sizes=(n_neurons,), activation='relu', solver='sgd',
                         learning_rate='constant', learning_rate_init=.001, early_stopping=False,
                         verbose=False)
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    time_taken, accuracy, cm = evaluate_model(model, training_time=training_time)
    
    results.append({
        'hidden_layers': 1,
        'neurons': n_neurons,
        'activation': 'relu',
        'solver': 'sgd',
        'learning_rate': 'constant',
        'learning_rate_init': 0.001,
        'early_stopping': False,
        'training_time': time_taken,
        'accuracy': accuracy,
        'confusion_matrix': cm
    })

# Test activations
for activation in activations:
    print(f"\n=== Testing {activation} activation ===")
    start_time = time.time()
    model = MLPClassifier(hidden_layer_sizes=(100,), activation=activation, solver='sgd',
                         learning_rate='constant', learning_rate_init=.001, early_stopping=False,
                         verbose=False)
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    time_taken, accuracy, cm = evaluate_model(model, training_time=training_time)
    
    results.append({
        'hidden_layers': 1,
        'neurons': 100,
        'activation': activation,
        'solver': 'sgd',
        'learning_rate': 'constant',
        'learning_rate_init': 0.001,
        'early_stopping': False,
        'training_time': time_taken,
        'accuracy': accuracy,
        'confusion_matrix': cm
    })

# Test solvers
for solver in solvers:
    print(f"\n=== Testing {solver} solver ===")
    start_time = time.time()
    model = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver=solver,
                         learning_rate='constant', learning_rate_init=.001, early_stopping=False,
                         verbose=False)
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    time_taken, accuracy, cm = evaluate_model(model, training_time=training_time)
    
    results.append({
        'hidden_layers': 1,
        'neurons': 100,
        'activation': 'relu',
        'solver': solver,
        'learning_rate': 'constant',
        'learning_rate_init': 0.001,
        'early_stopping': False,
        'training_time': time_taken,
        'accuracy': accuracy,
        'confusion_matrix': cm
    })

# Test learning rate types
for lr_type in learning_rate:
    print(f"\n=== Testing {lr_type} learning rate ===")
    start_time = time.time()
    model = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='sgd',
                         learning_rate=lr_type, learning_rate_init=.001, early_stopping=False,
                         verbose=False)
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    time_taken, accuracy, cm = evaluate_model(model, training_time=training_time)
    
    results.append({
        'hidden_layers': 1,
        'neurons': 100,
        'activation': 'relu',
        'solver': 'sgd',
        'learning_rate': lr_type,
        'learning_rate_init': 0.001,
        'early_stopping': False,
        'training_time': time_taken,
        'accuracy': accuracy,
        'confusion_matrix': cm
    })

# Test learning rate initializations
for lr_init in learning_rate_init:
    print(f"\n=== Testing {lr_init} learning rate init ===")
    start_time = time.time()
    model = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='sgd',
                         learning_rate='constant', learning_rate_init=lr_init, early_stopping=False,
                         verbose=False)
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    time_taken, accuracy, cm = evaluate_model(model, training_time=training_time)
    
    results.append({
        'hidden_layers': 1,
        'neurons': 100,
        'activation': 'relu',
        'solver': 'sgd',
        'learning_rate': 'constant',
        'learning_rate_init': lr_init,
        'early_stopping': False,
        'training_time': time_taken,
        'accuracy': accuracy,
        'confusion_matrix': cm
    })

# Test early stopping
for early_stop in early_stopping:
    print(f"\n=== Testing early_stopping={early_stop} ===")
    start_time = time.time()
    model = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='sgd',
                         learning_rate='constant', learning_rate_init=.001, early_stopping=early_stop,
                         verbose=False)
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    time_taken, accuracy, cm = evaluate_model(model, training_time=training_time)
    
    results.append({
        'hidden_layers': 1,
        'neurons': 100,
        'activation': 'relu',
        'solver': 'sgd',
        'learning_rate': 'constant',
        'learning_rate_init': 0.001,
        'early_stopping': early_stop,
        'training_time': time_taken,
        'accuracy': accuracy,
        'confusion_matrix': cm
    })

# Save results to CSV
df = pd.DataFrame(results)
# Convert confusion matrices to strings for CSV
df['confusion_matrix'] = df['confusion_matrix'].apply(lambda x: str(x).replace('\n', ';'))
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_filename = f'mlp_results_{timestamp}.csv'
df.to_csv(csv_filename, index=False)
print(f"\nResults saved to {csv_filename}")
print(f"\nTotal models tested: {len(results)}")
print("\nSummary:")
print(df[['hidden_layers', 'neurons', 'activation', 'solver', 'learning_rate', 'learning_rate_init', 'early_stopping', 'accuracy', 'training_time']])

# %% [markdown] id="a2qcKggmIH8T"
# # 3. Fashion-MNIST image classification  using pytorch

# %% id="B9IQwhgcIVOl"
import numpy as np
from tensorflow.keras.datasets import fashion_mnist
import torch
from torch.utils.data import TensorDataset, DataLoader

# Load Fashion-MNIST
# Classes (0-9): T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# scale to [0,1], add channel dimension -> (N, 1, 28, 28)
X_train = (X_train.astype("float32") / 255.0)[:, None, :, :]
X_test  = (X_test.astype("float32")  / 255.0)[:,  None, :, :]

y_train = y_train.astype(np.int64)
y_test  = y_test.astype(np.int64)

# train/val split: last 10k of train as validation
X_tr, X_val = X_train[:50000], X_train[50000:]
y_tr, y_val = y_train[:50000], y_train[50000:]

# wrap in PyTorch TensorDatasets and DataLoaders
train_ds = TensorDataset(torch.from_numpy(X_tr),  torch.from_numpy(y_tr))
val_ds   = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
test_ds  = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=256, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False)

# %% id="0REsDBunNmEl"
import torch.nn as nn
import torch.optim as optim

# In colab, you should ``change runtime type'' to GPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# your code here
