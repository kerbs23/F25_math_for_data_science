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
from torch.nn.init import _calculate_correct_fan

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



'''
print('Training the test model...')
start_time = time.time()
#classifier = MLPClassifier(max_iter=10000, verbose=True).fit(X_train, y_train)
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
                          max_iter=10000, verbose=True)
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
                         max_iter=10000, verbose=True)
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
                         max_iter=10000, verbose=True)
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
                         max_iter=10000, verbose=True)
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
                         max_iter=10000, verbose=True)
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
                         max_iter=10000, verbose=True)
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
                         max_iter=10000, verbose=True)
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
                         max_iter=10000, verbose=True)
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


print("\n=== Testing The big one ===")
start_time = time.time()
model = MLPClassifier(hidden_layer_sizes=(841, 841, 841, 500, 300, 69), activation='relu', solver='adam',
                     learning_rate='constant', learning_rate_init=.001, early_stopping=True,
                     max_iter=10000, verbose=True)
model.fit(X_train, y_train)
training_time = time.time() - start_time
time_taken, accuracy, cm = evaluate_model(model, training_time=training_time)

results.append({
    'hidden_layers': 6,
    'neurons': '841, 841, 841, 500, 300, 69',
    'activation': 'relu',
    'solver': 'adam',
    'learning_rate': 'constant',
    'learning_rate_init': 0.001,
    'early_stopping': False,
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
print(df[['hidden_layers', 'neurons', 'activation', 'solver', 'learning_rate', 'learning_rate_init', 'early_stopping', 'accuracy', 'training_time']])


'''


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
import matplotlib.pyplot as plt
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(filename='torch_crap.log', encoding='utf-8', filemode='w', level=logging.INFO)


#%%

# In colab, you should ``change runtime type'' to GPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

'''
# Some boilerplate from the docs to view the dataset. Comments mine
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    # Take a random integer on the reported length, and get it as an integer, rather then 1 element torch df
    sample_idx = torch.randint(len(train_ds), size=(1,)).item()
    # Use numpy syntax to get the randomly chosen row
    img, torch_label = train_ds[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[torch_label.item()]) #pyright: ignore
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()
'''

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten() #turns the square images into 1 dim
        self.linear_relu_section = nn.Sequential(#This lets you string together a chain of
                                               # nn api things together into a single 
                                               # callable thing
                                               nn.Linear(28*28, 512), #An affine transform
                                               #In this case, has 28x28 input layers and
                                               #512 output layers
                                               nn.ReLU(), # ReLU all of the inputs
                                               #Returns the same # of outputs 
                                               nn.Linear(512, 512),
                                               nn.ReLU(),
                                               nn.Linear(512, 10),
                                               nn.ReLU(),
                )
        self.softmax = nn.Softmax(dim=1) # Scales the logits into probabilities
    def forward(self, x): # Defines the order in which to call the things as we do a
        # forward pass in the NN
        x = self.flatten(x)
        logits = self.linear_relu_section(x)
        probabilities = self.softmax(logits)
        return probabilities

model = NeuralNetwork().to(device)
print(f"Model structure: {model}\n\n")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")



# Hyperparamaters for training
learning_rate = .001
batch_size = 64
epochs = 5

# the loss function is just an nn. method that we pass the crap to
loss_fn = nn.NLLLoss()
# similar for the optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

def train_loop(dataloader, model, loss_fn, optimizer):
    train_time = time.time()
    logging.debug(f'{train_time}: started a training loop')
    size = len(dataloader.dataset)
    # Set the model to training mode-- a best practice
    model.train()
    for batch, (X, y) in enumerate(dataloader): # Recall dataloaders are a set of a couple
        # samples in the data. So this is saying essentially for each of the bach, set the
        # X and y properly
        batch_time = time.time()
        logging.debug(f'{batch_time}: Starting a batch loop')
        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward() # use the mega-differentiation thing to compute the gradients for
        # the whole nn
        optimizer.step() # take a step in the right direction
        optimizer.zero_grad() # we have to manually 0 the gradient each time
        done_time = time.time()
        loss, current, taken=loss.item(),batch * batch_size + len(X),done_time-batch_time
        logging.debug(f'{done_time}: finished this batch, loss: {loss:>7f} [{current:>5d}/{size:>5d}], Took {taken}')
    done_time = time.time()
    logging.debug(f'{done_time}: completed training, took {train_time-done_time}')

def test_loop(dataloader, model, loss_fn):
    model.eval() # set to evaluate mode
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad(): # An optimization so as to not keep track of grad while testing
        for X, y in dataloader:
            pred = model (X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            

    test_loss /= num_batches
    correct /= size
    print(f'Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n')
    logger.debug(f'Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n')



epochs = 100
start_time = time.time()
print(f'Started training at {start_time}')
logger.info(f'Started training at {start_time}')

for t in range(epochs):
    logger.debug(f'Epoch {t}\n======================================')
    train_loop(train_loader, model, loss_fn, optimizer)
    test_loop(val_loader, model, loss_fn)
done_time = time.time()
print(f'{done_time}: finished training. Took {done_time-start_time}\n')
logger.info(f'{done_time}: finished training. Took {done_time-start_time}\n')
test_loop(test_loader, model, loss_fn)




