# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: hydrogen
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: python 3
#     name: python3
# ---

# %% [markdown] id="9fm9p3ve85y3"
# math 5750/6880: mathematics of data science \\
# project 1

# %% [markdown] id="vodhelnjxdro"
# # 3. python and google colab
# project euler problem  
# https://projecteuler.net/

# %% id="y_e6lm_newwd"
# pyright: basic
# ruff: noqa: e402
wd = "/home/magnoliad/Desktop/F2025/math_data_sci/projects/Project1"


# conjecture: this is a prime factorization problem kinda
# two numbers are relatively prime if their gcf == 1. so, if a # is not relatively prime to 2, then
# it is not relatively prime to all multiples of 2.
# similarly, if it is relatively prime to 2, and we add the condition it is relatively prime to 3, 
# it gains the multiples of 3 that are not multiples of 2 to its score.

# so, each distinct prime factor helps us, and each prime that is not a factor hurts us. and, there
# are diminishing returns to higher prime factors. this gives me an idea:
import math as m
from ntpath import exists
from os import makedirs, path
def phi(num, loud = False):
    rel_primes = []
    for n in range(num):
        if m.gcd(n, num) == 1:
            rel_primes.append(n)
    if loud:
        print(f"relative primes of {num}:{rel_primes}, phi = {len(rel_primes)}") #to test it
    return len(rel_primes)

def score(num):
    return num/phi(num)


print (phi(1*2*3*5, True)) #works
print(score(1*2*3*5)) #3.75

# so, the vibes tell me that the result will be the collection of primes multiplied togegher.
# not just that, but the frequency of primes that coung against us mechanicdally shrinks as we
# go up, so itll be the biggest of these in the range.

guess = 1*2*3*5*7*11*13*17 # 19 takes us too high
print(guess) # == 510510
print(score(guess)) # == 5.539 ish

# is my educated guess. however, i didnt write that nice score function to not then let the computer brute force it for me

import pandas as pd

# generate scores for n in range. takes a couple mins on my labtop but its ok
# ok fine ill admit doing this on 1 thread was not my wisest move ever. but its ok
scores = []
for n in range(1, 10): #set to 1000000 to see i am right
    scores.append((n, score(n)))
    print("\r",end="")
    print(f'checking num: {n}', end="")

# create dataframe and sort
df = pd.DataFrame(scores, columns=['n', 'score'])
df = df.sort_values('score', ascending=False)
print(df.head)
# shows i am right.


# %% [markdown] id="riltqhjexnoq"
# # 4. regression analysis
# california housing data  
# https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset
#

# %% id="5gtrmgu1kl6x"
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# %% colab={"base_uri": "https://localhost:8080/"} id="ydjnvnagkrpe" outputid="b9f703a8-5008-4b11-8dc9-a3636d7efd56"
# load the california housing data
cal = fetch_california_housing(as_frame=True)
x, y = cal.data, cal.target # pyright: ignore
feature_names = x.columns
print(feature_names)

# train/test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)

# %% id="peytefssxmk2"
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge

import os

print(x_train.columns)

# i do linear regression essentially professionally, so this section will be uniquely overbuilt
# first time using scikit-learn though so it is not totally free
# first, from experience i want a log() of the price columns, which changes the interpretation to roughly the % change in house prices on observables
# generally, finance crap changes as % of its total so this makes sense in that way

def y_prep(y):
    log_y = np.log(y)
    return log_y

def x_prep(x):
    x['log_medinc'] = np.log(x['MedInc']) # my priors are basically that medinc will moetly predict the outcome
    return x

def evaluate_regression(model, x_data, y_data):
    r2 = model.score(x_data, y_data)
    y_pred = model.predict(x_data)
    residuals = y_data - y_pred
    
    # calculate metrics
    mse = np.mean(residuals**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(residuals))
    
    # calculate coefficient standard errors
    n = len(y_data)
    p = x_data.shape[1]
    mse_residual = np.sum(residuals**2) / (n - p - 1)
    x_with_intercept = np.column_stack([np.ones(n), x_data]) if model.fit_intercept else x_train
    cov_matrix = mse_residual * np.linalg.inv(x_with_intercept.T @ x_with_intercept)
    std_errors = np.sqrt(np.diag(cov_matrix))
    
    # get coefficients (add intercept if present)
    coefs = np.concatenate([[model.intercept_], model.coef_]) if model.fit_intercept else model.coef_
    
    # create result row with coefficients
    result_data = {
        'r2': r2,
        'mse': mse,
        'rmse': rmse, 
        'mae': mae,
        'n_samples': n,
        'n_features': p
    }
    
    # add coefficients and standard errors
    for i, (coef, std_err) in enumerate(zip(coefs, std_errors)):
        result_data[f'coef_{i}'] = coef
        result_data[f'std_err_{i}'] = std_err
    
    return pd.DataFrame([result_data])

def plot_estimate_vs_actual(model, x_data, y_true, title=None, save_path=None):
    y_pred = model.predict(x_data)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter( y_true, y_pred, alpha=0.6, edgecolors='w', linewidth=0.5)
    ax.axline((0, 0), slope=1, color='r', linestyle='--', alpha=0.8)
    ax.set_xlabel('True values')
    ax.set_ylabel('Predicted values')
    if title:
        ax.set_title(title)

    if save_path:
        plt.savefig(save_path, dpi = 300)
    plt.show()
    return fig, ax


def plot_residuals_histogram(model, x_data, y_true, title=None, save_path=None):
    y_pred = model.predict(x_data)
    residuals = y_true - y_pred
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
    ax.axvline(x=0, color='r', linestyle='--', alpha=0.8)
    ax.set_xlabel('Residuals')
    ax.set_ylabel('Frequency')
    if title:
        ax.set_title(title)

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()
    return fig, ax


x_train_processed = x_prep(x_train)
y_train_processed = y_prep(y_train)

# first, lets look at each varible one by one without prep
results = []

for feature in x_train_processed.columns:
    # Fit simple linear regression for each feature
    X_single = x_train_processed[[feature]]  # Keep as DataFrame for feature name
    model = LinearRegression().fit(X_single, y_train_processed)
    evaluation = evaluate_regression(model, X_single, y_train_processed)
    evaluation['feature'] = feature
    results.append(evaluation)

results = pd.concat(results, axis=0, ignore_index=True)

print(results)
results.to_csv('coeffs.csv')

# Checking it out in dv, by far the highest r2 is for log_medinc, followed by ave_rooms
# It only looks like the population is clearly insignificant, although the ave bedrms and ave occp do not inspire confidence
# Interestingly, the log-log on the medinc has a slightly lower r2, but I think Ill leave it as is.
# also, I think the lat-long are misspecified. splitting them up into 2 interacted categoricals would probably help,
# but also be a lot of work and not the point of this assign.

# So next I think I will use all of the regressors except population, and see if anything looses its significance

plots_dir = f'{wd}/plots/'
os.makedirs(plots_dir, exist_ok=True)

x_spec_2 = x_train_processed[['HouseAge', 'AveRooms', 'AveBedrms', 'AveOccup', 'Latitude', 'Longitude', 'log_medinc']]
model = LinearRegression().fit(x_spec_2, y_train_processed)
evaluation = evaluate_regression(model, x_spec_2, y_train_processed)
print(evaluation)
evaluation.to_csv('coeffs2_OLS_train.csv')
error_chart = plot_estimate_vs_actual(model, x_spec_2, y_train_processed, title="Spec 2 pred v actual comparison", save_path=f'{plots_dir}Spec_2_pred_v_actual')
residual_histogram = plot_residuals_histogram(model, x_spec_2, y_train_processed, title="Spec 2 residual histogram", save_path=f'{plots_dir}Spec_2_residual_hist')
# Happy with that, now to validate with the test data

x_test_processed = x_prep(x_test)
y_test_processed = y_prep(y_test)
x_spec_2_test = x_test_processed[['HouseAge', 'AveRooms', 'AveBedrms', 'AveOccup', 'Latitude', 'Longitude', 'log_medinc']]
evaluation = evaluate_regression(model, x_spec_2_test, y_test_processed)
print(evaluation)
evaluation.to_csv('coeffs2_OLS_test.csv')
error_chart = plot_estimate_vs_actual(model, x_spec_2_test, y_test_processed, title="Spec 2 pred v actual comparison (test data)", save_path=f'{plots_dir}Spec_2_pred_v_actual_test')
residual_histogram = plot_residuals_histogram(model, x_spec_2_test, y_test_processed, title="Spec 2 residual histogram (test data)", save_path=f'{plots_dir}Spec_2_residual_hist_test')


# Ill use ridge regression? I really dont get any of these things and this seems to  be the most comprehensible.

x_spec_2 = x_train_processed[['HouseAge', 'AveRooms', 'AveBedrms', 'AveOccup', 'Latitude', 'Longitude', 'log_medinc']]
model = Ridge().fit(x_spec_2, y_train_processed)
evaluation = evaluate_regression(model, x_spec_2, y_train_processed)
print(evaluation)
evaluation.to_csv('coeffs2_Ridge_train.csv')
error_chart = plot_estimate_vs_actual(model, x_spec_2, y_train_processed, title="Spec 2 pred v actual comparison", save_path=f'{plots_dir}Spec_2_pred_v_actual')
residual_histogram = plot_residuals_histogram(model, x_spec_2, y_train_processed, title="Spec 2 residual histogram", save_path=f'{plots_dir}Spec_2_residual_hist')
# Happy with that, now to validate with the test data

x_test_processed = x_prep(x_test)
y_test_processed = y_prep(y_test)
x_spec_2_test = x_test_processed[['HouseAge', 'AveRooms', 'AveBedrms', 'AveOccup', 'Latitude', 'Longitude', 'log_medinc']]
evaluation = evaluate_regression(model, x_spec_2_test, y_test_processed)
print(evaluation)
evaluation.to_csv('coeffs2_Ridge_test.csv')
error_chart = plot_estimate_vs_actual(model, x_spec_2_test, y_test_processed, title="Spec 2 pred v actual comparison (test data)", save_path=f'{plots_dir}Spec_2_pred_v_actual_test')
residual_histogram = plot_residuals_histogram(model, x_spec_2_test, y_test_processed, title="Spec 2 residual histogram (test data)", save_path=f'{plots_dir}Spec_2_residual_hist_test')

# Well, no difference. I think that makes sense because there isnt a lot of colinearity here, which is what it says
# the ridge regression is better for.

# on to the next thing I suppose


# %% [markdown] id="QkSVMB7HXSZB"
# # 5. Classification Analysis
# Diagnostic Wisconsin Breast Cancer Database  
# https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-dataset

# %% id="vUfQhEk7zQBX"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# %% colab={"base_uri": "https://localhost:8080/"} id="IGeihULRzZxD" outputId="b7cff0d4-9149-4208-c92c-6fedb809d25b"
# Load Breast Cancer Wisconsin Dataset
data = load_breast_cancer(as_frame=True)
X = data.data
y = data.target                  # 0 = malignant, 1 = benign
feature_names = X.columns
label_names = {0: "malignant", 1: "benign"}
print(feature_names)

# Train/Test Split (stratified to preserve class balance)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0, stratify=y)

# Preprocess Data (fit on train ONLY; then transform both)
scaler = StandardScaler(with_mean=True, with_std=True)
X_train_std = scaler.fit_transform(X_train)   # fit on train
X_test_std  = scaler.transform(X_test)        # transform test with train stats

# %% id="QUYeDq2ZXY2x"
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, confusion_matrix, RocCurveDisplay, PrecisionRecallDisplay
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

print(X_train_std)
print(y_train)

# well ok similar situation.

def evaluate_model_performance(model, X, y_true, set_name="", feature_names=None):
    y_pred = model.predict(X)
    y_score = model.predict_proba(X)[:, 1]  # Gets probability for the positive class
    
    accuracy = accuracy_score(y_true, y_pred) 
    roc_auc = roc_auc_score(y_true, y_score)
    avg_precision = average_precision_score(y_true, y_score)
    
    # Print results
    print(f"\n--- {set_name} Set Performance ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"Average Precision: {avg_precision:.4f}")
    
    top_features = []
    if feature_names is not None and hasattr(model, 'coef_'):
        importance = np.abs(model.coef_[0])  # Absolute value of coefficients
        sorted_indices = np.argsort(importance)[::-1]  # Sort descending, get indices
        top_indices = sorted_indices[:5]  # First 5 indices
        
        print("\nTop 5 features by importance:")
        for idx in top_indices:
            print(f"{feature_names[idx]}: {importance[idx]:.4f}")
            top_features.append((feature_names[idx], importance[idx]))

    return {'accuracy': accuracy, 'roc_auc': roc_auc, 'avg_precision': avg_precision}

def generate_model_plots(model, X, y_true, set_name=None, save_path=None):
    """
    Generates and displays a confusion matrix, ROC curve, and Precision-Recall curve.
    """
    import matplotlib.pyplot as plt
    
    y_pred = model.predict(X)
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    if set_name:
        fig.suptitle(f'{set_name}', fontweight='bold')
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    ax1.matshow(cm, cmap='Blues', alpha=0.7)
    ax1.set_title('Confusion Matrix')
    for (i, j), val in np.ndenumerate(cm):
        ax1.text(j, i, f'{val}', ha='center', va='center')
    
    # ROC Curve
    RocCurveDisplay.from_estimator(model, X, y_true, ax=ax2)
    ax2.set_title('ROC Curve')
    ax2.plot([0, 1], [0, 1], 'k--')  # Diagonal line for random classifier
    
    # Precision-Recall Curve
    PrecisionRecallDisplay.from_estimator(model, X, y_true, ax=ax3)
    ax3.set_title('Precision-Recall Curve')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi = 300)
    plt.show()

model = SVC(probability=True, kernel='linear', random_state=42)# linear kernel lets us get the most important param

model.fit(X_train_std, y_train)

# Use the above functs to evaluate the Performance on the training data:
evaluate_model_performance(model, X_train_std, y_train, "SVC Training Performance Eval", feature_names=data.feature_names)
generate_model_plots(model, X_train_std, y_train, "SVC Training Performance Eval", save_path=f'{plots_dir}SVC_training_eval')

# and now the test data:
evaluate_model_performance(model, X_test_std, y_test, "SVC Test Performance Eval", feature_names=data.feature_names)
generate_model_plots(model, X_test_std, y_test, "SVC Test Performance Eval", save_path=f'{plots_dir}SVC_test_eval')
# takes a bit of a hit but well within things still being fine
# seems like the most important params are mean concave points, radius error, and worst fractal dimension. Whatever those are.
# Now onto another method. Used KNN, seems interesting and I think I kinda get it.
# Though im not sure it is strictly classification but.
results = []
for n in range(2, 20, 2):
    knn_model = KNeighborsClassifier(n_neighbors=n)
    knn_model.fit(X_train_std, y_train)
    evaluation = evaluate_model_performance(knn_model, X_test_std, y_test)
    evaluation['n_neighbors'] = n
    results.append(evaluation)

results_df = pd.DataFrame(results)
results_df.to_csv("knn_comparisons.csv")
# Looks best at 8, so

knn_model = KNeighborsClassifier(n_neighbors=8)
knn_model.fit(X_train_std, y_train)
evaluate_model_performance(knn_model, X_test_std, y_test, "knn Test Performance Eval")
generate_model_plots(knn_model, X_test_std, y_test, "knn Test Performance Eval", save_path=f'{plots_dir}knn_training_eval')

