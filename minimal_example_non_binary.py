# Load packages
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from utils import * 
from datasets_infos import datasets_ohe_vectors, predictions, datasets_ordinal_attrs, datasets_numerical_attrs
import numpy as np 

script_seed = 42
sample_size = 1500
train_size = 40
dataset = "default_credit_numerical"
debug_check = True # To perform additional checks and display information

# Load data information and dataset
ohe_vector = datasets_ohe_vectors[dataset] # list of sublists indicating sets of binary attributes one-hot-encoding the same original one
ordinal_attrs = datasets_ordinal_attrs[dataset] # list of ordinal attributes
numerical_attrs = datasets_numerical_attrs[dataset] # list of numerical attributes
prediction = predictions[dataset] # the attribute we want to predict

df = pd.read_csv("data/%s.csv" %dataset)

# --------- Set tight lower and upper bounds for the ordinal and numerical attributes ----------
# -------------- (as for now we assume knowledge of the attributes' domains) -------------------
# Note that they are computed on the whole dataset and may not be tight for the training set ---
# to avoid biasing the results -----------------------------------------------------------------
if debug_check:
    for index_attr, attr in enumerate(df):
        for num_index, num_f in enumerate(ordinal_attrs):
            if num_f[0] == index_attr:
                ordinal_attrs[num_index][1] = min(df[attr])
                ordinal_attrs[num_index][2] = max(df[attr])
        for num_index, num_f in enumerate(numerical_attrs):
            if num_f[0] == index_attr:
                numerical_attrs[num_index][1] = min(df[attr])
                numerical_attrs[num_index][2] = max(df[attr])
# ----------------------------------------------------------------------------------------------

# Data sampling
df = df.sample(n=sample_size, random_state = script_seed, ignore_index= True)
X_train, X_test, y_train, y_test = data_splitting(df, prediction, test_size=sample_size-train_size, seed=script_seed)
X_train = X_train.to_numpy()
X_test = X_test.to_numpy()

# --------- (debug only / verify correctness of the provided dataset information) --------------
if debug_check:
    checked_ohe = check_ohe(X_train, ohe_vector)
    print("OHE verified: ", checked_ohe)
    checked_ordinal_domains = check_domain(X_train, ordinal_attrs)
    print("Ordinal attributes domains verified: ", checked_ordinal_domains)
    checked_numerical_domains = check_domain(X_train, numerical_attrs)
    print("Numerical attributes domains verified: ", checked_numerical_domains)
    if checked_ohe and checked_ordinal_domains and checked_numerical_domains:
        print("Dataset information correct.\n")
    else:
        raise ValueError("Dataset information error.")
# ----------------------------------------------------------------------------------------------

# Train a Random Forest (without bootstrap sampling)
clf = RandomForestClassifier(random_state = script_seed)
clf = clf.fit(X_train, y_train)

# Reconstruct the Random Forest's training set
from DRAFT import DRAFT

extractor = DRAFT(clf, one_hot_encoded_groups=ohe_vector, ordinal_attributes=ordinal_attrs, numerical_attributes=numerical_attrs)
dict_res = extractor.fit(timeout=60, seed=script_seed) 

# Retrieve solving time and reconstructed data
duration = dict_res['duration']
x_sol = dict_res['reconstructed_data']

# --------- (debug only / verify correctness of the reconstructed dataset information) --------------
if debug_check:
    x_sol_np = np.asarray(x_sol)
    checked_ohe = check_ohe(x_sol_np, ohe_vector)
    print("(reconstruction) OHE verified: ", checked_ohe)
    checked_ordinal_domains = check_domain(x_sol_np, ordinal_attrs)
    print("(reconstruction) Ordinal attributes domains verified: ", checked_ordinal_domains)
    checked_numerical_domains = check_domain(x_sol_np, numerical_attrs)
    print("(reconstruction) Numerical attributes domains verified: ", checked_numerical_domains)
    if checked_ohe and checked_ordinal_domains and checked_numerical_domains:
        print("Reconstructed dataset information correct.\n")
    else:
        raise ValueError("Dataset information/reconstruction error.")
# ----------------------------------------------------------------------------------------------

# Evaluate the reconstruction error
e_mean, list_matching = average_error(x_sol, X_train, dataset_ordinal=ordinal_attrs, dataset_numerical=numerical_attrs)

# Print the pairs of original/reconstructed examples
if debug_check:
    for k in range(len(x_sol)):
        print("Reconstructed example %d:" %k, x_sol[k])
        print("Original example %d:" %k, list(X_train[list_matching[k]]))
        print("dist=",dist_individus(x_sol[k],list(X_train[list_matching[k]]), non_binary_attrs=ordinal_attrs+numerical_attrs))
        print()

print("Complete solving duration :", duration)

print("Reconstruction Error: ", e_mean)

# Evaluate random reconstruction error
nb_random_sols = 100
random_sols = generate_random_sols(X_train.shape[0], X_train.shape[1], dataset_ohe_groups=ohe_vector, n_sols=nb_random_sols, seed=script_seed, dataset_ordinal=ordinal_attrs, dataset_numerical=numerical_attrs)
rand_sum = 0
for e in random_sols:
    rand_sum += average_error(e, X_train, dataset_ordinal=ordinal_attrs, dataset_numerical=numerical_attrs)[0]
rand_avg = rand_sum/len(random_sols)
print("Baseline (Random) Error: ", rand_avg)

