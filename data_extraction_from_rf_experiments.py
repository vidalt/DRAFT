###Libraries imports
import pandas as pd
import numpy as np
import sklearn
import json
import argparse
import time
from sklearn.ensemble import RandomForestClassifier

from local_config import ccanada_expes, pegasus_expes
from datasets_infos import datasets_ohe_vectors, predictions
from utils import * 

parser = argparse.ArgumentParser(description='Dataset reconstruction from random forest')
parser.add_argument('--expe_id', type=int, default=0)
args = parser.parse_args()
expe_id=args.expe_id

if ccanada_expes:
    debug = False
    n_threads = 16
    print_logs = 0
    rank = expe_id
    local_tests = 0
# Added to be able to do runs in Waterloo's server (only applicable if not
# ccanada_expes)
elif pegasus_expes:
    debug = True
    n_threads = 16
    print_logs = 1
    rank = expe_id
    local_tests = 0
else:
    # Manually set parameters
    debug = True
    n_threads = 8 # -1 = all threads available
    print_logs = 1
    rank = expe_id
    local_tests = 1

time_out =  5*60*60
n_random_sols=100
datasets = ["compas", "adult", "default_credit"]
sample_size = 1500
train_size = 10
#method = = "milp" (requires GUROBI) or "cp-sat" (requires ORTOOLS)
#method = "bench" tries to find out how much one could possibly do, with knowledge of data
#method = "bench_partial_xxxx" runs the experiments by fixing part of the reconstructed data (partial recovery)
# xxxx can be either 'examples' or 'attributes'
method = "bench_partial_attributes"
if method in ['cp-sat', 'milp', 'bench']:
    val_trees = [1, 5, 10, 20,30,40,50,60,70,80,90,100]
    val_depths = [None, 2, 3,4,5,10]
    val_seed = [i for i in range(5)]

    if method == 'cp-sat':
        bagging_list = [True, False]
    elif method == 'milp':
        bagging_list = [False]
    elif method == 'bench':
        bagging_list = [True]

    params_list = []
    for bag in bagging_list:
        for t in val_trees:
            for d in val_depths:
                for s in val_seed:
                    for dd in datasets:
                        params_list.append([bag, t, d, s, dd])
elif method == "bench_partial_examples":
    val_trees = [100]
    val_depths = [None]
    val_seed = [i for i in range(5)]
    bagging_list = [True]
    known_proportion_list = [0.0, 0.01, 0.05]
    known_proportion_list.extend([np.round(0.1*b,2) for b in range(1,10)])
    known_proportion_list.extend([0.95, 0.99])
    
    params_list = []

    for bag in bagging_list: # 210 combinations
        for t in val_trees:
            for d in val_depths:
                for s in val_seed:
                    for p in known_proportion_list:
                        for dd in datasets:
                            params_list.append([bag, t, d, s, dd, p])
    known_proportion = params_list[rank][5]
elif method == "bench_partial_attributes":
    val_trees = [100]
    val_depths = [None]
    val_seed = [i for i in range(5)]
    bagging_list = [True]
    #nattrs_datasets = {"compas":14, "adult":19, "default_credit":21} 
    nattrs_datasets = {"compas":14-4-3, "adult":19-5, "default_credit":21-3-2} # All attributes OHEncoding the same original feature count only once
    params_list = []

    for bag in bagging_list: # 185 combinations
        for t in val_trees:
            for d in val_depths:
                for s in val_seed:
                    for dd in datasets:
                        for nattrs in range(nattrs_datasets[dd]):
                            params_list.append([bag, t, d, s, dd, nattrs])

    known_attributes_nb = params_list[rank][5]

bagging = params_list[rank][0]
n_estimators = params_list[rank][1]
max_depth_t =  params_list[rank][2]
seed = params_list[rank][3]
dataset = params_list[rank][4]

np.random.seed(seed)

if local_tests: # for local runs
    train_size = 10
    n_estimators = 10
    max_depth_t = None
    bagging = False
    dataset = "compas"
    method = "cp-sat" 
    known_proportion = 0.8
    known_attributes_nb = 2
    seed = 0#params_list[rank][2]

if debug:
    print("#params: ", len(params_list))
    print("Local Config: ")
    print(" dataset = ", dataset)
    print(" n_estimators = ", n_estimators)
    print(" max_depth_t = ", max_depth_t)
    print(" seed = ", seed)

# Prepare data
ohe_vector = datasets_ohe_vectors[dataset]

df = pd.read_csv("data/%s.csv" %dataset)
#print("dataset %s" %dataset, df)
#exit()
#df.drop([0,1],axis=0,inplace=True)

df = df.sample(n=sample_size, random_state = seed, ignore_index= True)

X_train, X_test, y_train, y_test = data_splitting(df, predictions[dataset], sample_size - train_size, seed)
X_train = X_train.to_numpy()
if debug:
    print("Using dataset %s, training set size is %d with %d attributes." %(dataset, train_size, X_test.shape[1]))
    checked_ohe = check_ohe(X_train, datasets_ohe_vectors[dataset])
    print("OHE verified: ", checked_ohe)
    if not checked_ohe:
        exit()

# Train and evaluate model
clf = RandomForestClassifier(max_depth = max_depth_t, n_estimators = n_estimators, bootstrap = bagging, random_state = seed) # criterion='gini'

clf = clf.fit(X_train, y_train)

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import numpy as np

savetrees = 0
if savetrees:
    for tid, t in enumerate( clf.estimators_ ):
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_tree(t,ax=ax)
        fig.savefig("./tree_%d.pdf" % (tid), bbox_inches='tight')
        plt.clf()
saveforest = 0
if saveforest:
    fig, axes_list = plt.subplots(1, n_estimators)
    #fig.suptitle('Example Random Forest')
    for i in range(n_estimators):
        plot_tree(clf.estimators_[i],ax=axes_list[i])
    fig.savefig("./forest.png", bbox_inches='tight')
    plt.clf()



accuracy_train = clf.score(X_train, y_train)
accuracy_test = clf.score(X_test, y_test)
if debug:
    print("accuracy_train=", accuracy_train, "accuracy_test=",accuracy_test)
# Perform the reconstruction
from DRAFT import DRAFT

extractor = DRAFT(clf, datasets_ohe_vectors[dataset])

if method == "bench":
    dict_res = extractor.perform_reconstruction_benchmark(X_train, y_train, n_threads=n_threads, time_out=time_out, verbosity=debug, seed=seed)
elif method == "bench_partial_examples":
    assert(bagging == True)
    n_known = int(known_proportion*X_train.shape[0])
    X_known = X_train[0:n_known]
    y_known = y_train[0:n_known]
    dict_res = extractor.perform_benchmark_partial_examples(n_threads=n_threads, time_out=time_out, verbosity=debug, seed=seed, X_known = X_known, y_known=y_known)
elif method == "bench_partial_attributes":
    assert(bagging == True)
    assert(nattrs_datasets[dataset] == X_train.shape[1] + len(ohe_vector) - sum([len(one_ohe) for one_ohe in ohe_vector]))
    # everything below is to convert the choice of original features into the actual choice of encoded (with OHE) attrs
    known_original_attributes_list = np.random.choice([i for i in range(nattrs_datasets[dataset])],size=known_attributes_nb, replace=False) # randomly pick the known_attributes_nb (original) attributes we assume knowledge of
    original_attributes_list = []
    one_encoded_attr = 0
    while one_encoded_attr < X_train.shape[1]:
        for one_ohe in ohe_vector:
            if one_encoded_attr in one_ohe:
                original_attributes_list.append(one_ohe)
                one_encoded_attr = max(one_ohe) + 1
                continue
        original_attributes_list.append([one_encoded_attr])
        one_encoded_attr += 1
    known_attributes_list = []
    for one_original_attr in known_original_attributes_list:
        known_attributes_list.extend(original_attributes_list[one_original_attr])
    # -----------------------------------------------------------------------------------------------------------------
    unknown_attributes_list = list(set([i for i in range(X_train.shape[1])]) - set(known_attributes_list))
    dict_res = extractor.perform_benchmark_partial_attributes(n_threads=n_threads, time_out=time_out, verbosity=debug, seed=seed, X_known = X_train, known_attributes=known_attributes_list)
else:
    dict_res = extractor.fit(bagging=bagging, method=method, timeout=time_out, verbosity=debug, n_jobs=n_threads, seed=seed) # 'status':solve_status, 'duration': duration, 'reconstructed_data':x_sol

solve_status = dict_res['status']
duration = dict_res['duration']
x_sol = dict_res['reconstructed_data']
max_max_depth = dict_res['max_max_depth']

if method == "bench_partial_examples": # only consider the unknown examples for the evaluation
    x_sol = x_sol[n_known:]
    X_train = X_train[n_known:]

# Evaluate the reconstruction
x_train_list = X_train.tolist() # ground truth

e_mean, list_matching = average_error(x_sol,x_train_list)

# For partial reconstr attr perform matching with entire dataset but eval only on subset
if method == "bench_partial_attributes": # only consider the unknown attributes for the evaluation
    '''moyenne = 0
    for i in range(len(x_train_list)):
        moyenne += dist_individus(x_sol[i], x_train_list[list_matching[i]])
    moyenne = moyenne/len(x_train_list)
    assert(moyenne == e_mean) # just a check
    if debug:
        print("Error entire dataset:", moyenne)'''
    x_sol = np.asarray(x_sol)
    x_sol = x_sol[:,unknown_attributes_list]
    X_train = X_train[:,unknown_attributes_list]
    x_train_list_unknown = X_train.tolist() # ground truth
    moyenne = 0
    for i in range(len(x_train_list_unknown)):
        moyenne += dist_individus(x_sol[i], x_train_list_unknown[list_matching[i]])
    moyenne = moyenne/len(x_train_list_unknown)
    if debug:
        print("Error only unknown attrs same matching:", moyenne)
    e_mean = moyenne

moyenne_rand = 0
liste_random_sol = generate_random_sols(len(x_train_list), X_test.shape[1], dataset_ohe_groups=datasets_ohe_vectors[dataset], n_sols=n_random_sols, seed=seed)
if method == "bench_partial_attributes":   # Again, matching is computed on the entire examples
    x_train_list = x_train_list_unknown    # But evaluation only on unknown samples
for e in liste_random_sol:
    if method == "bench_partial_attributes":      
        e = np.asarray(e)[:,unknown_attributes_list] # only consider the unknown attributes for the random baseline evaluation as well
    moyenne_rand += average_error(e, x_train_list)[0]
moyenne_rand = moyenne_rand/len(liste_random_sol)

if solve_status == 'UNKNOWN':
    e_mean = moyenne_rand 

sol_dict = {}
sol_dict["values"] = {"n_trees": n_estimators,
                    "max_depth": max_depth_t,
                    "real_max_depth": max_max_depth,
                    "seed": seed,
                    "accuracy test": accuracy_test,
                    "accuracy train": accuracy_train,
                    "solve_status": solve_status,
                    "mean-error": e_mean,
                    "random_error": moyenne_rand,
                    "solve_duration_time": duration}

filename = str(dataset) + "_" + str(n_estimators) +"_"+ str(max_depth_t)+ "_" + str(seed) + "_bagging-" + str(bagging) + "_" + str(method)

if method == "bench_partial_examples":
    filename += "_" + str(known_proportion)
    sol_dict["values"]["known_proportion"] = known_proportion
if method == "bench_partial_attributes":
    filename += "_" + str(known_attributes_nb)
    sol_dict["values"]["known_attributes"] = known_attributes_nb

with open(f"./results/{filename}.json", 'w') as f:
    json.dump(sol_dict, f, indent =4)

if debug:
    print("Complete solving duration :", duration)
    print("Reconstruction Error: ", e_mean)
    print("Baseline (Random) Error: ", moyenne_rand)


# Possible improvement:
'''
Consider that we know whether there are more often zeros or ones in the entire training set. 
Baseline: full zeros or full ones

if np.average(X_train) > 0.5:
    randlist.append(np.ones((N,M)).tolist())
else:
    randlist.append(np.zeros((N,M)).tolist())


MILP: maximize or minimize the training set variables' sum.

m.setObjective(quicksum(x[k,i] for i in range(M) for k in range(N)), GRB.MINIMIZE)
'''


