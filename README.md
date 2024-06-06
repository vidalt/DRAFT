# DRAFT: Dataset Reconstruction Attack From Trained ensembles. 

This repository contains the implementation of our proposed dataset reconstruction attacks against Random Forests, introduced in the paper **"Trained Random Forests Completely Reveal your Dataset"** authored by Julien Ferry, Ricardo Fukasawa, Timothée Pascal, and Thibaut Vidal (2024). 

## Installation

Our attack is implemented as a Python module contained in the `DRAFT.py` file.

Depending on the particular method used to conduct the dataset reconstruction attack, a third party solver may need to be installed, in particular:

* our **CP based formulations** (with or without the use of bagging to train the target random forest) use the `OR-Tools` CP-SAT solver. Setup instructions are available on:
    * https://developers.google.com/optimization/install/python

* our **MILP-based formulation** (note that bagging is not supported) uses the `Gurobi` MILP solver through its Python wrapper. Note that free academic licenses are available. Setup instructions are available on:
    * https://www.gurobi.com/academia/academic-program-and-licenses/
    * https://www.gurobi.com/downloads/end-user-license-agreement-academic/

Other dependencies may be required: 

* Our implementation of the attack evaluation (function `average_error` within the `utils.py` file) requires the use of the `scipy` python module to compute the minimum cost matching between reconstructed and actual training examples. Setup instructions are available on:
    * https://scipy.org/install/

* The target models we consider are Random Forests learnt using the `scikit-learn` Python library. Setup instructions are available on:
    * https://scikit-learn.org/stable/install.html

* Our scripts commonly use popular libraries such as `numpy`, `pandas`, and `matplotlib`. Setup instructions are available on:
    * https://numpy.org/install/
    * https://pandas.pydata.org/docs/getting_started/install.html
    * https://matplotlib.org/stable/users/installing/index.html

## Getting started

A minimal working example using DRAFT to reconstruct a given forest's training set is presented below.

```python

# Load packages
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from utils import * 

# Load data and subsample a toy training dataset with 10 examples
df = pd.read_csv("data/compas.csv")
df = df.sample(n=50, random_state = 0, ignore_index= True)
X_train, X_test, y_train, y_test = data_splitting(df, "recidivate-within-two-years:1", test_size=40, seed=42)

# Train a Random Forest (without bootstrap sampling)
clf = RandomForestClassifier(bootstrap = False, random_state = 0)
clf = clf.fit(X_train, y_train)

# Reconstruct the Random Forest's training set
from DRAFT import DRAFT

extractor = DRAFT(clf)
dict_res = extractor.fit(bagging=False, method="cp-sat", timeout=60, verbosity=False, n_jobs=-1, seed=42) 

# Retrieve solving time and reconstructed data
duration = dict_res['duration']
x_sol = dict_res['reconstructed_data']

# Evaluate and display the reconstruction rate
e_mean, list_matching = average_error(x_sol,X_train.to_numpy())

print("Complete solving duration :", duration)
print("Reconstruction Error: ", e_mean)

```

Expected output:

``` bash

Complete solving duration : 0.348616361618042
Reconstruction Error:  0.0

```

## Files Description

Hereafter is a description of the different files:

* `DRAFT.py` contains the implementation of the main module (DRAFT attack).

* `minimal_example.py` provides a minimal example use of our training set reconstruction tools.

* `minimal_example_non_binary.py` provides a minimal example use of our training set reconstruction tools, handling all different types of attributes ((one-hot encoded) categorical ones, binary ones, ordinal ones and numerical ones).

* The `data` folder contains the three binary datasets considered in our experiments and `datasets_infos.py` provides information regarding these datasets (in particular, the list of binary attributes one hot encoding the same original feature). It also contains an additional dataset with ordinal and numerical attributes to illustrate the applicability of our approach on these different types of attributes.

* `utils.py` contains several helper functions used in our experiments, such as `average_error` which computes the average reconstruction error between the proposed reconstruction `x_sol` and the actual training set `x_train_list`. Both must have the same shape. As described in our paper, we first perform a minimum cost matching to determine which reconstructed example corresponds to which actual example. We then compute the average error over all attributes of all (matched) examples and return it.

### To launch our experiments

Main experiments file is `data_extraction_from_rf_experiments.py`. It can be run with:

 `python3.10 data_extraction_from_rf_experiments.py --expe_id=xx` 
 
 where `xx` indicates which combination of parameters should be used (namely, dataset, random seed, whether to use bagging or not, maximum depth of the trees and number of trees within the forest) within the precomputed list of all combinations of parameters. The size of the list (number of *parameters combinations*) depends on the used method and is indicated hereafter.

Key variable that must be set in `data_extraction_from_rf_experiments.py` is `method`: 

* `cp-sat` to use the CP formulations with or without bagging (solved using OR-Tools) *(2160 parameters combinations)*
* `milp` to use the MILP formulation without bagging (solved using Gurobi) (Appendix A) *(1080 parameters combinations)*
* `bench` corresponds to the additional experiments on the impact of bagging on data privacy (Appendix B) *(1080 parameters combinations)*
* `bench_partial_{attributes, examples}` corresponds to the complementary experiments on partial reconstruction with knowledge of part of the training set attributes *(185 parameters combinations)* or examples *(210 parameters combinations)*

For a given method, all experiments can be launched on a computing platform using Slurm by:

* setting `ccanada_expes` to `True` in `local_config.py`
* setting the number of *parameters combinations* (NB) (aforementionned) corresponding to the chosen method in the batch file `main_expes_batch.sh` as follows: `#SBATCH --array=0-NB`

### To reproduce the figures within the paper

Results are stored within the `results_{cp-sat, milp, bench, partial_examples, partial_attributes}` folder for our different experiments.

* To plot the example trees presented in the paper: 
`python3.10 trees_figures_paper.py` (note that four trees will be computed and plotted, i.e., one not using exactly one of the four attributes - in the paper we only use two of them)
    * => Generates within the `figures` folder files `tree_toy_example_no_{1,2,3,4}.pdf`

* To plot the reconstruction accuracy results presented in the paper (Section 7.2 and Appendices A and B):
`python3.10 plot_results_v2.py`
    * => Generates within the `figures` folder files `{adult, compas, default_credit}_{cp-sat, milp}_bagging=False_average_acc.pdf` 
    and files `{adult, compas, default_credit}_{bench, cp-sat}_bagging=True_average_acc.pdf`

* To plot the results of the partial reconstruction experiments (Appendix D):
`python3.10 plot_results_partial_reconstr.py`
    * => Generates within the `figures` folder files `{adult, compas, default_credit}_bench_partial_{attributes, examples}_bagging=True_average_acc.pdf`

* To generate the tables providing the average runtimes and number of runs for which the solver did not find a feasible solution:
    * `python3.10 table_runtimes_fails_per_depth.py` for the runs using no depth constraint (sklearn default value) (and different number of trees values) (like Table 3)
        * => Generates within the `tables` folder files `time_table_None_depth_{adult, compas, default_credit}_cp-sat_Bagging_{True, False}.csv`
    * `python3.10 table_runtimes_fails_per_ntrees.py` for the runs using 100 trees (sklearn default value) (and different maximum depth values) (like Table 2)
        * => Generates within the `tables` folder files `time_table_100_trees_{adult, compas, default_credit}_cp-sat_Bagging_{True, False}.csv`

* To generate the tables providing the average, std, min and max runtimes (Table 4): `python3.10 table_comp_milp_cp.py`
    * => Generates within the `tables` folder files `comp_table_time_table_compas_{cp-sat, milp}.pdf`

* To plot the performances (train and test) of the target models (i.e., the random forests):
`python3.10 plot_results_target_models_perfs.py`
    * => Generates within the `figures` folder files `{adult, compas, default_credit}_cp-sat_bagging={True, False}_target_models_perfs_{train, test}.pdf`

### On the type of attributes that can be reconstructed

Our `cp-sat` models (with or without bagging) (called whenever `method='cp-sat'` when calling the `.fit` method) support all types of attributes, but the attributes' types have to be provided to the attack, along with their domains (see the next section regarding the attack parameters). An example reconstruction including both (one-hot encoded) categorical variables, binary variables, ordinal variables and numerical ones is provided within the `minimal_example_non_binary.py` file, using the non-binarized Default of Credit Card Client dataset.

Categorical attributes should be one-hot encoded (they could also be treated as ordinal ones but since there is by definition no order relationship between their different values/categories the resulting Random Forest performance could be suboptimal). Ordinal and numerical attributes are directly handled. In a nutshell, ordinal features are directly modelled as integer variables within the CP solver. Numerical ones are discretized to determine within which interval (i.e., between which split values) they lie on, and after running the solver their particular value is fixed to the middle of the associated interval. More details can be found in the Appendix D of our ICML24 paper.

## Attack Parameters

Our proposed Dataset Reconstruction Attacks against Tree Ensembles are implemented within the `DRAFT` module, contained in the `DRAFT.py` Python file. The different parameters and methods of this module are detailed hereafter:

## Main methods (base use)

* Constructor: **\__init__(self, random_forest, one_hot_encoded_groups=[], ordinal_attributes=[], numerical_attributes=[])**
    * `random_forest`: *instance of sklearn.ensemble.RandomForestClassifier.* The (target) random forest whose structure will be leveraged to reconstruct its training set.

    * `one_hot_encoded_groups`: *list, optional (default []).* List of lists, where each sub-list contains the IDs of a group of attributes corresponding to a one-hot encoding of the same original feature. Not mandatory but if provided, can improve the performed reconstruction and may speed up the process.

    * `ordinal_attributes`: *list, optional (default []).* List of lists, where each sub-list corresponds to an ordinal attribute, and has the form: 
    `[attribute_id, attribute_lower_bound, attribute_upper_bound]`

    * `numerical_attributes`: *list, optional (default []).* List of lists, where each sub-list corresponds to a numerical attribute, and has the form: 
    `[attribute_id, attribute_lower_bound, attribute_upper_bound]`

* **fit(self, bagging=False, method='cp-sat', timeout=60, verbosity=True, n_jobs=-1, seed=0)** <br> Reconstructs a dataset compatible with the knowledge provided by `random_forest`, using the provided parameters. In other terms, fits the data to the given model.

    * `bagging`: *bool, optional (default False).* Indicates whether bootstrap sampling was used to train the base learners. The reconstruction model will be constructed accordingly.

    * `method`: *str in {'cp-sat', 'milp'}, optional (default 'cp-sat').* The type of formulation that will be used to perform the reconstruction. Note that `cp-sat` requires the OR-Tools Python library and `milp` the GurobiPy one (see the **Installation** section of this README).

    * `timeout`: *int, optional (default 60).* Maximum CPU time (in seconds) to be used by the search. If the solver is not able to return a solution within the given time frame, it will be indicated in the returned dictionary.

    * `verbosity`: *bool, optional (default True).* Whether to print information about the search progress or not.

    * `n_jobs`: *int in {-1, positives}, optional (default -1).* Maximum number of threads to be used by the solver to parallelize search. If -1, use all available threads.

    * `seed`: *int, optional (default 0).* Random number generator seed used to fix the behaviour of the solvers.

    * **=> returns:** dictionary containing:
        * `max_max_depth`: maximum depth found when parsing the trees within the forest. 
        * `status`: the solve status returned by the solver. 
            * When method=`cp-sat` (OR-Tools solver), it can be 'UNKNOWN', 'MODEL_INVALID', 'FEASIBLE', 'INFEASIBLE', or 'OPTIMAL'.
            * When method=`milp` (Gurobi solver), it can be "LOADED", "OPTIMAL", "INFEASIBLE", "INF_OR_UNBD", "UNBOUNDED", "CUTOFF", "ITERATION_LIMIT", "NODE_LIMIT", "TIME_LIMIT", "SOLUTION_LIMIT", "INTERRUPTED", "NUMERIC", "SUBOPTIMAL", "INPROGRESS", "USER_OBJ_LIMIT", or "WORK_LIMIT".
        * `duration`: duration of the solve (in seconds).
        * `reconstructed_data`: array of shape = [n_samples, n_attributes] encoding the reconstructed dataset.


## Other methods

### Methods called by .fit() (regular users should not call them directly and use .fit() instead)

* **perform_reconstruction_v1_CP_SAT(self, n_threads=0, time_out=60, verbosity=1, seed=0)** <br>  Constructs and solves the CP based dataset reconstruction model (without the use of bagging to train the target random forest, as described in the last paragraph of Section 6 of our paper) using the OR-Tools CP-SAT solver (see the **Installation** section of this README).

    * `n_threads`: *int >= 0, optional (default 0).* Maximum number of threads to be used by the solver to parallelize search. If 0, use all available threads.

    * `time_out`: *int, optional (default 60).* Maximum CPU time (in seconds) to be used by the search. If the solver is not able to return a solution within the given time frame, it will be indicated in the returned dictionary.

    * `verbosity`: *int, optional (default 1).* Whether to print information about the search progress (1) or not (0).

    * `seed`: *int, optional (default 0).* Random number generator seed used to fix the behaviour of the solver.

    * **=> returns:** dictionary containing:
        * `max_max_depth`: maximum depth found when parsing the trees within the forest. 
        * `status`: the solve status returned by the solver. It can be 'UNKNOWN', 'MODEL_INVALID', 'FEASIBLE', 'INFEASIBLE', or 'OPTIMAL'.
        * `duration`: duration of the solve (in seconds).
        * `reconstructed_data`: array of shape = [n_samples, n_attributes] encoding the reconstructed dataset.

* **perform_reconstruction_v1_MILP(self, n_threads=0, time_out=60, verbosity=1, seed=0)** <br>  Constructs and solves the MILP based dataset reconstruction model (without the use of bagging to train the target random forest, as described in the Appendix A of our paper) using the Gurobi MILP solver through its Python wrapper (see the **Installation** section of this README).

    * `n_threads`: *int in {-1, positives}, optional (default -1).* Maximum number of threads to be used by the solver to parallelize search. If -1, use all available threads.

    * `time_out`: *int, optional (default 60).* Maximum CPU time (in seconds) to be used by the search. If the solver is not able to return a solution within the given time frame, it will be indicated in the returned dictionary.

    * `verbosity`: *int, optional (default 1).* Whether to print information about the search progress (1) or not (0).

    * `seed`: *int, optional (default 0).* Random number generator seed used to fix the behaviour of the solver.

    * **=> returns:** dictionary containing:
        * `max_max_depth`: maximum depth found when parsing the trees within the forest. 
        * `status`: the solve status returned by the solver. It can be "LOADED", "OPTIMAL", "INFEASIBLE", "INF_OR_UNBD", "UNBOUNDED", "CUTOFF", "ITERATION_LIMIT", "NODE_LIMIT", "TIME_LIMIT", "SOLUTION_LIMIT", "INTERRUPTED", "NUMERIC", "SUBOPTIMAL", "INPROGRESS", "USER_OBJ_LIMIT", or "WORK_LIMIT".
            -> 'duration': duration
        * `duration`: duration of the solve (in seconds).
        * `reconstructed_data`: array of shape = [n_samples, n_attributes] encoding the reconstructed dataset.

* **perform_reconstruction_v2_CP_SAT(self, n_threads=0, time_out=60, verbosity=1, seed=0)** <br>  Constructs and solves the CP based dataset reconstruction model (with the use of bagging to train the target random forest, as described in Section 6 of our paper) using the OR-Tools CP-SAT solver (see the **Installation** section of this README).

    * `n_threads`: *int >= 0, optional (default 0).* Maximum number of threads to be used by the solver to parallelize search. If 0, use all available threads.

    * `time_out`: *int, optional (default 60).* Maximum CPU time (in seconds) to be used by the search. If the solver is not able to return a solution within the given time frame, it will be indicated in the returned dictionary.

    * `verbosity`: *int, optional (default 1).* Whether to print information about the search progress (1) or not (0).

    * `seed`: *int, optional (default 0).* Random number generator seed used to fix the behaviour of the solver.

    * `use_mleobj`: *int, optional (default 1).* Whether to use the maximum likelihood objective described in our paper (1) or another one minimizing the absolute difference to the cumulative distribution of probability that a sample is used at least b times, for every tree.

    * `useprobctr`: *int, optional (default 0).* Whether to use constraints that are not necessarily valid, but valid with high probability (measured by epsilon specified within that function) (1) or not (0).

    * **=> returns:** dictionary containing:
        * `max_max_depth`: maximum depth found when parsing the trees within the forest. 
        * `status`: the solve status returned by the solver. It can be 'UNKNOWN', 'MODEL_INVALID', 'FEASIBLE', 'INFEASIBLE', or 'OPTIMAL'.
        * `duration`: duration of the solve (in seconds).
        * `reconstructed_data`: array of shape = [n_samples, n_attributes] encoding the reconstructed dataset.

### Methods implementing the complementary experiments provided in our paper (Appendices B and D)

* **perform_benchmark_partial_examples(self, n_threads=0, time_out=60, verbosity=1, seed=0, X_known = [], y_known=[])** <br>  Runs the complementary experiments on reconstruction with knowledge of part of the training set examples, mentionned in the Appendix D of our paper. The model builds upon the CP based dataset reconstruction model (with the use of bagging to train the target random forest) using the OR-Tools CP-SAT solver, but pre-fixes a number of (supposedly known) training set examples (rows).

    * `n_threads`: *int >= 0, optional (default 0).* Maximum number of threads to be used by the solver to parallelize search. If 0, use all available threads.

    * `time_out`: *int, optional (default 60).* Maximum CPU time (in seconds) to be used by the search. If the solver is not able to return a solution within the given time frame, it will be indicated in the returned dictionary.

    * `verbosity`: *int, optional (default 1).* Whether to print information about the search progress (1) or not (0).

    * `seed`: *int, optional (default 0).* Random number generator seed used to fix the behaviour of the solver.

    * `X_known`: *array-like, shape = [n_known, n_features] (default []).* The set of known examples with `n_known` <= `N` (there can not be more known than actual examples) and `n_features` == `M` (for the known examples we know all of their features).

    * `y_known`: *array-like, shape = [n_known]  (default []).* The labels of the known examples with `n_known` <= `N` (there can not be more known than actual examples)


    * **=> returns:** dictionary containing:
        * `max_max_depth`: maximum depth found when parsing the trees within the forest. 
        * `status`: the solve status returned by the solver. It can be 'UNKNOWN', 'MODEL_INVALID', 'FEASIBLE', 'INFEASIBLE', or 'OPTIMAL'.
        * `duration`: duration of the solve (in seconds).
        * `reconstructed_data`: array of shape = [n_samples, n_attributes] encoding the reconstructed dataset.

* **perform_benchmark_partial_attributes(self, n_threads=0, time_out=60, verbosity=1, seed=0, X_known = [], known_attributes=[])** <br>  Runs the complementary experiments on reconstruction with knowledge of part of the training set attributes, described in the Appendix D of our paper. The model builds upon the CP based dataset reconstruction model (with the use of bagging to train the target random forest) using the OR-Tools CP-SAT solver, but pre-fixes a number of (supposedly known) training set attributes (columns).

    * `n_threads`: *int >= 0, optional (default 0).* Maximum number of threads to be used by the solver to parallelize search. If 0, use all available threads.

    * `time_out`: *int, optional (default 60).* Maximum CPU time (in seconds) to be used by the search. If the solver is not able to return a solution within the given time frame, it will be indicated in the returned dictionary.

    * `verbosity`: *int, optional (default 1).* Whether to print information about the search progress (1) or not (0).

    * `seed`: *int, optional (default 0).* Random number generator seed used to fix the behaviour of the solver.

    * `X_known`: *array-like, shape = [n_known, n_features] (default []).* The entire training set with `n_known = N` and `n_features = M` (note that in the current implementation we provide the entire training set although only the columns specified by `known_attributes` are known).

    * `known_attributes`: *array-like, shape = [n_features'] (default []).* The features (id) that are known (the associated entire colums provided through X_known will be fixed) with n_features' <= M (there can not be more known than actual features).


    * **=> returns:** dictionary containing:
        * `max_max_depth`: maximum depth found when parsing the trees within the forest. 
        * `status`: the solve status returned by the solver. It can be 'UNKNOWN', 'MODEL_INVALID', 'FEASIBLE', 'INFEASIBLE', or 'OPTIMAL'.
        * `duration`: duration of the solve (in seconds).
        * `reconstructed_data`: array of shape = [n_samples, n_attributes] encoding the reconstructed dataset.

* **perform_reconstruction_benchmark(self, x_train, y_train, n_threads=0, time_out=60, verbosity=1, seed=0)** <br> Runs the complementary experiments on the impact of bagging on data protection, mentionned in the Appendix B of our paper. The model builds upon the CP based dataset reconstruction model (with the use of bagging to train the target random forest) using the OR-Tools CP-SAT solver, but first pre-computes the optimal assignments of the examples' occurences within each tree's training set (using the actual forest's training set) before computing how bad the reconstruction could be at worst given these correct occurences assignements.

    * `x_train`: *array-like, shape = [n_known, n_features] (default []).* The actual training set of the forest (used only to compute the optimal #occurences of the examples within the trees' training sets) with `n_known = N` and `n_features = M`

    * `y_train`: *array-like, shape = [n_known] (default []).* The labels of the training set examples with `n_known = N`.

    * `n_threads`: *int >= 0, optional (default 0).* Maximum number of threads to be used by the solver to parallelize search. If 0, use all available threads.

    * `time_out`: *int, optional (default 60).* Maximum CPU time (in seconds) to be used by the search. If the solver is not able to return a solution within the given time frame, it will be indicated in the returned dictionary.

    * `verbosity`: *int, optional (default 1).* Whether to print information about the search progress (1) or not (0).

    * `seed`: *int, optional (default 0).* Random number generator seed used to fix the behaviour of the solver.

    * **=> returns:** dictionary containing:
        * `max_max_depth`: maximum depth found when parsing the trees within the forest. 
        * `status`: the solve status returned by the solver. It can be 'UNKNOWN', 'MODEL_INVALID', 'FEASIBLE', 'INFEASIBLE', or 'OPTIMAL'.
        * `duration`: duration of the solve (in seconds).
        * `reconstructed_data`: array of shape = [n_samples, n_attributes] encoding the reconstructed (worst-case reconstruction) dataset.

