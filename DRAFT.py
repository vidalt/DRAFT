from scipy.stats import binom
import math

class DRAFT:

    # HELPER FUNCTIONS
    # Functions to calculate probabilities for the bagging model
    def _proba(self, k, N):
        return binom.pmf(k, N, 1 / N)

    def _proba_inf(self, k, N):
        calc = 0
        for i in range(k):
            calc += self._proba(i, N)
        return calc

    def _compute_max_b_val(self, N, maxbvalmaxvalue, confidence):
        maxbval = 0
        cumulated_probas = 0
        for occ in range(0,maxbvalmaxvalue):
            #print("P(#occs >= %d) = %.10f" %(occ, 1-cumulated_probas))
            if (1-cumulated_probas) < confidence:
                maxbval = occ
                break
            cumulated_probas += self._proba(occ, N)
        #print("Retained value for maxbval = %d as P(#occs >= %d) = %.10f" %(maxbval, maxbval, self._proba_inf(maxbval, N)))
        return maxbval

    def _parse_forest(self, clf, verbosity=False):
        """
        Parses a given Random Forest learnt using the scikit-learn library and returns the different
        values needed to build our reconstruction models.

        Arguments
        ---------
        clf: instance of sklearn.ensemble.RandomForestClassifier
                       The random forest whose structure will be leveraged to reconstruct its training set

        verbosity: bool, optional (default True)
                        whether to print information about the search progress or not

        Returns
        -------
        T, M, N, C, Z, max_max_depth, trees_branches where:

        T: list of instances of sklearn.tree.DecisionTreeClassifier
                        The different trees within the provided forest

        M: Number of attributes within the forest's training set

        N: Number of examples within the forest's training set

        C: Number of different classes within the forest's training set

        Z: List of the different classes within the forest's training set (according to the first tree) -> only used for the models without bagging

        max_max_depth: Maximum depth reached by any tree within the forest

        trees_branches: list of list
                        for each tree, stores for each branch the list of associated splits along with the corresponding leaf's cardinalities

        maxcards: Maximum per-class cardinalities among all trees -> only used for the models with bagging (without bagging, all these cardinalities should be equal accross all trees)
        """
        import numpy as np
        import sklearn
        from copy import deepcopy

        ## Parse the forest
        # Trees of the forest
        T = clf.estimators_

        # Nombre de features du jeu de données étudiées
        M = T[0].n_features_in_

        # Nombre de classes vues au total
        # Note that np.unique does not work if classes_ are not of the same dimension, which can happen
        unique_classes = set()
        for tree in T:
            for c in tree.classes_:
                unique_classes.add(c)
        C = len(unique_classes)
        unique_classes = sorted(list(unique_classes))

        # Will parse all the branches to determine split levels for numerical features
        # and convert their value to integers modelling intervals between them
        num_indices = [f[0] for f in self.numerical_attrs]
        for i in range(len(self.numerical_attrs)):
            self.numerical_attrs[i] = self.numerical_attrs[i] + [set()] + [[]] # adds lists of split values and intervals

        def retrieve_branches(number_nodes, children_left_list, children_right_list, nodes_features_list, nodes_value_list, nodes_thresholds ):
            """Retrieve decision tree branches"""

            # Calculate if a node is a leaf
            is_leaves_list = [(False if cl != cr else True) for cl, cr in zip(children_left_list, children_right_list)]

            # Store the branches paths
            paths = []

            for i in range(number_nodes):
                if is_leaves_list[i]:
                    # Search leaf node in previous paths
                    end_node = [path[-1] for path in paths]

                    # If it is a leave node yield the path
                    if i in end_node:
                        output = paths.pop(np.argwhere(i == np.array(end_node))[0][0])
                        output = output[:-1]
                        yield (output, list(nodes_value_list[i][0]))

                else:
                    # Origin and end nodes
                    origin, end_l, end_r = i, children_left_list[i], children_right_list[i]

                    # Iterate over previous paths to add nodes
                    for index, path in enumerate(paths):
                        if origin == path[-1]:
                            path[-1] = [-nodes_features_list[origin], nodes_thresholds[origin]]
                            paths[index] = path + [end_l]
                            path[-1] = [nodes_features_list[origin], nodes_thresholds[origin]]
                            paths.append(path + [end_r])

                    # Initialize path in first iteration
                    if i == 0:
                        paths.append([[-nodes_features_list[i], nodes_thresholds[i]], children_left[i] ])
                        paths.append([[nodes_features_list[i], nodes_thresholds[i]], children_right[i] ])

        max_max_depth = 0

        trees_branches = []
        first_tree = True
        maxcards = []

        # iterate over the trees of the forest
        for tree in T:
            t = tree.tree_

            # Depending on sklearn version different parsing must be done here
            sklearn_version = str(sklearn.__version__).split(".")
            if (int(sklearn_version[0]) <= 1 and int(sklearn_version[1]) <= 3) or self.is_DP_RF:
                nodes_value = t.value # For all nodes in the tree, list of their value (support for both classes)
            else:
                total_examples = t.weighted_n_node_samples
                nodes_value = deepcopy(t.value) # For all nodes in the tree, list of their value (relative support for both classes)
                for i in range(len(nodes_value)):     # For each node
                    #print(total_examples[i], nodes_value[i])
                    for j in range(len(nodes_value[i][0])):
                        nodes_value[i][0][j] = np.round(nodes_value[i][0][j] * total_examples[i], decimals=0)
                    #print(total_examples[i], nodes_value[i], '\n')
                    assert(sum(nodes_value[i][0]) == total_examples[i]) # just make sure there were no rounding error

            cards = list(nodes_value[0][0]) # at the root

            if first_tree: # Init constants (only do it once)

                # Nombre d'individus (calculé ici en sommant les cardinalités par classes à la racine de l'arbre 0)
                N = 0
                for c in range(C):
                    N += int(cards[c])

                # Classes des exemples
                Z = np.zeros((N, C), dtype=int)
                deb = 0
                for c in range(C): # for each class
                    for i in range(deb, deb + int(cards[c])):
                        Z[i, c] = 1
                    deb += int(cards[c])
                maxcards = [ 0 for c in range(C) ]

                first_tree = False

            for c in range(C):
                if cards[c] > maxcards[c]:
                    maxcards[c] = cards[c]

            max_depth = t.max_depth

            if max_depth > max_max_depth:
                max_max_depth = max_depth

            n_nodes = t.node_count
            children_left = t.children_left # For all nodes in the tree, list of their left children (or -1 for leaves)
            children_right = t.children_right # For all nodes in the tree, list of their right children (or -1 for leaves)
            nodes_features = deepcopy(t.feature) # For all nodes in the tree, list of their used feature (or -2 for leaves)
            # Note that we need to make deep copies of the lists for which we modify elements

            # Not sure we need deepcopy of threshold, but doing it to be safe
            nodes_thresholds = deepcopy( t.threshold )
            nodes_features += 1

            if not(tree.n_classes_ == C): # If some classes were not observed, fill the other ones' counts with zeros
                #print("unique_classes = ", unique_classes)
                #print("tree.classes_ = ", tree.classes_)
                #print(nodes_value[2])
                new_nodes_values = []
                for i in range(len(nodes_value)): # for each leaf/path
                    new_value = np.zeros(C)
                    for cid, a_class in enumerate(tree.classes_): # retrieve each class counts (for the classes appearing in the tree)
                        new_value[unique_classes.index(a_class)] = nodes_value[i][0][cid]
                    new_nodes_values.append([new_value])
                nodes_value = new_nodes_values
                #print(nodes_value[2])

            if n_nodes == 1: # Special case if there is only the root
                #if tree.n_classes_ == C:
                all_branches = [[[], list(nodes_value[0][0])]]
                #else: # The tree only saw of a subset of all existing classes, need to correct its cardinalities
                #    all_branches = [[[], np.zeros(C)]]
                #    for cid, a_class in enumerate(tree.classes_):
                #        all_branches[0][1][unique_classes.index(a_class)] = nodes_value[0][0][cid]
            else:
                all_branches = list(retrieve_branches(n_nodes, children_left, children_right, nodes_features, nodes_value, nodes_thresholds ))

            trees_branches.append(all_branches)

            # Retrieve all split values for numerical features
            for a_feat, threshold_val in zip(nodes_features, nodes_thresholds):
                feature_val = a_feat-1
                if feature_val in num_indices: # numerical feature
                    idfeature = num_indices.index(feature_val)
                    #if not(threshold_val in self.numerical_attrs[idfeature][3]): # useless now since we use sets (more efficient)
                    self.numerical_attrs[idfeature][3].add(threshold_val)

        # Sort the split levels ($\mathcal{A}_i$) for numerical features
        for i in range(len(self.numerical_attrs)):
            self.numerical_attrs[i][3] = list(self.numerical_attrs[i][3])
            self.numerical_attrs[i][3].sort()

        # Concatenate the split levels with bounds ($\mathcal{I}_i$) to get all the possible intervals for numerical features
        for i in range(len(self.numerical_attrs)):
            self.numerical_attrs[i][4] = [self.numerical_attrs[i][1]] + self.numerical_attrs[i][3] + [self.numerical_attrs[i][2]]

        if verbosity:
            print("RF parsing done!")

        return T, M, N, C, Z, max_max_depth, trees_branches, maxcards

    # MAIN FUNCTIONS
    def __init__(self, random_forest, one_hot_encoded_groups=None, ordinal_attributes=None, numerical_attributes=None):
        """
        Constructor.

        Attributes
        ---------
        random_forest: instance of sklearn.ensemble.RandomForestClassifier
                       The random forest whose structure will be leveraged to reconstruct its training set

        one_hot_encoded_groups: list, optional
                        list of lists, where each sub-list contains the IDs of a group of attributes corresponding to a one-hot encoding of the same original feature.
                        not mandatory but if provided, can improve the performed reconstruction and may speed up the process

        ordinal_attributes: list, optional
                        list of lists, where each sub-list contains the ID of an ordinal attribute
                        (i.e., an attribute taking values within a discrete set of ordered values),
                        followed by the lower and upper bounds of its domain.

        numerical_attributes: list, optional
                list of lists, where each sub-list contains the ID of a numerical attribute
                (i.e., an attribute taking values in a continuous domain),
                followed by the lower and upper bounds of its domain.

        For formatting of the features' domains-related attributes
            (one_hot_encoded_groups, ordinal_attributes, and numerical_attributes),
            see an example at https://github.com/vidalt/DRAFT/blob/main/datasets_infos.py

        """
        from sklearn.ensemble import RandomForestClassifier
        from copy import deepcopy
        if not isinstance(random_forest, RandomForestClassifier):
            try:
                from DP_RF import DP_RF
                if not isinstance(random_forest, DP_RF):
                    raise TypeError("Expected a RandomForestClassifier or DP_RF but provided random_forest is of type " + str(type(random_forest)))
                else:
                    self.is_DP_RF = True
            except ImportError:
                raise TypeError("Expected a RandomForestClassifier but provided random_forest is of type " + str(type(random_forest)))
        else:
            self.is_DP_RF = False
        self.clf = random_forest
        self.ohe_groups = deepcopy([] if one_hot_encoded_groups is None else one_hot_encoded_groups)
        self.ordinal_attrs = deepcopy([] if ordinal_attributes is None else ordinal_attributes)
        self.numerical_attrs = deepcopy([] if numerical_attributes is None else numerical_attributes) # Since we will modify it to include split values

    def fit(self, *, timeout=60, verbosity=False, n_threads=-1, seed=0, bagging=None, method='cp-sat'):
        """
        Reconstructs a dataset compatible with the knowledge provided by random_forest.
        In other terms, fits the data to the given model.

        Arguments
        ---------
        bagging: bool, optional (default None, meaning that this parameter will be retrieved within the provided RF's attributes)
                        whether bootstrap sampling was used to train the base learners
                        the reconstruction model will be constructed accordingly

        method: str in {'cp-sat', 'milp'}, optional (default 'cp-sat')
                        the type of formulation that will be used to perform the reconstruction
                        Note that `cp-sat` requires the OR-Tools Python library
                        and `milp` the GurobiPy one (see the Installation section of our README).

        timeout: int, optional (default 60)
                        maximum solver search time (in seconds; excludes model construction) to be used by the search
                        if the solver is not able to return a solution within the given time frame, it will be indicated in the returned dictionary

        verbosity: bool, optional (default False)
                        whether to print information about the search progress or not

        n_threads: int in {-1, positives}, optional (default -1)
                        maximum number of threads to be used by the solver to parallelize search
                        if -1, use all available threads

        seed: int, optional (default 0)
                       random number generator seed
                       used to fix the behaviour of the solvers

        Returns
        -------
        output: dictionary containing:
            -> 'max_max_depth': maximum depth found when parsing the trees within the forest.
            -> 'status': the solve status returned by the solver.
                - When method=`cp-sat` (OR-Tools solver), it can be 'UNKNOWN', 'MODEL_INVALID', 'FEASIBLE', 'INFEASIBLE', or 'OPTIMAL'.
                - When method=`milp` (Gurobi solver), it can be "LOADED", "OPTIMAL", "INFEASIBLE", "INF_OR_UNBD", "UNBOUNDED", "CUTOFF", "ITERATION_LIMIT", "NODE_LIMIT", "TIME_LIMIT", "SOLUTION_LIMIT", "INTERRUPTED", "NUMERIC", "SUBOPTIMAL", "INPROGRESS", "USER_OBJ_LIMIT", or "WORK_LIMIT".
            -> 'duration': duration
            -> 'reconstructed_data': array of shape = [n_samples, n_attributes] encoding the reconstructed dataset.
            Note that if the status is not OPTIMAL or FEASIBLE the reconstruction should not be used.
        """
        from _validation import validate_solver_options
        validate_solver_options(timeout, n_threads, seed)
        self.numerical_attrs = [attr[:3] for attr in self.numerical_attrs]
        self.result_dict = {}
        if not method in ['cp-sat', 'milp']:
            raise ValueError("Supported methods are either 'cp-sat' or 'milp', got: " + method)

        try: # try to retrieve parameters from the sklearn object
            if bagging is None:
                if self.is_DP_RF:
                    bagging = False
                else:
                    bagging = self.clf.bootstrap
        except:
            import warnings
            if bagging is None:
                bagging = True
            warnings.warn("Couln't retrieve the parameter bootstrap from the given RF. Defaulting to True, but might result in suboptimal results if actual value is False.")

        if method == 'cp-sat':

            if n_threads == -1:
                n_threads = 0 # value for OR-Tools
            if not bagging:
                self._reconstruct_without_bagging(n_threads=n_threads, time_out=timeout, verbosity=verbosity, seed=seed)
            else:
                self._reconstruct_with_bagging(n_threads=n_threads, time_out=timeout,
                    verbosity=verbosity, seed=seed)

        elif not bagging and method == 'milp':
            if len(self.ordinal_attrs) > 0 or len(self.numerical_attrs) > 0:
                raise AttributeError("Currently numerical and ordinal attributes are not supported with MILP formulations.")
            if n_threads == -1:
                n_threads = 0 # value for Gurobi
            self._reconstruct_milp(n_threads=n_threads, time_out=timeout, verbosity=int(verbosity), seed=seed)
        else:
            raise AttributeError("Currently bagging is not supported with MILP formulations.")

        if hasattr(self, 'result_dict'):
            return self.result_dict
        else:
            raise RuntimeError('Something went wrong and the reconstruction could not be performed. Please report this issue to the developers.')

    def _reconstruct_without_bagging(self, n_threads=0, time_out=60, verbosity=1, seed=0):
        """
        Constructs and solves the CP based dataset reconstruction model (without the use of bagging to train the target random forest) using the OR-Tools CP-SAT solver.

        Arguments
        ---------
        n_threads: int >= 0, optional (default 0)
                        maximum number of threads to be used by the solver to parallelize search
                        if 0, use all available threads

        time_out: int, optional (default 60)
                        maximum solver search time (in seconds; excludes model construction) to be used by the search
                        if the solver is not able to return a solution within the given time frame, it will be indicated in the returned dictionary

        verbosity: int, optional (default 1)
                        whether to print information (1) about the search progress or not (0)

        seed: int, optional (default 0)
                       random number generator seed
                       used to fix the behaviour of the solver

        Returns
        -------
        output: dictionary containing:
            -> 'max_max_depth': maximum depth found when parsing the trees within the forest.
            -> 'status': the solve status returned by the solver. It can be 'UNKNOWN', 'MODEL_INVALID', 'FEASIBLE', 'INFEASIBLE', or 'OPTIMAL'.
            -> 'duration': duration
            -> 'reconstructed_data': array of shape = [n_samples, n_attributes] encoding the reconstructed dataset.
            Note that if the status is not OPTIMAL or FEASIBLE the reconstruction should not be used.
        """
        from ortools.sat.python import cp_model
        import numpy as np # useful
        import time # time measurements

        clf = self.clf
        one_hot_encoded_groups = self.ohe_groups

        start = time.time()


        T, M, N, C, Z, max_max_depth, trees_branches, _ = self._parse_forest(clf, verbosity=verbosity)
        ### Create the CP model

        ## Variables
        model = cp_model.CpModel()

        # Reconstruction variables
        ord_indices = [f[0] for f in self.ordinal_attrs] # Ordinal attributes => Integer variables
        num_indices = [f[0] for f in self.numerical_attrs] # Numerical attributes => Integer variables modelling intervals between two splits

        x_vars = [[None] * M for k in range(N)]
        for i in range(M):
            if i in ord_indices:
                idx = ord_indices.index(i)
                for k in range(N):
                    x_vars[k][i] = model.NewIntVar(self.ordinal_attrs[idx][1], self.ordinal_attrs[idx][2], 'x_%d_%d' % (k, i) )
            elif i in num_indices:
                idx = num_indices.index(i)
                #print("Num attr ", i, " : ", self.numerical_attrs[idx][4])
                for k in range(N):
                    x_vars[k][i] = model.NewIntVar(0, len(self.numerical_attrs[idx][3]), 'x_%d_%d' % (k, i) )
            else:
                for k in range(N):
                    x_vars[k][i] = model.NewBoolVar('x_%d_%d' %(k,i))

        # Contraintes
        # one-hot encoding
        for k in range(N):
            for w in range(len(one_hot_encoded_groups)): # for each group of binary attributes one-hot encoding the same attribute
                model.Add(cp_model.LinearExpr.Sum([x_vars[k][i] for i in one_hot_encoded_groups[w]]) == 1)

        for all_branches_t in trees_branches: # for each tree
            examples_capts = [[] for k in range(N)] # for each example we will ensure it is captured by exactly one branch
            for a_branch_nb, a_branch in enumerate(all_branches_t): # iterate over its branches
                branch_vars_c = [[] for c in range(C)] # for each class, list of examples of this class indicating whether they go to this branch
                for k in range(N):
                    k_label_one_hot = Z[k]
                    for c in range(C):
                        if a_branch[1][c] > 0 and k_label_one_hot[c] == 1: # example is of the correct class and the branch contains examples from it
                            local_capt_var = model.NewBoolVar('branch_capt_%d_%d_%d' %(c,a_branch_nb,k))
                            branch_vars_c[c].append(local_capt_var)
                            examples_capts[k].append(local_capt_var)
                            for a_split in a_branch[0]:
                                feature_val = a_split[0]
                                feature_id = abs(feature_val)-1
                                threshold_val = a_split[1]
                                if feature_id in num_indices:
                                    idx = num_indices.index(feature_id)
                                    threshold_val = self.numerical_attrs[idx][3].index(threshold_val) # replace the actual split value with its index (interval nb)
                                if feature_val > 0:
                                    model.Add( x_vars[k][feature_id] >= int( math.floor( threshold_val ) ) + 1 ).OnlyEnforceIf( local_capt_var )
                                elif feature_val < 0:
                                    model.Add(x_vars[k][feature_id] <= int( math.floor( threshold_val ) ) ).OnlyEnforceIf( local_capt_var)
                                else:
                                    raise ValueError("Feat 0 shouldn't be used here (1-indexed now)")
                for c in range(C):
                    model.Add(cp_model.LinearExpr.Sum(branch_vars_c[c]) ==  int(a_branch[1][c])) # enforces the branch per-class cardinality
            for k in range(N):
                if len(examples_capts[k]) > 0:
                    model.Add(cp_model.LinearExpr.Sum(examples_capts[k]) == 1) # each example captured by exactly one branch

        # Ordre lexicographique au sein des classes
        '''deb = 0
        for c in range(C): # for each class
            for k in range(deb, deb + int(cards[c])-1):
                model.Add(sum(2 ** (i) * x_vars[k][i] for i in range(M)) <= sum(2 ** (i) * x_vars[k+1][i] for i in range(M)))
            deb += int(cards[c])'''

        if verbosity:
            print("Model creation done!")

        # Résolution
        solver = cp_model.CpSolver()

        # Sets a time limit of XX seconds.
        solver.parameters.log_search_progress = verbosity
        solver.parameters.max_time_in_seconds = time_out
        solver.parameters.num_workers = n_threads
        solver.parameters.random_seed = seed

        status = solver.Solve(model)

        end = time.time()
        duration = end - start

        # Récupération statut/valeurs

        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            x_sol = [[solver.Value(x_vars[k][i]) for i in range(M)] for k in range(N)]
            # Retrieve actual numerical attributes' values from their assigned interval ID
            for k in range(N):
                for i in range(len(self.numerical_attrs)):
                    attr_id = self.numerical_attrs[i][0]
                    interval_id = x_sol[k][attr_id]
                    reconstructed_value = (self.numerical_attrs[i][4][interval_id]+self.numerical_attrs[i][4][interval_id+1])/2
                    x_sol[k][attr_id]=reconstructed_value

        else:
            if status == cp_model.INFEASIBLE or status == cp_model.MODEL_INVALID:
                raise RuntimeError('Infeasible model: the reconstruction problem has no solution. Please make sure the provided one-hot encoding constraints are correct. Else, report this issue to the developers.')
            else:
                x_sol = None

        solve_status = {0: 'UNKNOWN',
        1: 'MODEL_INVALID',
        2: 'FEASIBLE',
        3: 'INFEASIBLE',
        4:'OPTIMAL'}[status]

        self.result_dict = {'max_max_depth':max_max_depth, 'status':solve_status, 'duration': duration, 'reconstructed_data':x_sol}

    def _reconstruct_milp(self, n_threads=0, time_out=60, verbosity=1, seed=0):
        """
        Constructs and solves the MILP based dataset reconstruction model (without the use of bagging to train the target random forest) using the Gurobi MILP solver through its Python wrapper.

        Arguments
        ---------
        n_threads: int in {-1, positives}, optional (default -1)
                        maximum number of threads to be used by the solver to parallelize search
                        if -1, use all available threads

        time_out: int, optional (default 60)
                        maximum solver search time (in seconds; excludes model construction) to be used by the search
                        if the solver is not able to return a solution within the given time frame, it will be indicated in the returned dictionary

        verbosity: int, optional (default 1)
                        whether to print information (1) about the search progress or not (0)

        seed: int, optional (default 0)
                       random number generator seed
                       used to fix the behaviour of the solvers

        Returns
        -------
        output: dictionary containing:
            -> 'max_max_depth': maximum depth found when parsing the trees within the forest.
            -> 'status': the solve status returned by the solver. It can be "LOADED", "OPTIMAL", "INFEASIBLE", "INF_OR_UNBD", "UNBOUNDED", "CUTOFF", "ITERATION_LIMIT", "NODE_LIMIT", "TIME_LIMIT", "SOLUTION_LIMIT", "INTERRUPTED", "NUMERIC", "SUBOPTIMAL", "INPROGRESS", "USER_OBJ_LIMIT", or "WORK_LIMIT".
            -> 'duration': duration
            -> 'reconstructed_data': array of shape = [n_samples, n_attributes] encoding the reconstructed dataset.
            Note that if the status is not OPTIMAL or FEASIBLE the reconstruction should not be used.
        """
        import gurobipy as gp # solver
        import numpy as np # useful
        from gurobipy import GRB, quicksum # solver
        import time # time measurements
        import sklearn

        clf = self.clf
        one_hot_encoded_groups = self.ohe_groups

        start = time.time()

        ### Create the MILP model
        m = gp.Model("reconstruction_model")
        m.setParam('LogToConsole', verbosity) # 0 or 1
        m.setParam('TimeLimit', time_out) # in seconds
        m.setParam('Threads', n_threads) # 0 = all threads available
        m.setParam('Seed', seed) # for reproducibility

        ## Parse the forest
        # Trees of the forest
        T = clf.estimators_
        # Depths attained in each tree
        D = []
        # Nodes of each tree
        V = []
        # Internal nodes of each tree
        V_I = []
        # List of the internal nodes of each trees for each  depth
        V_Id = []
        # List of the internal nodes of each trees for each  feature
        V_If = []
        # List of the leafs of each tree
        V_L = []
        # List of Left and Right children of each tree
        L = []
        R = []
        # List for each tree, for each node, le nombre d'éléments de chaque classe qui passe par le noeud
        Nb_c = []
        # Nombre de features du jeu de données étudiées
        M = T[0].n_features_in_
        max_max_depth = 0
        # iterate over the trees of the forest
        from copy import deepcopy
        for tree in T:
            t = tree.tree_
            max_depth = t.max_depth

            if max_depth > max_max_depth:
                max_max_depth = max_depth

            tamp_leaves = []
            tamp_inside = []
            tamp_nd = [[]] * max_depth
            tamp_nf = [[]] * M

            n_nodes = t.node_count
            children_left = t.children_left
            children_right = t.children_right
            feature = t.feature
            threshold = t.threshold

            node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
            is_leaves = np.zeros(shape=n_nodes, dtype=bool)
            stack = [(0, 0)]  # start with the root node id (0) and its depth (0)

            while len(stack) > 0:
                # `pop` ensures each node is only visited once
                node_id, depth = stack.pop()
                node_depth[node_id] = depth

                # If the left and right child of a node is not the same we have a split
                # node
                is_split_node = children_left[node_id] != children_right[node_id]
                # If a split node, append left and right children and depth to `stack`
                # so we can loop through them
                if is_split_node:
                    stack.append((children_left[node_id], depth + 1))
                    stack.append((children_right[node_id], depth + 1))
                else:
                    is_leaves[node_id] = True

            for i in range(n_nodes):
                if is_leaves[i]:
                    tamp_leaves.append(i)
                else:
                    tamp_inside.append(i)
                    tamp_nd[node_depth[i]] = tamp_nd[node_depth[i]] + [i]
                    tamp_nf[feature[i]] = tamp_nf[feature[i]] + [i]

            V_I.append(tamp_inside)
            V_Id.append(tamp_nd)
            V_If.append(tamp_nf)
            V_L.append(tamp_leaves)

            D.append([d for d in range(max_depth)])
            V.append([v for v in range(n_nodes)])
            L.append(t.children_left)
            R.append(t.children_right)
            # Depending on sklearn version different parsing must be done here
            sklearn_version = str(sklearn.__version__).split(".")
            if int(sklearn_version[0]) <= 1 and int(sklearn_version[1]) <= 3:
                nodes_value = t.value # For all nodes in the tree, list of their value (support for both classes)
            else:
                total_examples = t.weighted_n_node_samples
                nodes_value = deepcopy(t.value) # For all nodes in the tree, list of their value (relative support for both classes)
                for i in range(len(nodes_value)):     # For each node
                    #print(total_examples[i], nodes_value[i])
                    for j in range(len(nodes_value[i][0])):
                        nodes_value[i][0][j] = np.round(nodes_value[i][0][j] * total_examples[i], decimals=0)
                    #print(total_examples[i], nodes_value[i], '\n')
                    assert(sum(nodes_value[i][0]) == total_examples[i]) # just make sure there were no rounding error

            Nb_c.append(nodes_value.tolist())

        # Nombre de classes (lisible par exemple ici en regardant la taille du tableau des cardinalités à la racine de l'arbre 0)
        C = len(Nb_c[0][0][0])

        # Nombre d'individus (calculé ici en sommant les cardinalités par classes à la racine de l'arbre 0)
        N = 0
        for c in range(C):
            N += int(Nb_c[0][0][0][c])

        # Nombre d'exemples par classe
        distrib_classes = Nb_c[0][0][0]

        # Classes des exemples
        Z = np.zeros((C, N), dtype=int)
        deb = 0
        for c in range(C): # for each class
            for i in range(deb, deb + int(distrib_classes[c])):
                Z[c, i] = 1
            deb += int(distrib_classes[c])

        # Tableau qui pour chaque classe C nous donne la liste des individus de la classe C
        Z_bis = []
        for c in range(C):
            list_tampon = []
            for k in range(N):
                if Z[c, k]:
                    list_tampon.append(k)
            Z_bis.append(list_tampon)

        if verbosity > 0:
            print("RF parsing done!")

        ## Variables

        # Variables lambda
        indices_lambda = [(t,d,k) for t in range(len(T)) for d in D[t] for k in range(N)]
        lam = m.addVars(indices_lambda, vtype = GRB.BINARY, name = "lambda")

        # Variables de flot y
        indices_y = [(t,v,k) for t in range(len(T)) for v in V[t] for k in range(N)]
        y = m.addVars(indices_y, lb = 0, ub = 1, vtype = GRB.CONTINUOUS, name = "y")

        # Variables x de reconstitution
        x = m.addMVar((N,M), vtype = GRB.BINARY, name = "x")
        #x = m.addMVar((N,M), lb = 0, ub = 1, vtype = GRB.CONTINUOUS, name = "x")

        ## Contraintes

        for t in range(len(T)):
            # Contraintes modélisant le flot des exemples
            m.addConstrs((1 <= y[t, 0, k] for k in range(N)), name='c1')

            #m.addConstrs((y[t, 0, k] <= 1 for k in range(N)), name='c2') # déjà dans l'ub

            m.addConstrs((y[t, v, k] - y[t, L[t][v], k] - y[t, R[t][v], k] == 0 for v in V_I[t] for k in range(N)), name='c3')

            m.addConstrs((quicksum(y[t, L[t][v], k] for v in V_Id[t][d]) <= lam[t, d, k] for d in D[t] for k in range(N)),
                        name='c4')

            m.addConstrs((quicksum(y[t, R[t][v], k] for v in V_Id[t][d]) <= (1 - lam[t, d, k]) for d in D[t] for k in range(N)),
                        name='c5')

            # Contraintes liant ces flots aux valeurs des attributs des exemples
            m.addConstrs((x[k, i] <= 1 - y[t, L[t][v], k] for k in range(N) for i in range(M) for v in V_If[t][i]), name='c6')

            m.addConstrs((y[t, R[t][v], k] <= x[k, i] for k in range(N) for i in range(M) for v in V_If[t][i] ), name='c7')

            # Contraintes liant ces flots à la structure des arbres
            m.addConstrs((Nb_c[t][v][0][c] <= quicksum(y[t, v, k] * Z[c, k] for k in range(N)) for v in V[t] for c in range(C)),
                        name='c8')
            m.addConstrs((quicksum(y[t, v, k] * Z[c, k] for k in range(N)) <= Nb_c[t][v][0][c] for v in V[t] for c in range(C)),
                        name='c9')
            #m.addConstrs((quicksum(y[t, v, k] * Z[c, k] for k in range(N)) == Nb_c[t][v][0][c] for v in V[t] for c in range(C)), name='c8')

        # Ordre lexicographique au sein des classes
        for c in range(C):
            n_bis = len(Z_bis[c])
            m.addConstrs((quicksum(2 ** (i) * x[Z_bis[c][k], i] for i in range(M)) <= quicksum(
                2 ** (i) * x[Z_bis[c][k + 1], i] for i in range(M)) for k in range(n_bis - 1)), name='c10')

        #contraintes particulières au dataset liées au one-hot encoding

        m.addConstrs((quicksum(x[k, i] for i in one_hot_encoded_groups[w]) == 1 for k in range(N) for w in range(len(one_hot_encoded_groups))),
                    name='c11')
        #Fonction objectif Maximiser ou Minimiser le nombre de 1

        #m.setObjective(quicksum(x[k, i] for i in range(M) for k in range(N)), GRB.MINIMIZE) #Modify MINIMIZE in MAXIMIZE if we want to maximize

        if verbosity > 0:
            print("Model creation done!")

        m.optimize()

        end = time.time()

        duration = end - start
        '''
        m.computeIIS()
        m.write("my_iis.ilp")
        '''

        # see https://www.gurobi.com/documentation/9.5/refman/optimization_status_codes.html#sec:StatusCodes
        statuses = {1: "LOADED", 2: "OPTIMAL", 3: "INFEASIBLE", 4: "INF_OR_UNBD", 5: "UNBOUNDED", 6: "CUTOFF", 7: "ITERATION_LIMIT", 8: "NODE_LIMIT", 9: "TIME_LIMIT", 10: "SOLUTION_LIMIT", 11: "INTERRUPTED", 12: "NUMERIC", 13: "SUBOPTIMAL", 14: "INPROGRESS", 15: "USER_OBJ_LIMIT", 16: "WORK_LIMIT"}
        solve_status = statuses[m.status]

        if solve_status == "INFEASIBLE":
            raise RuntimeError('Infeasible model: the reconstruction problem has no solution. Please report this issue to the developers.')

        if verbosity > 0:
            print("Solve status: ", solve_status)

        ##Récupération des solutions
        x_sol = x.getAttr(GRB.Attr.X).tolist() if m.SolCount > 0 else None

        self.result_dict = {'max_max_depth':max_max_depth, 'status':solve_status, 'duration': duration, 'reconstructed_data':x_sol}

    def _reconstruct_with_bagging(self, n_threads=0, time_out=60, verbosity=1, seed=0, use_mleobj=1, useprobctr = 0 ):
        """
        Constructs and solves the CP based dataset reconstruction model (with the use of bagging to train the target random forest) using the OR-Tools CP-SAT solver.

        Arguments
        ---------
        n_threads: int >= 0, optional (default 0)
                        maximum number of threads to be used by the solver to parallelize search
                        if 0, use all available threads

        time_out: int, optional (default 60)
                        maximum solver search time (in seconds; excludes model construction) to be used by the search
                        if the solver is not able to return a solution within the given time frame, it will be indicated in the returned dictionary

        verbosity: int, optional (default 1)
                        whether to print information (1) about the search progress or not (0)

        seed: int, optional (default 0)
                       random number generator seed
                       used to fix the behaviour of the solvers

        (it is not recommended to play with the two arguments below)

        use_mleobj: int, optional (default 1)
                       Whether to use the new maximum likelihood objective (1)
                       or another one minimizing the absolute difference to the cumulative distribution of
                       probability that a sample is used at least b times, for every tree.

        useprobctr: int, optional (default 0)
                       Whether to use constraints that are not necessarily valid, but valid with high probability (measured by epsilon specified within that function) (1) or not (0).
        Returns
        -------
        output: dictionary containing:
            -> 'max_max_depth': maximum depth found when parsing the trees within the forest.
            -> 'status': the solve status returned by the solver. It can be 'UNKNOWN', 'MODEL_INVALID', 'FEASIBLE', 'INFEASIBLE', or 'OPTIMAL'.
            -> 'duration': duration
            -> 'reconstructed_data': array of shape = [n_samples, n_attributes] encoding the reconstructed dataset.
            Note that if the status is not OPTIMAL or FEASIBLE the reconstruction should not be used.
        """
        import ortools
        from ortools.sat.python import cp_model
        import numpy as np  # useful
        import time  # time measurements

        clf = self.clf
        one_hot_encoded_groups = self.ohe_groups

        start = time.time()

        ### Create the CP model

        ## Parse the forest
        _, M, N, C, _, max_max_depth, trees_branches, maxcards = self._parse_forest(clf, verbosity=verbosity)

        ## Try to retrieve #Occurences
        try: # try to retrieve parameters from the sklearn object
            true_occurences = clf.estimators_samples_
        except:
            import warnings
            import sklearn
            true_occurences = None
            sklearn_version = str(sklearn.__version__).split(".")
            if int(sklearn_version[0]) <= 1 and int(sklearn_version[1]) <= 3: # After sklearn >= 1.4.0, #occurences is normally provided
                warnings.warn("Couln't retrieve the parameter estimators_samples_ from the given RF, will try to infer and optimize it. This is normal since your scikit-learn version is <= 1.3.2.")
            else:
                warnings.warn("Couln't retrieve the parameter estimators_samples_ from the given RF, will try to infer and optimize it.")

        # This is the maximum number of times a sample can appear in a tree (note it will go from 0 to maxbval-1)
        # We fix maxbval to ensure that P(actual #occs >= maxbval) < confidence
        maxbvalmaxvalue = 12
        confidence = 1e-5

        maxbval = self._compute_max_b_val(N, maxbvalmaxvalue, confidence)
        # ------------------------------------------------------------------------

        # Defines the probabilities that an item will appear b times
        P = []
        Pexact = [0 for i in range(maxbval)]
        for i in range(maxbval):
            #P.append( 1 - self._proba_inf(i + 1, N) )
            P.append(1 - self._proba_inf(i , N))
        for i in range(maxbval):
            if i < maxbval - 1:
                Pexact[i] = P[i] - P[i+1]
            else:
                Pexact[i] = P[i]

        '''if verbosity:
            print("Probabilities of an item appearing at least b times:")
            print(P)
            print(sum(P))

            print("Probabilities of an item appearing at EXACTLY b times:")
            print(Pexact)
            print(sum(Pexact))'''

        ntrees = len( trees_branches )

        ## Variables
        model = cp_model.CpModel()

        # Reconstruction variables
        # x[k][i] : Variables that represent what is sample k (each of its features i)
        ord_indices = [f[0] for f in self.ordinal_attrs] # Ordinal attributes => Integer variables
        num_indices = [f[0] for f in self.numerical_attrs] # Numerical attributes => Integer variables modelling intervals between two splits

        x_vars = [[None] * M for k in range(N)]
        for i in range(M):
            if i in ord_indices:
                idx = ord_indices.index(i)
                for k in range(N):
                    x_vars[k][i] = model.NewIntVar(self.ordinal_attrs[idx][1], self.ordinal_attrs[idx][2], 'x_%d_%d' % (k, i) )
            elif i in num_indices:
                idx = num_indices.index(i)
                #print("Num attr ", i, " : ", self.numerical_attrs[idx][4])
                for k in range(N):
                    x_vars[k][i] = model.NewIntVar(0, len(self.numerical_attrs[idx][3]), 'x_%d_%d' % (k, i) )
            else:
                for k in range(N):
                    x_vars[k][i] = model.NewBoolVar('x_%d_%d' %(k,i))

        # y_vars[k][c][t][v]: Variables that represent the number of times sample k is used as class c
        #   in leaf/branch v of tree t
        y_vars = [[[[] for t in range(ntrees)] for c in range(C)] for k in range(N)]

        # w_vars[k][t][v]: Variables that represent if sample k is used in leaf v of tree t
        w_vars = [[[] for t in range(ntrees) ] for k in range(N)]

        # z_vars[k][c]: Variables that represent if sample k is assigned class c
        z_vars = [[model.NewBoolVar('z_%d_%d'%(k,c)) for c in range(C) ] for k in range(N) ]


        # eta_vars[k][t]: Variables that count how many times sample k is used in tree t
        eta_vars = [[ model.NewIntVar(0,maxbval, 'eta_%d_%d'%(k,t)) for t in range(ntrees)] for k in range(N) ]

        # q_vars[k][t][b] Variables that represent if sample k appears b times in tree t (needed for objective function)
        q_vars = [[[model.NewBoolVar('q_%d_%d_%d' %( t, k, b )) for b in range(maxbval) ] for t in range(ntrees) ] for k in range(N) ]


        if true_occurences is None: # Could not retrieve #occurences, but can still pre-fix a few classes
            fixedzidx = 0
            for c in range(C):
                mincz = math.floor( maxcards[c] / maxbval)
                for offset in range(mincz):
                    model.Add( z_vars[fixedzidx+offset][c] == 1)
                fixedzidx += mincz
        else: # Could retrieve #occurences, pre-fix them!
            all_indices = [k for k in range(N)]
            for t in range(ntrees):
                max_occs = -1
                list_indices_t = true_occurences[t]
                occurences_t = np.unique(list_indices_t, return_counts=True)
                assert(sum(occurences_t[1]) == N)
                if max(occurences_t[1]) > max_occs:
                    max_occs = max(occurences_t[1])
                for occ_id, occ_nb in zip(occurences_t[0], occurences_t[1]):
                    model.Add(eta_vars[occ_id][t] == occ_nb)
                for occ_id in np.setdiff1d(all_indices, occurences_t[0]):
                    model.Add(eta_vars[occ_id][t] == 0)
                assert(max_occs <= maxbval)
        if use_mleobj == 0:
            # obj_vars[t][b]: Variables that will capture the difference between sum_{k} q_{tkb}  - N * p_b, for fixed t and b
            obj_vars = [ [ model.NewIntVar(-N,N, 'obj_%d_%d' % (t,b) ) for b in range(maxbval) ] for t in range(ntrees) ]

            # abs_obj_vars[t][b]: Variables that will capture the absolute value of obj_vars for fixed t and b
            abs_obj_vars = [ [ model.NewIntVar(0,N, 'absobj_%d_%d' % (t,b) ) for b in range(maxbval) ] for t in range(ntrees) ]

        objfun = []
        if use_mleobj:
            objfuncoeff = []
        for t in range(ntrees):
            for b in range(maxbval):
                if use_mleobj == 1:
                    for k in range(N):
                        objfuncoeff.append( int( 10 * math.log(Pexact[b]) ) )
                        objfun.append( q_vars[k][t][b] )

                else:
                    # Set obj value abs constraints
                    model.AddAbsEquality( abs_obj_vars[t][b], obj_vars[t][b] )

                    #Set relationship between obj_vars and q_vars
                    model.Add( cp_model.LinearExpr.Sum( [q_vars[k][t][bp] for k in range(N) for bp in range(b,maxbval) ] ) - int( N * P[b] ) == obj_vars[t][b] )

                    objfun.append( abs_obj_vars[t][b] )

        if use_mleobj == 1:
            model.Maximize( cp_model.LinearExpr.WeightedSum( objfun, objfuncoeff ) )
        else:
            model.Minimize( cp_model.LinearExpr.Sum(objfun) )

        # Contraints
        # one-hot encoding
        for k in range(N):
            for w in range(
                    len(one_hot_encoded_groups)):  # for each group of binary attributes one-hot encoding the same attribute
                model.Add(cp_model.LinearExpr.Sum([x_vars[k][i] for i in one_hot_encoded_groups[w]]) == 1)

            # Enforces that every sample must be in at most one class
            model.Add( cp_model.LinearExpr.Sum( z_vars[k] ) == 1)

            for t in range(ntrees):
                # Enforces relationship between counting variable eta and binary variables q
                ortools_version = str(ortools.__version__).split(".")
                if int(ortools_version[0]) <= 9 and int(ortools_version[1]) <= 8:
                    model.AddMapDomain( eta_vars[k][t], q_vars[k][t], offset=0 )
                else:
                    model.add_map_domain( eta_vars[k][t], q_vars[k][t], offset=0 )

        nleaves = []

        for tid, all_branches_t in enumerate(trees_branches):  # for each tree
            etayvars = [[] for k in
                              range(N)]  # for each example we will ensure it is captured by exactly one branch

            nleaves.append( len(all_branches_t) )
            for a_branch_nb, a_branch in enumerate(all_branches_t):  # iterate over its branches

                # Variables that will be involved in making sure number of samples in each leaf/branch is consistent
                branch_vars_c = [[] for c in range(C)]

                for k in range(N):
                    w_vars[k][tid].append( model.NewBoolVar('w_%d_%d_%d'%(k,tid,a_branch_nb)) )

                    # The loop right after is used to construct the constraints that if k is used in a given tree at a given leaf/branch
                    # Then the x's must respect all the splits within the corresponding branch
                    for a_split in a_branch[0]:
                        feature_val = a_split[0]
                        feature_id = abs(feature_val)-1
                        threshold_val = a_split[1]
                        if feature_id in num_indices:
                            idx = num_indices.index(feature_id)
                            threshold_val = self.numerical_attrs[idx][3].index(threshold_val) # replace the actual split value with its index (interval nb)
                        if feature_val > 0:
                            model.Add( x_vars[k][feature_id] >= int( math.floor( threshold_val ) ) + 1 ).OnlyEnforceIf( w_vars[k][tid][a_branch_nb] )
                        elif feature_val < 0:
                            model.Add(x_vars[k][feature_id] <= int( math.floor( threshold_val ) ) ).OnlyEnforceIf( w_vars[k][tid][a_branch_nb])
                        else:
                            raise ValueError("Feat 0 shouldn't be used here (1-indexed now)")

                    for c in range(C):
                        # Variable y represents how many times sample k is used in tree tid, node a_branch_nb,
                        #    being that k is classified as class c
                        y_vars[k][c][tid].append( model.NewIntVar(0, maxbval, 'y_%d_%d_%d_%d' % (tid, a_branch_nb, k, c)) )

                        etayvars[k].append(y_vars[k][c][tid][a_branch_nb])

                        branch_vars_c[c].append(y_vars[k][c][tid][a_branch_nb])


                        # Constraints that enforce consistency between w and y variables
                        model.Add( y_vars[k][c][tid][a_branch_nb] == 0 ).OnlyEnforceIf( w_vars[k][tid][a_branch_nb].Not() )

                        # Constraints that enforce consistency between y and z variables
                        model.Add( y_vars[k][c][tid][a_branch_nb] == 0 ).OnlyEnforceIf( z_vars[k][c].Not() )


                for c in range(C):
                    model.Add(cp_model.LinearExpr.Sum(branch_vars_c[c]) == int(
                        a_branch[1][c]))  # enforces the branch per-class cardinality

            for k in range(N):
                model.Add(
                    cp_model.LinearExpr.Sum(etayvars[k]) == eta_vars[k][tid] )  # eta (Number of samples in a tree) is consistent with y


            ## ADDS probabilistic constraints, i.e. constraints that are not necessarily valid, but hold with high probability
            ## High here means <= eps
            if useprobctr:
                eps = 0.005
                for b in range(2,maxbval):
                    # prob gets probability that a sample appears at least b times
                    prob = P[b]
                    cnt = 0
                    while prob > eps:
                        prob = prob*P[b]
                        cnt += 1
                    # This means that with prob >= 1-eps, cannot have more than cnt many q's having value at least b
                    print("Probabilistic constraint: At most %d samples appear %d or more times in a tree ( probability of this being true is %g) " %(cnt,b,1.0-prob) )
                    model.Add( cp_model.LinearExpr.Sum( [ q_vars[k][t][bp] for k in range(N) for bp in range(b,maxbval) ] ) <= cnt )

        # Ordre lexicographique au sein des classes
        '''deb = 0
        for c in range(C): # for each class
            for k in range(deb, deb + int(cards[c])-1):
                model.Add(sum(2 ** (i) * x_vars[k][i] for i in range(M)) <= sum(2 ** (i) * x_vars[k+1][i] for i in range(M)))
            deb += int(cards[c])'''

        if verbosity:
            print("Model creation done!")

        # Résolution
        solver = cp_model.CpSolver()

        # Sets a time limit of XX seconds.
        solver.parameters.log_search_progress = verbosity
        solver.parameters.max_time_in_seconds = time_out
        solver.parameters.num_workers = n_threads
        solver.parameters.random_seed = seed

        status = solver.Solve(model)

        end = time.time()
        duration = end - start

        # Récupération statut/valeurs
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            x_sol = [[solver.Value(x_vars[k][i]) for i in range(M)] for k in range(N)]
            obj_val = solver.ObjectiveValue()
        else:
            if status == cp_model.INFEASIBLE or status == cp_model.MODEL_INVALID:
                raise RuntimeError(
                    'Infeasible model: the reconstruction problem has no solution. Please make sure the provided one-hot encoding constraints are correct. Else, report this issue to the developers.')
            else:
                x_sol = None
                obj_val = float("nan")

        # Retrieve actual numerical attributes' values from their assigned interval ID
        for k in range(N if x_sol is not None else 0):
            for i in range(len(self.numerical_attrs)):
                attr_id = self.numerical_attrs[i][0]
                interval_id = x_sol[k][attr_id]
                reconstructed_value = (self.numerical_attrs[i][4][interval_id]+self.numerical_attrs[i][4][interval_id+1])/2
                x_sol[k][attr_id]=reconstructed_value
        if verbosity:
            print("*************************************************************")
            print("*************************************************************")
            print("  Solver specific:  Objval = %g,  duration = %g " % (obj_val, duration))
            print("*************************************************************")
            print("*************************************************************")

        solve_status = {0: 'UNKNOWN',
                        1: 'MODEL_INVALID',
                        2: 'FEASIBLE',
                        3: 'INFEASIBLE',
                        4: 'OPTIMAL'}[status]

        self.result_dict = {'max_max_depth': max_max_depth, 'status': solve_status, 'duration': duration,
                            'reconstructed_data': x_sol}
