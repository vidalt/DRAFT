from scipy.stats import binom
import math

class DRAFT:

    # HELPER FUNCTIONS
    # Functions to calculate probabilities for the bagging model
    def proba(self, k, N):
        return binom.pmf(k, N, 1 / N)

    def proba_inf(self, k, N):
        calc = 0
        for i in range(k):
            calc += self.proba(i, N)
        return calc

    def parse_forest(self, clf, verbosity=False):
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

        ## Parse the forest
        # Trees of the forest
        T = clf.estimators_

        # Nombre de features du jeu de données étudiées
        M = T[0].n_features_in_

        
        def retrieve_branches(number_nodes, children_left_list, children_right_list, nodes_features_list, nodes_value_list):
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
                            path[-1] = -nodes_features_list[origin]
                            paths[index] = path + [end_l]
                            path[-1] = nodes_features_list[origin]
                            paths.append(path + [end_r])

                    # Initialize path in first iteration
                    if i == 0:
                        paths.append([-nodes_features_list[i], children_left[i]])
                        paths.append([nodes_features_list[i], children_right[i]])

        max_max_depth = 0

        trees_branches = []
        first_tree = True
        maxcards = []
        # iterate over the trees of the forest
        for tree in T:
            t = tree.tree_

            cards = list(t.value[0][0])

            if first_tree: # Init constants (only do it once)
                # Nombre de classes (lisible par exemple ici en regardant la taille du tableau des cardinalités à la racine de l'arbre 0)
                C = len(cards)

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
            '''print(" Cards = " + str( cards ) )
            print(" maxcards = " + str( maxcards) )'''

            max_depth = t.max_depth

            if max_depth > max_max_depth:
                max_max_depth = max_depth

            n_nodes = t.node_count
            children_left = t.children_left # For all nodes in the tree, list of their left children (or -1 for leaves)
            children_right = t.children_right # For all nodes in the tree, list of their right children (or -1 for leaves)
            nodes_features = t.feature # For all nodes in the tree, list of their used feature (or -2 for leaves)
            nodes_value = t.value # For all nodes in the tree, list of their value (support for both classes)
            nodes_features += 1

            all_branches = list(retrieve_branches(n_nodes, children_left, children_right, nodes_features, nodes_value))

            trees_branches.append(all_branches)

        if verbosity:
            print("RF parsing done!")

        return T, M, N, C, Z, max_max_depth, trees_branches, maxcards
    
    # MAIN FUNCTIONS
    def __init__(self, random_forest, one_hot_encoded_groups=[]):
        """
        Constructor.

        Attributes
        ---------
        random_forest: instance of sklearn.ensemble.RandomForestClassifier
                       The random forest whose structure will be leveraged to reconstruct its training set

        one_hot_encoded_groups: list, optional
                        list of lists, where each sub-list contains the IDs of a group of attributes corresponding to a one-hot encoding of the same original feature.
                        not mandatory but if provided, can improve the performed reconstruction and may speed up the process

        """
        from sklearn.ensemble import RandomForestClassifier
        if not isinstance(random_forest, RandomForestClassifier):
            raise TypeError("Expected a RandomForestClassifier but provided random_forest is of type " + str(type(random_forest)))
        self.clf = random_forest
        self.ohe_groups = one_hot_encoded_groups

    def fit(self, bagging=False, method='cp-sat', timeout=60, verbosity=True, n_jobs=-1, seed=0):
        """
        Reconstructs a dataset compatible with the knowledge provided by random_forest.
        In other terms, fits the data to the given model.

        Arguments
        ---------
        bagging: bool, optional (default False)
                        whether bootstrap sampling was used to train the base learners
                        the reconstruction model will be constructed accordingly

        method: str in {'cp-sat', 'milp'}, optional (default 'cp-sat')
                        the type of formulation that will be used to perform the reconstruction
                        Note that `cp-sat` requires the OR-Tools Python library 
                        and `milp` the GurobiPy one (see the Installation section of our README).

        timeout: int, optional (default 60)
                        maximum cpu time (in seconds) to be used by the search
                        if the solver is not able to return a solution within the given time frame, it will be indicated in the returned dictionary

        verbosity: bool, optional (default True)
                        whether to print information about the search progress or not

        n_jobs: int in {-1, positives}, optional (default -1)
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
        """
        if not method in ['cp-sat', 'milp']:
            raise ValueError("Supported methods are either 'cp-sat' or 'milp', got: " + method)
        if method == 'cp-sat':

            # Whether to use the alternative formulation (described in alt_cp_model_bag.tex)
            use_alt = 0

            # Whether to use the new maximum likelihood objective
            # (the old one minimizes the absolute difference to the cumulative distribution of
            # probability that a sample is used at least b times, for every tree)
            use_mleobj = 1

            # Cannot use MLE objective with alternative formulation
            assert( use_mleobj + use_alt <= 1 )

            # Whether to use constraints that are not necessarily valid, but valid with high probability (measured by epsilon specified within that function)
            useprobctr = 0

            if n_jobs == -1:
                n_jobs = 0 # value for OR-Tools
            if not bagging:
                self.perform_reconstruction_v1_CP_SAT(n_threads=n_jobs, time_out=timeout, verbosity=verbosity, seed=seed)
            else:
                if use_alt:
                    self.perform_reconstruction_v2_CP_SAT_alt(n_threads=n_jobs, time_out=timeout, verbosity=verbosity, seed=seed, useprobctr=useprobctr)
                else:
                    self.perform_reconstruction_v2_CP_SAT(n_threads=n_jobs, time_out=timeout, verbosity=verbosity, seed=seed, use_mleobj=use_mleobj, useprobctr=useprobctr )

        elif not bagging and method == 'milp':
            if n_jobs == -1:
                n_jobs = 0 # value for Gurobi
            self.perform_reconstruction_v1_MILP(n_threads=n_jobs, time_out=timeout, verbosity=int(verbosity), seed=seed)
        else:
            raise AttributeError("Currently bagging is not supported.")

        if hasattr(self, 'result_dict'):
            return self.result_dict
        else:
            raise RuntimeError('Something went wrong and the reconstruction could not be performed. Please report this issue to the developers.')

    def perform_reconstruction_v1_CP_SAT(self, n_threads=0, time_out=60, verbosity=1, seed=0):
        """
        Constructs and solves the CP based dataset reconstruction model (without the use of bagging to train the target random forest) using the OR-Tools CP-SAT solver.

        Arguments
        ---------
        n_threads: int >= 0, optional (default 0)
                        maximum number of threads to be used by the solver to parallelize search
                        if 0, use all available threads

        time_out: int, optional (default 60)
                        maximum cpu time (in seconds) to be used by the search
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
        """
        from ortools.sat.python import cp_model
        import numpy as np # useful
        import time # time measurements

        clf = self.clf
        one_hot_encoded_groups = self.ohe_groups

        start = time.time()


        T, M, N, C, Z, max_max_depth, trees_branches, _ = self.parse_forest(clf, verbosity=verbosity)

        ### Create the CP model

        ## Variables
        model = cp_model.CpModel()

        x_vars = [[model.NewBoolVar('x_%d_%d' %(k,i)) for i in range(M)] for k in range(N)] # table of x_{ki}

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
                            literals = []
                            for a_split in a_branch[0]:
                                if a_split > 0:
                                    literals.append(x_vars[k][abs(a_split)-1])
                                elif a_split < 0:
                                    literals.append(x_vars[k][abs(a_split)-1].Not())
                                else:
                                    raise ValueError("Feat 0 shouldn't be used here (1-indexed now)")
                            model.AddBoolAnd(literals).OnlyEnforceIf(local_capt_var)
                for c in range(C):
                    model.Add(cp_model.LinearExpr.Sum(branch_vars_c[c]) ==  int(a_branch[1][c])) # enforces the branch per-class cardinality
            for k in range(N):
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
        else:
            if status == cp_model.INFEASIBLE or status == cp_model.MODEL_INVALID:
                raise RuntimeError('Infeasible model: the reconstruction problem has no solution. Please make sure the provided one-hot encoding constraints are correct. Else, report this issue to the developers.')
            else:
                x_sol = np.random.randint(2,size = (N,M))

        solve_status = {0: 'UNKNOWN',
        1: 'MODEL_INVALID',
        2: 'FEASIBLE',
        3: 'INFEASIBLE',
        4:'OPTIMAL'}[status]

        self.result_dict = {'max_max_depth':max_max_depth, 'status':solve_status, 'duration': duration, 'reconstructed_data':x_sol}

    def perform_reconstruction_v1_MILP(self, n_threads=0, time_out=60, verbosity=1, seed=0):
        """
        Constructs and solves the MILP based dataset reconstruction model (without the use of bagging to train the target random forest) using the Gurobi MILP solver through its Python wrapper.

        Arguments
        ---------
        n_threads: int in {-1, positives}, optional (default -1)
                        maximum number of threads to be used by the solver to parallelize search
                        if -1, use all available threads

        time_out: int, optional (default 60)
                        maximum cpu time (in seconds) to be used by the search
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
        """
        import gurobipy as gp # solver
        import numpy as np # useful
        from gurobipy import GRB, quicksum # solver
        import time # time measurements

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
            Nb_c.append(t.value.tolist())

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
        x_sol = x.getAttr(GRB.Attr.X)

        x_sol = x_sol.tolist()

        self.result_dict = {'max_max_depth':max_max_depth, 'status':solve_status, 'duration': duration, 'reconstructed_data':x_sol}

    def perform_reconstruction_v2_CP_SAT(self, n_threads=0, time_out=60, verbosity=1, seed=0, use_mleobj=1, useprobctr = 0 ):
        """
        Constructs and solves the CP based dataset reconstruction model (with the use of bagging to train the target random forest) using the OR-Tools CP-SAT solver.

        Arguments
        ---------
        n_threads: int >= 0, optional (default 0)
                        maximum number of threads to be used by the solver to parallelize search
                        if 0, use all available threads

        time_out: int, optional (default 60)
                        maximum cpu time (in seconds) to be used by the search
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
        """
        from ortools.sat.python import cp_model
        import numpy as np  # useful
        import time  # time measurements


        # This is the maximum number of times a sample can appear in a tree (note it will go from 0 to maxbval-1)
        maxbval = 8

        clf = self.clf
        one_hot_encoded_groups = self.ohe_groups

        start = time.time()

        ### Create the CP model

        ## Parse the forest
        T, M, N, C, Z, max_max_depth, trees_branches, maxcards = self.parse_forest(clf, verbosity=verbosity)

        # Defines the probabilities that an item will appear b times
        P = []
        Pexact = [0 for i in range(maxbval)]
        for i in range(maxbval):
            #P.append( 1 - self.proba_inf(i + 1, N) )
            P.append(1 - self.proba_inf(i , N))
        for i in range(maxbval):
            if i < maxbval - 1:
                Pexact[i] = P[i] - P[i+1]
            else:
                Pexact[i] = P[i]



        if verbosity:
            print("Probabilities of an item appearing at least b times:")
            print(P)
            print(sum(P))

            print("Probabilities of an item appearing at EXACTLY b times:")
            print(Pexact)
            print(sum(Pexact))

        ntrees = len( trees_branches )

        ## Variables
        model = cp_model.CpModel()

        # x[k][i] : Variables that represent what is sample k (each of its features i)
        x_vars = [[model.NewBoolVar('x_%d_%d' % (k, i)) for i in range(M)] for k in range(N)]  # table of x_{ki}

        # y_vars[k][c][t][v]: Variables that represent the number of times sample k is used as class c
        #   in leaf/branch v of tree t
        y_vars = [[[[] for t in range(ntrees)] for c in range(C)] for k in range(N)]

        # w_vars[k][t][v]: Variables that represent if sample k is used in leaf v of tree t
        w_vars = [[[] for t in range(ntrees) ] for k in range(N)]

        # z_vars[k][c]: Variables that represent if sample k is assigned class c
        z_vars = [[model.NewBoolVar('z_%d_%d'%(k,c)) for c in range(C) ] for k in range(N) ]


        fixedzidx = 0
        for c in range(C):
            mincz = math.floor( maxcards[c] / maxbval)
            for offset in range(mincz):
                model.Add( z_vars[fixedzidx+offset][c] == 1)
            fixedzidx += mincz



        # eta_vars[k][t]: Variables that count how many times sample k is used in tree t
        eta_vars = [[ model.NewIntVar(0,maxbval, 'eta_%d_%d'%(k,t)) for t in range(ntrees)] for k in range(N) ]

        # q_vars[k][t][b] Variables that represent if sample k appears b times in tree t (needed for objective function
        q_vars = [[[model.NewBoolVar('q_%d_%d_%d' %( t, k, b )) for b in range(maxbval) ] for t in range(ntrees) ] for k in range(N) ]

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
                model.AddMapDomain( eta_vars[k][t], q_vars[k][t], offset=0 )


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

                    # literals is used to construct the constraint that if k is used in a given tree at a given leaf/branch
                    # Then the x's must respect the corresponding branch
                    literals = []
                    for a_split in a_branch[0]:
                        if a_split > 0:
                            literals.append(x_vars[k][abs(a_split) - 1])
                        elif a_split < 0:
                            literals.append(x_vars[k][abs(a_split) - 1].Not())
                        else:
                            raise ValueError("Feat 0 shouldn't be used here (1-indexed now)")
                    # Constraint that enforce consistency between w and x variables
                    model.AddBoolAnd(literals).OnlyEnforceIf(w_vars[k][tid][a_branch_nb])

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



        # Some of this assumes that there are only two classes, so we will check this
        singlerun = 1
        if singlerun:
            nfreezperrun = N
        else:
            assert( C == 2 )
            # Let the solver decide on nfreezperrun number of z's at a time
            nfreezperrun = 5

        nruns = math.ceil( (N - fixedzidx) / nfreezperrun )
        if nruns == 0 or singlerun == 1:
            nruns = 1

        dur_runs = [ 0 for r in range(nruns) ]
        if use_mleobj:
            best_obj = -1000000000
        else:
            best_obj = 1000000000
        best_x = np.random.randint(2, size=(N, M))
        for r in range(nruns):
            # For every run, the variables from (fizedzidx) until min(N-1, fixedzidx + (r) * nfreezperrun - 1 )
            # will be fixed to 1.
            # The variables from fixedidx + (r) * nfreezperrun until min(N-1, fixedzidx + (r+1) * nfreezperrun - 1 )
            # will be left free
            # The remaining variables will be fixed to 0
            if singlerun == 0:
                fix1 = fixedzidx + r * nfreezperrun
                fix0 = fixedzidx + (r+1) * nfreezperrun
                print(" Run %d of %d ------------ "%(r, nruns)  )
                print("      Fixed to 1: from %d to %d "% (fixedzidx,fix1) )
                print("      Fixed to 0: from %d to %d "% (fix0,N) )

                for k in range( fixedzidx, N ):
                    z_vars[k][1].Proto().domain[:] = []
                    z_vars[k][0].Proto().domain[:] = []
                    if k < fix1:
                        print(" Fixing %d to 1 " % k )
                        z_vars[k][1].Proto().domain.extend(cp_model.Domain(1, 1).FlattenedIntervals())
                        z_vars[k][0].Proto().domain.extend(cp_model.Domain(0, 0).FlattenedIntervals())
                    elif k >=  fix0:
                        print(" Fixing %d to 0 " % k)
                        z_vars[k][1].Proto().domain.extend(cp_model.Domain(0, 0).FlattenedIntervals())
                        z_vars[k][0].Proto().domain.extend(cp_model.Domain(1, 1).FlattenedIntervals())
                    else:
                        z_vars[k][1].Proto().domain.extend(cp_model.Domain(0, 1).FlattenedIntervals())
                        z_vars[k][0].Proto().domain.extend(cp_model.Domain(0, 1).FlattenedIntervals())

            # Résolution
            solver = cp_model.CpSolver()

            # Sets a time limit of XX seconds.
            solver.parameters.log_search_progress = verbosity
            solver.parameters.max_time_in_seconds = time_out / nruns
            solver.parameters.num_workers = n_threads
            solver.parameters.random_seed = seed

            status = solver.Solve(model)

            end = time.time()
            duration = end - start
            dur_runs[r] = duration

            print(" --- RUN %d ---  %g seconds" % (r,duration) )

            # Récupération statut/valeurs
            if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
                x_sol = [[solver.Value(x_vars[k][i]) for i in range(M)] for k in range(N)]

                obj_val = solver.ObjectiveValue()
                if use_mleobj:
                    if obj_val > best_obj:
                        print(" Found new best solution, with value " + str(obj_val) + "  at run " + str(r))
                        best_obj = obj_val
                        best_x = x_sol.copy()
                else:
                    if obj_val < best_obj:
                        print(" Found new best solution, with value " + str(obj_val) + "  at run " + str(r) )
                        best_obj = obj_val
                        best_x = x_sol.copy()

                # GET and print solution (For debugging mostly)
                debug = 0
                if debug :
                    y_sol = [[[[] for t in range(ntrees)] for c in range(C) ] for k in range(N) ]

                    # w_vars[k][t][v]: Variables that represent if sample k is used in leaf v of tree t
                    w_sol = [[[] for t in range(ntrees)] for k in range(N)]

                    # z_vars[k][c]: Variables that represent if sample k is assigned class c
                    z_sol = [[solver.Value(z_vars[k][c]) for c in range(C)] for k in range(N)]

                    # eta_vars[k][t]: Variables that count how many times sample k is used in tree t
                    eta_sol = [[solver.Value( eta_vars[k][t] ) for t in range(ntrees)] for k in range(N)]

                    # q_vars[k][t][b] Variables that represent if sample k appears b times in tree t (needed for objective function
                    q_sol = [[[ solver.Value( q_vars[k][t][b]) for b in range(maxbval)] for t in range(ntrees)] for k in range(N)]

                    count_samples = [ 0 for t in range(ntrees) ]
                    count_samples_fromy = [ 0 for t in range(ntrees) ]
                    count_bvals = [ [0 for b in range(maxbval)] for t in range(ntrees) ]

                    for k in range(N):
                        for t in range(ntrees):
                            w_sol[k][t] = [ solver.Value(w_vars[k][t][v]) for v in range(len(w_vars[k][t])) ]
                            for c in range(C):
                                y_sol[k][c][t] = [ solver.Value(y_vars[k][c][t][v]) for v in range(len(y_vars[k][c][t]))  ]


                    for k in range(N):
                        print("SOL: SAMPLE %d  :  "%(k)  + str( x_sol[k] ) )
                        kc = -1
                        for c in range(C):
                            if z_sol[k][c] == 1:
                                # Check that class has not changed
                                assert( kc == -1 )
                                kc = c
                                print("   - assigned class %d" %(c))

                        # Check that it has been assigned some class c
                        assert( kc != -1 )

                        for t in range(ntrees):
                            if eta_sol[k][t] > 0:
                                count_samples[t] += eta_sol[k][t]
                                print("   - used %d times in tree %d"%(eta_sol[k][t],t))

                            for b in range(maxbval):
                                if int(eta_sol[k][t]) == b:
                                    assert( q_sol[k][t][b] == 1 )
                                    count_bvals[t][b] += 1
                                else:
                                    assert( q_sol[k][t][b] == 0 )

                            for v in range(len(y_sol[k][kc][t])):
                                if y_sol[k][kc][t][v] > 0:
                                    assert( w_sol[k][t][v] > 0 )
                                    count_samples_fromy[t] += y_sol[k][kc][t][v]
                                    print("   -  at node %d: %d times"%(v,y_sol[k][kc][t][v]))

                    ## Tree view
                    print('------ Tree view -------')
                    for t in range(ntrees):
                        for v in range(nleaves[t]):
                            print('  Samples at node %d of tree %d' % (v, t))
                            for c in range(C):
                                for k in range(N):
                                    if y_sol[k][c][t][v] > 0:
                                        print("     Sample %d of class %d appears %d times" % (k, c, y_sol[k][c][t][v]))

                    print( count_samples )
                    print( count_samples_fromy )
                    for t in range(ntrees):
                        print( "Distribution of samples for tree %d is:"%t + str( [ count_bvals[t][b] / N for b in range(maxbval) ]))
                        print( " Expected (values of exact p) were:" + str(Pexact) )
            else:
                # If it is not a singlerun, we will try other times, so infeasibility is not an issue
                if singlerun:
                    if status == cp_model.INFEASIBLE or status == cp_model.MODEL_INVALID:
                        raise RuntimeError(
                            'Infeasible model: the reconstruction problem has no solution. Please make sure the provided one-hot encoding constraints are correct. Else, report this issue to the developers.')
                    else:
                        x_sol = np.random.randint(2, size=(N, M))

        x_sol = best_x
        duration = sum( dur_runs )

        print("*************************************************************")
        print("*************************************************************")
        print("  Solver specific:  Objval = %d,  duration = %g " % (best_obj, duration))
        print("*************************************************************")
        print("*************************************************************")

        solve_status = {0: 'UNKNOWN',
                        1: 'MODEL_INVALID',
                        2: 'FEASIBLE',
                        3: 'INFEASIBLE',
                        4: 'OPTIMAL'}[status]

        self.result_dict = {'max_max_depth': max_max_depth, 'status': solve_status, 'duration': duration,
                            'reconstructed_data': x_sol}

    def perform_benchmark_partial_examples(self, n_threads=0, time_out=60, verbosity=1, seed=0, X_known = [], y_known=[]):
        """
        Runs the complementary experiments on reconstruction with knowledge of part of the training set examples, mentionned in the Appendix D of our paper.
        The model builds upon the CP based dataset reconstruction model (with the use of bagging to train the target random forest) using the OR-Tools CP-SAT solver,
        but pre-fixes a number of (supposedly known) training set examples.

        Arguments
        ---------
        n_threads: int >= 0, optional (default 0)
                        maximum number of threads to be used by the solver to parallelize search
                        if 0, use all available threads

        time_out: int, optional (default 60)
                        maximum cpu time (in seconds) to be used by the search
                        if the solver is not able to return a solution within the given time frame, it will be indicated in the returned dictionary

        verbosity: int, optional (default 1)
                        whether to print information (1) about the search progress or not (0)

        seed: int, optional (default 0)
                       random number generator seed
                       used to fix the behaviour of the solvers

        X_known: array-like, shape = [n_known, n_features] (default [])
                        the set of known examples 
                        with n_known <= N (there can not be more known than actual examples)
                        and n_features == M (for the known examples we know all of their features)

        y_known: array-like, shape = [n_known] (default [])
                        the labels of the known examples 
                        with n_known <= N (there can not be more known than actual examples)
        
        Returns
        -------
        output: dictionary containing:
            -> 'max_max_depth': maximum depth found when parsing the trees within the forest. 
            -> 'status': the solve status returned by the solver. It can be 'UNKNOWN', 'MODEL_INVALID', 'FEASIBLE', 'INFEASIBLE', or 'OPTIMAL'.
            -> 'duration': duration
            -> 'reconstructed_data': array of shape = [n_samples, n_attributes] encoding the reconstructed dataset.
        """
        from ortools.sat.python import cp_model
        import numpy as np  # useful
        import time  # time measurements

        # This is the maximum number of times a sample can appear in a tree (note it will go from 0 to maxbval-1)
        maxbval = 8

        clf = self.clf
        one_hot_encoded_groups = self.ohe_groups

        start = time.time()

        ### Create the CP model

        ## Parse the forest
        T, M, N, C, Z, max_max_depth, trees_branches, maxcards = self.parse_forest(clf, verbosity=verbosity)

        # Defines the probabilities that an item will appear b times
        P = []
        Pexact = [0 for i in range(maxbval)]
        for i in range(maxbval):
            #P.append( 1 - self.proba_inf(i + 1, N) )
            P.append(1 - self.proba_inf(i , N))
        for i in range(maxbval):
            if i < maxbval - 1:
                Pexact[i] = P[i] - P[i+1]
            else:
                Pexact[i] = P[i]



        if verbosity:
            print("Probabilities of an item appearing at least b times:")
            print(P)
            print(sum(P))

            print("Probabilities of an item appearing at EXACTLY b times:")
            print(Pexact)
            print(sum(Pexact))

        ntrees = len( trees_branches )

        ## Variables
        model = cp_model.CpModel()

        # x[k][i] : Variables that represent what is sample k (each of its features i)
        x_vars = [[model.NewBoolVar('x_%d_%d' % (k, i)) for i in range(M)] for k in range(N)]  # table of x_{ki}

        # y_vars[k][c][t][v]: Variables that represent the number of times sample k is used as class c
        #   in leaf/branch v of tree t
        y_vars = [[[[] for t in range(ntrees)] for c in range(C)] for k in range(N)]

        # w_vars[k][t][v]: Variables that represent if sample k is used in leaf v of tree t
        w_vars = [[[] for t in range(ntrees) ] for k in range(N)]

        # z_vars[k][c]: Variables that represent if sample k is assigned class c
        z_vars = [[model.NewBoolVar('z_%d_%d'%(k,c)) for c in range(C) ] for k in range(N) ]

        # Assume knowledge of dataset
        assert(X_known.shape[0] == y_known.size)
        assert(X_known.shape[0] <= N)
        assert(X_known.shape[1] == M)

        for k in range(X_known.shape[0]):
            for w in range(len(one_hot_encoded_groups)):
                onehotsum = 0
                for i in one_hot_encoded_groups[w]:
                    onehotsum += X_known[k][i]
                if onehotsum != 1:
                    print( " ERROR: SAMPLE " + str(k) + " has onehotencoding total " + str(onehotsum) + " for onehot: " + str(one_hot_encoded_groups[w]))
                    exit(1)
            for i in range(M):
                model.Add( x_vars[k][i] == X_known[k][i] )
            for c in range(C):
                if y_known[k] == c:
                    model.Add(z_vars[k][c] == 1)

        if verbosity:
            print("==> bench partial reconstr (examples): prefixed %d examples" %y_known.size)

        classes, classes_counts = np.unique(y_known, return_counts=True)

        fixedzidx = X_known.shape[0] # start pre-fixing after the pre-fixed examples
        for c in range(C):
            if c in classes:
                class_count = classes_counts[c]
            else:
                class_count = 0
            mincz = math.floor( maxcards[c] / maxbval) - class_count # subtract already set examples from this class
            for offset in range(mincz):
                model.Add( z_vars[fixedzidx+offset][c] == 1)
            fixedzidx += mincz



        # eta_vars[k][t]: Variables that count how many times sample k is used in tree t
        eta_vars = [[ model.NewIntVar(0,maxbval, 'eta_%d_%d'%(k,t)) for t in range(ntrees)] for k in range(N) ]

        # q_vars[k][t][b] Variables that represent if sample k appears b times in tree t (needed for objective function
        q_vars = [[[model.NewBoolVar('q_%d_%d_%d' %( t, k, b )) for b in range(maxbval) ] for t in range(ntrees) ] for k in range(N) ]

        objfun = []
        objfuncoeff = []
        for t in range(ntrees):
            for b in range(maxbval):
                for k in range(N):
                    objfuncoeff.append( int( 10 * math.log(Pexact[b]) ) )
                    objfun.append( q_vars[k][t][b] )

        model.Maximize( cp_model.LinearExpr.WeightedSum( objfun, objfuncoeff ) )

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
                model.AddMapDomain( eta_vars[k][t], q_vars[k][t], offset=0 )


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

                    # literals is used to construct the constraint that if k is used in a given tree at a given leaf/branch
                    # Then the x's must respect the corresponding branch
                    literals = []
                    for a_split in a_branch[0]:
                        if a_split > 0:
                            literals.append(x_vars[k][abs(a_split) - 1])
                        elif a_split < 0:
                            literals.append(x_vars[k][abs(a_split) - 1].Not())
                        else:
                            raise ValueError("Feat 0 shouldn't be used here (1-indexed now)")
                    # Constraint that enforce consistency between w and x variables
                    model.AddBoolAnd(literals).OnlyEnforceIf(w_vars[k][tid][a_branch_nb])

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
        elif status == cp_model.INFEASIBLE or status == cp_model.MODEL_INVALID:
            raise RuntimeError(
                'Infeasible model: the reconstruction problem has no solution. Please make sure the provided one-hot encoding constraints are correct. Else, report this issue to the developers.')
        else:
            x_sol = np.random.randint(2, size=(N, M))

        solve_status = {0: 'UNKNOWN',
                        1: 'MODEL_INVALID',
                        2: 'FEASIBLE',
                        3: 'INFEASIBLE',
                        4: 'OPTIMAL'}[status]

        self.result_dict = {'max_max_depth': max_max_depth, 'status': solve_status, 'duration': duration,
                            'reconstructed_data': x_sol}
        
        return self.result_dict

    def perform_benchmark_partial_attributes(self, n_threads=0, time_out=60, verbosity=1, seed=0, X_known = [], known_attributes=[]):
        """
        Runs the complementary experiments on reconstruction with knowledge of part of the training set attributes, described in the Appendix D of our paper. 
        The model builds upon the CP based dataset reconstruction model (with the use of bagging to train the target random forest) using the OR-Tools CP-SAT solver, 
        but pre-fixes a number of (supposedly known) training set attributes (columns).

        Arguments
        ---------
        n_threads: int >= 0, optional (default 0)
                        maximum number of threads to be used by the solver to parallelize search
                        if 0, use all available threads

        time_out: int, optional (default 60)
                        maximum cpu time (in seconds) to be used by the search
                        if the solver is not able to return a solution within the given time frame, it will be indicated in the returned dictionary

        verbosity: int, optional (default 1)
                        whether to print information (1) about the search progress or not (0)

        seed: int, optional (default 0)
                       random number generator seed
                       used to fix the behaviour of the solvers

        X_known: array-like, shape = [n_known, n_features] (default [])
                        The entire training set
                        with n_known = N
                        and n_features == M
                        (note that in the current implementation we provide the entire training set
                        although only the columns specified by known_attributes are known) 

        known_attributes: array-like, shape = [n_features'] (default [])
                        the features (id) that are known 
                        (the associated entire colums provided through X_known will be fixed)
                        with n_features' <= M (there can not be more known than actual features)
        
        Returns
        -------
        output: dictionary containing:
            -> 'max_max_depth': maximum depth found when parsing the trees within the forest. 
            -> 'status': the solve status returned by the solver. It can be 'UNKNOWN', 'MODEL_INVALID', 'FEASIBLE', 'INFEASIBLE', or 'OPTIMAL'.
            -> 'duration': duration
            -> 'reconstructed_data': array of shape = [n_samples, n_attributes] encoding the reconstructed dataset.
        """
        from ortools.sat.python import cp_model
        import numpy as np  # useful
        import time  # time measurements

        # This is the maximum number of times a sample can appear in a tree (note it will go from 0 to maxbval-1)
        maxbval = 8

        clf = self.clf
        one_hot_encoded_groups = self.ohe_groups

        start = time.time()

        ### Create the CP model

        ## Parse the forest
        T, M, N, C, Z, max_max_depth, trees_branches, maxcards = self.parse_forest(clf, verbosity=verbosity)

        # Defines the probabilities that an item will appear b times
        P = []
        Pexact = [0 for i in range(maxbval)]
        for i in range(maxbval):
            #P.append( 1 - self.proba_inf(i + 1, N) )
            P.append(1 - self.proba_inf(i , N))
        for i in range(maxbval):
            if i < maxbval - 1:
                Pexact[i] = P[i] - P[i+1]
            else:
                Pexact[i] = P[i]

        if verbosity:
            print("Probabilities of an item appearing at least b times:")
            print(P)
            print(sum(P))

            print("Probabilities of an item appearing at EXACTLY b times:")
            print(Pexact)
            print(sum(Pexact))

        ntrees = len( trees_branches )

        ## Variables
        model = cp_model.CpModel()

        # x[k][i] : Variables that represent what is sample k (each of its features i)
        x_vars = [[model.NewBoolVar('x_%d_%d' % (k, i)) for i in range(M)] for k in range(N)]  # table of x_{ki}

        # y_vars[k][c][t][v]: Variables that represent the number of times sample k is used as class c
        #   in leaf/branch v of tree t
        y_vars = [[[[] for t in range(ntrees)] for c in range(C)] for k in range(N)]

        # w_vars[k][t][v]: Variables that represent if sample k is used in leaf v of tree t
        w_vars = [[[] for t in range(ntrees) ] for k in range(N)]

        # z_vars[k][c]: Variables that represent if sample k is assigned class c
        z_vars = [[model.NewBoolVar('z_%d_%d'%(k,c)) for c in range(C) ] for k in range(N) ]

        # Assume knowledge of dataset
        assert(X_known.shape[1] >= len(known_attributes))
        assert(X_known.shape[1] == M) # not mandatory actually but that's what I do in the experiments
        assert(X_known.shape[0] == N)

        for i in known_attributes:
            for k in range(N):
                model.Add( x_vars[k][i] == X_known[k][i] )

        if verbosity:
            print("==> bench partial reconstr (attributes): prefixed %d attributes" %len(known_attributes))

        # eta_vars[k][t]: Variables that count how many times sample k is used in tree t
        eta_vars = [[ model.NewIntVar(0,maxbval, 'eta_%d_%d'%(k,t)) for t in range(ntrees)] for k in range(N) ]

        # q_vars[k][t][b] Variables that represent if sample k appears b times in tree t (needed for objective function
        q_vars = [[[model.NewBoolVar('q_%d_%d_%d' %( t, k, b )) for b in range(maxbval) ] for t in range(ntrees) ] for k in range(N) ]

        objfun = []
        objfuncoeff = []
        for t in range(ntrees):
            for b in range(maxbval):
                for k in range(N):
                    objfuncoeff.append( int( 10 * math.log(Pexact[b]) ) )
                    objfun.append( q_vars[k][t][b] )


        model.Maximize( cp_model.LinearExpr.WeightedSum( objfun, objfuncoeff ) )


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
                model.AddMapDomain( eta_vars[k][t], q_vars[k][t], offset=0 )


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

                    # literals is used to construct the constraint that if k is used in a given tree at a given leaf/branch
                    # Then the x's must respect the corresponding branch
                    literals = []
                    for a_split in a_branch[0]:
                        if a_split > 0:
                            literals.append(x_vars[k][abs(a_split) - 1])
                        elif a_split < 0:
                            literals.append(x_vars[k][abs(a_split) - 1].Not())
                        else:
                            raise ValueError("Feat 0 shouldn't be used here (1-indexed now)")
                    # Constraint that enforce consistency between w and x variables
                    model.AddBoolAnd(literals).OnlyEnforceIf(w_vars[k][tid][a_branch_nb])

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
        elif status == cp_model.INFEASIBLE or status == cp_model.MODEL_INVALID:
            raise RuntimeError(
                'Infeasible model: the reconstruction problem has no solution. Please make sure the provided one-hot encoding constraints are correct. Else, report this issue to the developers.')
        else:
            x_sol = np.random.randint(2, size=(N, M))
            obj_val = -1

        solve_status = {0: 'UNKNOWN',
                        1: 'MODEL_INVALID',
                        2: 'FEASIBLE',
                        3: 'INFEASIBLE',
                        4: 'OPTIMAL'}[status]

        self.result_dict = {'max_max_depth': max_max_depth, 'status': solve_status, 'duration': duration,
                            'reconstructed_data': x_sol}
        
        return self.result_dict
    
    def perform_reconstruction_v2_CP_SAT_alt(self, n_threads=0, time_out=60, verbosity=1, seed=0, useprobctr = 0):
        """
        Constructs and solves an alternate formulation of our CP based dataset reconstruction model (with the use of bagging to train the target random forest) using the OR-Tools CP-SAT solver.
        Note that its objective function is NOT the one presented in our paper 
        but rather minimizes the absolute difference to the cumulative distribution of
        probability that a sample is used at least b times, for every tree.

        Arguments
        ---------
        n_threads: int >= 0, optional (default 0)
                        maximum number of threads to be used by the solver to parallelize search
                        if 0, use all available threads

        time_out: int, optional (default 60)
                        maximum cpu time (in seconds) to be used by the search
                        if the solver is not able to return a solution within the given time frame, it will be indicated in the returned dictionary

        verbosity: int, optional (default 1)
                        whether to print information (1) about the search progress or not (0)

        seed: int, optional (default 0)
                       random number generator seed
                       used to fix the behaviour of the solvers

        (it is not recommended to play with the argument below)

        useprobctr: int, optional (default 0)
                       Whether to use constraints that are not necessarily valid, but valid with high probability (measured by epsilon specified within that function) (1) or not (0).
        Returns
        -------
        output: dictionary containing:
            -> 'max_max_depth': maximum depth found when parsing the trees within the forest. 
            -> 'status': the solve status returned by the solver. It can be 'UNKNOWN', 'MODEL_INVALID', 'FEASIBLE', 'INFEASIBLE', or 'OPTIMAL'.
            -> 'duration': duration
            -> 'reconstructed_data': array of shape = [n_samples, n_attributes] encoding the reconstructed dataset.
        """
        from ortools.sat.python import cp_model
        import numpy as np  # useful
        import time  # time measurements


        # This is the maximum number of times a sample can appear in a tree (note it will go from 0 to maxbval-1)
        maxbval = 8

        clf = self.clf
        one_hot_encoded_groups = self.ohe_groups

        start = time.time()

        ### Create the CP model

        ## Parse the forest
        T, M, N, C, Z, max_max_depth, trees_branches, maxcards = self.parse_forest(clf, verbosity=verbosity)

        # Defines the probabilities that an item will appear b times
        P = []
        Pexact = [0 for i in range(maxbval)]
        for i in range(maxbval):
            #P.append( 1 - self.proba_inf(i + 1, N) )
            P.append(1 - self.proba_inf(i , N))
        for i in range(maxbval):
            if i < maxbval - 1:
                Pexact[i] = P[i] - P[i+1]
            else:
                Pexact[i] = P[i]



        if verbosity:
            print("Probabilities of an item appearing at least b times:")
            print(P)
            print(sum(P))

            print("Probabilities of an item appearing at EXACTLY b times:")
            print(Pexact)
            print(sum(Pexact))

        ntrees = len( trees_branches )

        ## Variables
        model = cp_model.CpModel()

        # x[k][i] : Variables that represent what is sample k (each of its features i)
        x_vars = [[model.NewBoolVar('x_%d_%d' % (k, i)) for i in range(M)] for k in range(N)]  # table of x_{ki}

        # y_vars[k][c][t][v]: Variables that represent the number of times sample k is used as class c
        #   in leaf/branch v of tree t
        y_vars = [[[[] for t in range(ntrees)] for c in range(C)] for k in range(N)]

        # z_vars[k][c]: Variables that represent if sample k is assigned class c
        z_vars = [[model.NewBoolVar('z_%d_%d'%(k,c)) for c in range(C) ] for k in range(N) ]

        fixedzidx = 0
        for c in range(C):
            mincz = math.floor( maxcards[c] / maxbval)
            for offset in range(mincz):
                model.Add( z_vars[fixedzidx+offset][c] == 1)
            fixedzidx += mincz

        # q_vars[t][v][c][b] Variables that represent how many samples are used b times in node v of tree t as class c
        q_vars = [[] for t in range(ntrees) ]

        # obj_vars[t][b]: Variables that will capture the difference between sum_{k} q_{tkb}  - N * p_b, for fixed t and b
        obj_vars = [ [ model.NewIntVar(-N,N, 'obj_%d_%d' % (t,b) ) for b in range(maxbval) ] for t in range(ntrees) ]

        # abs_obj_vars[t][b]: Variables that will capture the absolute value of obj_vars for fixed t and b
        abs_obj_vars = [ [ model.NewIntVar(0,N, 'absobj_%d_%d' % (t,b) ) for b in range(maxbval) ] for t in range(ntrees) ]



        # Contraints
        # one-hot encoding
        for k in range(N):
            for w in range(
                    len(one_hot_encoded_groups)):  # for each group of binary attributes one-hot encoding the same attribute
                model.Add(cp_model.LinearExpr.Sum([x_vars[k][i] for i in one_hot_encoded_groups[w]]) == 1)

            # Enforces that every sample must be in at most one class
            model.Add( cp_model.LinearExpr.Sum( z_vars[k] ) == 1)

        nleaves = []
        for tid, all_branches_t in enumerate(trees_branches):  # for each tree
            nleaves.append( len(all_branches_t) )
            for a_branch_nb, a_branch in enumerate(all_branches_t):  # iterate over its branches

                # Variables that will be involved in making sure number of samples in each leaf/branch is consistent
                branch_vars_c = [[] for c in range(C)]
                coeff_branch_vars_c = [[] for c in range(C) ]

                q_vars[tid].append( [[] for c in range(C) ] )
                for c in range(C):
                    for b in range(maxbval):
                        q_vars[tid][a_branch_nb][c].append( model.NewIntVar(0, N, 'q_%d_%d_%d_%d' % (tid, a_branch_nb, c, b)))

                        branch_vars_c[c].append( q_vars[tid][a_branch_nb][c][b] )
                        coeff_branch_vars_c[c].append( b )


                for k in range(N):
                    # literals is used to construct the constraint that if k is used in a given tree at a given leaf/branch
                    # Then the x's must respect the corresponding branch
                    literals = []
                    for a_split in a_branch[0]:
                        if a_split > 0:
                            literals.append(x_vars[k][abs(a_split) - 1])
                        elif a_split < 0:
                            literals.append(x_vars[k][abs(a_split) - 1].Not())
                        else:
                            raise ValueError("Feat 0 shouldn't be used here (1-indexed now)")

                    for c in range(C):
                        # Variable y represents how many times sample k is used in tree tid, node a_branch_nb,
                        #    being that k is classified as class c
                        y_vars[k][c][tid].append( model.NewBoolVar('y_%d_%d_%d_%d' % (tid, a_branch_nb, k, c)) )

                        # Constraint that enforce consistency between w and x variables
                        model.AddBoolAnd(literals).OnlyEnforceIf(y_vars[k][c][tid][a_branch_nb])


                        # Constraints that enforce consistency between y and z variables
                        model.Add( y_vars[k][c][tid][a_branch_nb] == 0 ).OnlyEnforceIf( z_vars[k][c].Not() )


                for c in range(C):
                    model.Add(cp_model.LinearExpr.WeightedSum(branch_vars_c[c], coeff_branch_vars_c[c] ) == int(
                        a_branch[1][c]))  # enforces the branch per-class cardinality

                    # If samples are being used multiple times, there must be at least enough samples being used at least once
                    model.Add( cp_model.LinearExpr.Sum( [q_vars[tid][a_branch_nb][c][b] for b in range(maxbval) ]  ) <= cp_model.LinearExpr.Sum( [y_vars[k][c][tid][a_branch_nb] for k in range(N)] ) )

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
                    model.Add( cp_model.LinearExpr.Sum( [ q_vars[tid][v][c][bp] for v in range(nleaves[tid]) for c in range(C) for bp in range(b,maxbval) ] ) <= cnt )


        objfun = []
        for t in range(ntrees):
            for b in range(maxbval):
                # Set obj value abs constraints
                model.AddAbsEquality( abs_obj_vars[t][b], obj_vars[t][b] )

                #Set relationship between obj_vars and q_vars
                model.Add( cp_model.LinearExpr.Sum( [q_vars[t][v][c][bp] for v in range(nleaves[t]) for c in range(C) for bp in range(b,maxbval) ] ) - int( N * P[b] ) == obj_vars[t][b] )

                objfun.append( abs_obj_vars[t][b] )

        model.Minimize( cp_model.LinearExpr.Sum(objfun) )


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

            # GET and print solution (For debugging mostly)
            debug = 1
            if debug :
                y_sol = [[[[] for t in range(ntrees)] for c in range(C) ] for k in range(N) ]

                y_sample_cnt = [[[] for t in range(ntrees)] for c in range(C) ]

                # z_vars[k][c]: Variables that represent if sample k is assigned class c
                z_sol = [[solver.Value(z_vars[k][c]) for c in range(C)] for k in range(N)]

                # q_vars[t][v][c][b] Variables that represent how many samples are used b times in node v of tree t as class c
                q_sol = [[] for t in range(ntrees) ]

                count_samples_fromy = [ 0 for t in range(ntrees) ]
                count_bvals = [ [ 0 for b in range(maxbval)] for t in range(ntrees) ]

                for t in range(ntrees):
                    for v in range( len(q_vars[t]) ):
                        q_sol[t].append( [ [solver.Value(q_vars[t][v][c][b]) for b in range(maxbval) ] for c in range(C) ] )

                for t in range(ntrees):
                    for c in range(C):
                        for v in range(len(y_vars[k][c][t])):
                            y_sample_cnt[c][t].append(0)


                for k in range(N):
                    for t in range(ntrees):
                        for c in range(C):
                            y_sol[k][c][t] = [ solver.Value(y_vars[k][c][t][v]) for v in range(len(y_vars[k][c][t]))  ]

                            for v in range(len(y_vars[k][c][t])):
                                if y_sol[k][c][t][v] > 0 :
                                    y_sample_cnt[c][t][v] += 1

                # y_sample_cnt will be used to make sure q's are ok
                for t in range(ntrees):
                    for v in range(len(q_vars[t])):
                        for c in range(C):
                            for b in range(maxbval):
                                if q_sol[t][v][c][b] > 0:
                                    assert( y_sample_cnt[c][t][v] >= q_sol[t][v][c][b] )
                                    y_sample_cnt[c][t][v] -= q_sol[t][v][c][b]
                                    count_bvals[t][b] += q_sol[t][v][c][b]




                for k in range(N):
                    print("SOL: SAMPLE %d  :  "%(k)  + str( x_sol[k] ) )
                    kc = -1
                    for c in range(C):
                        if z_sol[k][c] == 1:
                            # Check that class has not changed
                            assert( kc == -1 )
                            kc = c
                            print("   - assigned class %d" %(c))

                    # Check that it has been assigned some class c
                    assert( kc != -1 )

                    for t in range(ntrees):
                        for v in range(len(y_sol[k][kc][t])):
                            if y_sol[k][kc][t][v] > 0:
                                count_samples_fromy[t] += y_sol[k][kc][t][v]
                                print("   -  at node %d of tree %d: %d times"%(v,t,y_sol[k][kc][t][v]))

                ## Tree view
                print('------ Tree view -------')
                for t in range(ntrees):
                    for v in range(nleaves[t]):
                        print( '  Samples at node %d of tree %d'%(v,t) )
                        for c in range(C):
                            for k in range(N):
                                if y_sol[k][c][t][v] > 0:
                                    print("     Sample %d of class %d"%(k,c))
                            for b in range(maxbval):
                                if q_sol[t][v][c][b] > 0:
                                    print("     Total %d samples appear %d times of class %d"%(q_sol[t][v][c][b],b,c))



                print( count_samples_fromy )

                for t in range(ntrees):
                    print("Distribution of samples for tree %d is:" % t + str( [ count_bvals[t][b] / N for b in range(maxbval) ]))
                    print(" Expected (values of exact p) were:" + str(Pexact))
        else:
            if status == cp_model.INFEASIBLE or status == cp_model.MODEL_INVALID:
                raise RuntimeError(
                    'Infeasible model: the reconstruction problem has no solution. Please make sure the provided one-hot encoding constraints are correct. Else, report this issue to the developers.')
            else:
                x_sol = np.random.randint(2, size=(N, M))


        print("*************************************************************")
        print("*************************************************************")
        print("  Solver specific:  Objval = %d,  duration = %g " % (obj_val, duration))
        print("*************************************************************")
        print("*************************************************************")

        solve_status = {0: 'UNKNOWN',
                        1: 'MODEL_INVALID',
                        2: 'FEASIBLE',
                        3: 'INFEASIBLE',
                        4: 'OPTIMAL'}[status]

        self.result_dict = {'max_max_depth': max_max_depth, 'status': solve_status, 'duration': duration,
                            'reconstructed_data': x_sol}

    # Function created to try and figure out what is the best performance that can be expected
    # For this, we assume that one knows ALL samples in advance
    # This will compute first the maximum likelihood solution (with bagging) assuming all data is given
    # Then tries to see how much we can modify the data and still get something that conforms with all the parameters
    def perform_reconstruction_benchmark(self, x_train, y_train, n_threads=0, time_out=60, verbosity=1, seed=0):
        """
        Runs the complementary experiments on the impact of bagging on data protection, mentionned in the Appendix B of our paper. 
        The model builds upon the CP based dataset reconstruction model (with the use of bagging to train the target random forest) 
        using the OR-Tools CP-SAT solver, but first pre-computes the optimal assignments of the examples' occurences within each
        tree's training set (using the actual forest's training set) before computing how bad the reconstruction could be at worst 
        given these correct occurences assignements.

        Arguments
        ---------
        x_train: array-like, shape = [n_known, n_features] (default [])
                        the actual training set of the forest 
                        (used only to compute the optimal #occurences of the examples within the trees' training sets)
                        with n_known = N
                        and n_features = M

        y_train: array-like, shape = [n_known] (default [])
                        the labels of the training set examples 
                        with n_known = N

        n_threads: int >= 0, optional (default 0)
                        maximum number of threads to be used by the solver to parallelize search
                        if 0, use all available threads

        time_out: int, optional (default 60)
                        maximum cpu time (in seconds) to be used by the search
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
            -> 'status': the solve status returned by the solver. It can be 'UNKNOWN', 'MODEL_INVALID', 'FEASIBLE', 'INFEASIBLE', or 'OPTIMAL'.
            -> 'duration': duration
            -> 'reconstructed_data': array of shape = [n_samples, n_attributes] encoding the reconstructed dataset.
        """
        from ortools.sat.python import cp_model
        import numpy as np  # useful
        import time  # time measurements


        # This is the maximum number of times a sample can appear in a tree (note it will go from 0 to maxbval-1)
        maxbval = 8

        clf = self.clf
        one_hot_encoded_groups = self.ohe_groups

        start = time.time()

        ### Create the CP model

        ## Parse the forest
        T, M, N, C, Z, max_max_depth, trees_branches, maxcards = self.parse_forest(clf, verbosity=verbosity)

        # Defines the probabilities that an item will appear b times
        P = []
        Pexact = [0 for i in range(maxbval)]
        for i in range(maxbval):
            #P.append( 1 - self.proba_inf(i + 1, N) )
            P.append(1 - self.proba_inf(i , N))
        for i in range(maxbval):
            if i < maxbval - 1:
                Pexact[i] = P[i] - P[i+1]
            else:
                Pexact[i] = P[i]



        if verbosity:
            print("Probabilities of an item appearing at least b times:")
            print(P)
            print(sum(P))

            print("Probabilities of an item appearing at EXACTLY b times:")
            print(Pexact)
            print(sum(Pexact))

        ntrees = len( trees_branches )

        ## Variables
        model = cp_model.CpModel()

        # x[k][i] : Variables that represent what is sample k (each of its features i)
        x_vars = [[model.NewBoolVar('x_%d_%d' % (k, i)) for i in range(M)] for k in range(N)]  # table of x_{ki}

        # y_vars[k][c][t][v]: Variables that represent the number of times sample k is used as class c
        #   in leaf/branch v of tree t
        y_vars = [[[[] for t in range(ntrees)] for c in range(C)] for k in range(N)]

        # w_vars[k][t][v]: Variables that represent if sample k is used in leaf v of tree t
        w_vars = [[[] for t in range(ntrees) ] for k in range(N)]

        # z_vars[k][c]: Variables that represent if sample k is assigned class c
        z_vars = [[model.NewBoolVar('z_%d_%d'%(k,c)) for c in range(C) ] for k in range(N) ]





        # eta_vars[k][t]: Variables that count how many times sample k is used in tree t
        eta_vars = [[ model.NewIntVar(0,maxbval, 'eta_%d_%d'%(k,t)) for t in range(ntrees)] for k in range(N) ]

        # q_vars[k][t][b] Variables that represent if sample k appears b times in tree t (needed for objective function
        q_vars = [[[model.NewBoolVar('q_%d_%d_%d' %( t, k, b )) for b in range(maxbval) ] for t in range(ntrees) ] for k in range(N) ]


        objfun = []
        objfuncoeff = []
        for t in range(ntrees):
            for b in range(maxbval):
                for k in range(N):
                    objfuncoeff.append( int( 10 * math.log(Pexact[b]) ) )
                    objfun.append( q_vars[k][t][b] )


        model.Maximize( cp_model.LinearExpr.WeightedSum( objfun, objfuncoeff ) )


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
                model.AddMapDomain( eta_vars[k][t], q_vars[k][t], offset=0 )


        nleaves = []

        # This stores the information of which variables are set to 1 and 0 in each branch
        branch_info = [[] for t in range(len(trees_branches))]

        for tid, all_branches_t in enumerate(trees_branches):  # for each tree

            etayvars = [[] for k in
                              range(N)]  # for each example we will ensure it is captured by exactly one branch

            nleaves.append( len(all_branches_t) )
            for a_branch_nb, a_branch in enumerate(all_branches_t):  # iterate over its branches

                binfo = []
                for a_split in a_branch[0]:
                    if a_split > 0:
                        binfo.append([abs(a_split) - 1, 1])
                    elif a_split < 0:
                        binfo.append([abs(a_split) - 1, 0])
                    else:
                        raise ValueError("Feat 0 shouldn't be used here (1-indexed now)")
                branch_info[tid].append(binfo)


                # Variables that will be involved in making sure number of samples in each leaf/branch is consistent
                branch_vars_c = [[] for c in range(C)]

                for k in range(N):
                    w_vars[k][tid].append( model.NewBoolVar('w_%d_%d_%d'%(k,tid,a_branch_nb)) )

                    # literals is used to construct the constraint that if k is used in a given tree at a given leaf/branch
                    # Then the x's must respect the corresponding branch
                    literals = []
                    for a_split in a_branch[0]:
                        if a_split > 0:
                            literals.append(x_vars[k][abs(a_split) - 1])
                        elif a_split < 0:
                            literals.append(x_vars[k][abs(a_split) - 1].Not())
                        else:
                            raise ValueError("Feat 0 shouldn't be used here (1-indexed now)")
                    # Constraint that enforce consistency between w and x variables
                    model.AddBoolAnd(literals).OnlyEnforceIf(w_vars[k][tid][a_branch_nb])


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

        # Assume knowledge of dataset
        for k in range(N):
            for w in range(len(one_hot_encoded_groups)):
                onehotsum = 0
                for i in one_hot_encoded_groups[w]:
                    onehotsum += x_train[k][i]
                if onehotsum != 1:
                    print( " ERROR: SAMPLE " + str(k) + " has onehotencoding total " + str(onehotsum) + " for onehot: " + str(one_hot_encoded_groups[w]))
                    exit(1)
            for i in range(M):
                model.Add( x_vars[k][i] == x_train[k][i] )
            for c in range(C):
                if y_train[k] == c:
                    model.Add(z_vars[k][c] == 1)




        if verbosity:
            print("Model creation done!")

#        for t in range(ntrees):
#            print(" TREE INFO for tree %d" % t)
#            for v in range( len( branch_info[t] ) ):
#                print("   LEAF %d"%v)
#                print(branch_info[t][v])

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

        print(" --- BENCHMARK RUN  ---  %g seconds" % (duration) )

        # Récupération statut/valeurs
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            # Start with solution value of -1, to say it has not been fixed yet
            x_sol = [[-1 for i in range(M)] for k in range(N)]

            obj_val = solver.ObjectiveValue()

            y_sol = [[[[] for t in range(ntrees)] for c in range(C) ] for k in range(N) ]

            # w_vars[k][t][v]: Variables that represent if sample k is used in leaf v of tree t
            w_sol = [[[] for t in range(ntrees)] for k in range(N)]

            # z_vars[k][c]: Variables that represent if sample k is assigned class c
            z_sol = [[solver.Value(z_vars[k][c]) for c in range(C)] for k in range(N)]

            # eta_vars[k][t]: Variables that count how many times sample k is used in tree t
            eta_sol = [[solver.Value( eta_vars[k][t] ) for t in range(ntrees)] for k in range(N)]

            # q_vars[k][t][b] Variables that represent if sample k appears b times in tree t (needed for objective function
            q_sol = [[[ solver.Value( q_vars[k][t][b]) for b in range(maxbval)] for t in range(ntrees)] for k in range(N)]

            count_samples = [ 0 for t in range(ntrees) ]
            count_samples_fromy = [ 0 for t in range(ntrees) ]
            count_bvals = [ [0 for b in range(maxbval)] for t in range(ntrees) ]

            for k in range(N):
                for t in range(ntrees):
                    w_sol[k][t] = [ solver.Value(w_vars[k][t][v]) for v in range(len(w_vars[k][t])) ]
                    for c in range(C):
                        y_sol[k][c][t] = [ solver.Value(y_vars[k][c][t][v]) for v in range(len(y_vars[k][c][t]))  ]


            for k in range(N):
                if verbosity:
                    print("SOL: SAMPLE %d  :  "%(k)  + str( x_sol[k] ) )
                kc = -1
                for c in range(C):
                    if z_sol[k][c] == 1:
                        # Check that class has not changed
                        assert( kc == -1 )
                        kc = c
                        if verbosity:
                            print("   - assigned class %d" %(c))

                # Check that it has been assigned some class c
                assert( kc != -1 )

                for t in range(ntrees):
                    if eta_sol[k][t] > 0:
                        count_samples[t] += eta_sol[k][t]
                        if verbosity:
                            print("   - used %d times in tree %d"%(eta_sol[k][t],t))

                    for b in range(maxbval):
                        if int(eta_sol[k][t]) == b:
                            assert( q_sol[k][t][b] == 1 )
                            count_bvals[t][b] += 1
                        else:
                            assert( q_sol[k][t][b] == 0 )

                    for v in range(len(y_sol[k][kc][t])):
                        if y_sol[k][kc][t][v] > 0:
                            assert( w_sol[k][t][v] > 0 )
                            count_samples_fromy[t] += y_sol[k][kc][t][v]
                            if verbosity:
                                print("   -  at node %d: %d times"%(v,y_sol[k][kc][t][v]))

                            assert( kc == y_train[k] )
                            for f in range( len( branch_info[t][v] ) ):
                                idx = branch_info[t][v][f][0]
                                val = branch_info[t][v][f][1]
                                if x_sol[k][idx] == -1:
                                    x_sol[k][idx] = val
                                else:
                                    assert( x_sol[k][idx] == val )
                print( " Solution %d (what is known) " % k)
                print(x_sol[k])
            ## Tree view
            if verbosity:
                print('------ Tree view -------')
            for t in range(ntrees):
                for v in range(nleaves[t]):
                    if verbosity:
                        print('  Samples at node %d of tree %d' % (v, t))
                    for c in range(C):
                        for k in range(N):
                            if y_sol[k][c][t][v] > 0:
                                if verbosity:
                                    print("     Sample %d of class %d appears %d times" % (k, c, y_sol[k][c][t][v]))

            if verbosity:
                print( count_samples )
                print( count_samples_fromy )
            for t in range(ntrees):
                if verbosity:
                    print( "Distribution of samples for tree %d is:"%t + str( [ count_bvals[t][b] / N for b in range(maxbval) ]))
                    print( " Expected (values of exact p) were:" + str(Pexact) )

            fixedcoords = [ 0 for k in range(N) ]
            for k in range(N):
                for i in range(M):
                    if x_sol[k][i] == -1:
                        x_sol[k][i] = 1 - x_train[k][i]
                    else:
                        fixedcoords[k] += 1
            print( " Total fixed coords: " + str( sum(fixedcoords) ) + "  out of " + str( N*M ) + " = (percent) " + str( sum(fixedcoords) / (N*M) ) )
        else:
            if status == cp_model.INFEASIBLE or status == cp_model.MODEL_INVALID:
                raise RuntimeError(
                    'Infeasible model: the reconstruction problem has no solution. Please make sure the provided one-hot encoding constraints are correct. Else, report this issue to the developers.')
            else:
                x_sol = np.random.randint(2, size=(N, M))


        print("*************************************************************")
        print("*************************************************************")
        print("  Solver specific:  Objval = %d,  duration = %g " % (obj_val, duration))
        print("*************************************************************")
        print("*************************************************************")

        solve_status = {0: 'UNKNOWN',
                        1: 'MODEL_INVALID',
                        2: 'FEASIBLE',
                        3: 'INFEASIBLE',
                        4: 'OPTIMAL'}[status]

        result_dict = {'max_max_depth': max_max_depth, 'status': solve_status, 'duration': duration,
                            'reconstructed_data': x_sol}

        return result_dict


