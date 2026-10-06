"""Reconstruction of training data from supported DP random forests."""

from ortools.sat.python import cp_model
import numpy as np
from scipy.stats import t as student_t
from _validation import validate_solver_options
import time
import ortools
import copy
from scipy import integrate as intg

class DRAFT_DP:

    def __init__(self, random_forest, epsilon, one_hot_encoded_groups=None):
        from DP_RF import DP_RF
        from sklearn.exceptions import NotFittedError
        if not isinstance(random_forest, DP_RF):
            raise TypeError("DRAFT_DP expects a DP_RF trained with the supported noise mechanism.")
        if not hasattr(random_forest, "estimators_"):
            raise NotFittedError("Train DP_RF with fit before constructing DRAFT_DP.")
        clf, eps = random_forest, epsilon
        if not np.isfinite(eps) or eps <= 0:
            raise ValueError("epsilon must be finite and strictly positive.")
        self.clf = clf
        self.estimators = clf.estimators_
        self.eps = eps
        self.eps_v = eps / clf.N_trees
        self.result_dict = {}
        self.ohe_groups = [] if one_hot_encoded_groups is None else copy.deepcopy(one_hot_encoded_groups)

    def _interval_N(self):
        nb = self._format_nb()
        N_avg = round(sum((nb[t][v][c] for t in range(self.clf.N_trees) for v in range(len(nb[0])) for c in range(self.clf.n_classes_))) / self.clf.N_trees)
        quantile = student_t.ppf(0.95, self.clf.N_trees - 1) if self.clf.N_trees > 1 else 6.314
        N_leaves = 2 ** self.clf.estimators_[0].tree_.max_depth
        N_classes = self.clf.n_classes_
        std = np.sqrt(2.0 * N_leaves * N_classes) / self.eps_v
        bound_inf = int(N_avg - max(quantile, 1) * std)
        bound_sup = int(N_avg + max(quantile, 1) * std)
        return (N_avg, bound_inf, bound_sup)

    def _log_liste_laplace(self):
        bound = round(np.ceil(12 / self.eps_v))
        res = [0] * (bound + 1)
        res[0] = np.log(self._proba_laplace(-1, 1)[0])
        for i in range(1, bound + 1):
            res[i] = np.log(2 * self._proba_laplace(i, i + 1)[0])
        return res

    def _proba_laplace(self, a, b):

        def f(x):
            return self.eps_v / 2 * np.exp(-self.eps_v * np.abs(x))
        return intg.quad(f, a, b)

    def _format_nb(self):
        num_leaves = self._get_numeros_leaves(self.clf.estimators_[0].tree_)
        nb_b = [[[round(t.tree_.value[v][0][c]) for c in range(self.clf.n_classes_)] for v in num_leaves] for t in self.clf.estimators_]
        return nb_b

    def _get_numeros_leaves(self, tree):
        leaves = []

        def browse_nodes(nodes):
            if tree.children_left[nodes] == tree.children_right[nodes]:
                leaves.append(nodes)
            else:
                if tree.children_left[nodes] != -1:
                    browse_nodes(tree.children_left[nodes])
                if tree.children_right[nodes] != -1:
                    browse_nodes(tree.children_right[nodes])
        browse_nodes(0)
        return leaves

    def _parse_forest(self, clf, verbosity=False):
        """Read raw noisy counts; DP_RF stores counts even on sklearn >=1.4."""
        trees_branches = []
        for estimator in clf.estimators_:
            tree = estimator.tree_
            branches = []
            def visit(node, path):
                if tree.children_left[node] == tree.children_right[node]:
                    branches.append((path, tree.value[node][0].tolist()))
                    return
                if not 0 <= tree.threshold[node] < 1:
                    raise ValueError("DRAFT_DP supports binary features only.")
                feature = int(tree.feature[node]) + 1
                visit(tree.children_left[node], path + [-feature])
                visit(tree.children_right[node], path + [feature])
            visit(0, [])
            if len(branches) != 2 ** tree.max_depth:
                raise ValueError("DRAFT_DP expects complete random trees from DP_RF.")
            trees_branches.append(branches)
        if verbosity:
            print("Parsing done")
        return trees_branches

    def fit(self, n_samples=None, *, timeout=60, verbosity=False, n_threads=-1, seed=0, max_samples=400,
            X_known=None, y_known=None, target_ratio=None):
        """Reconstruct binary data with known or inferred training-set size.

        n_samples: known row count, or None to estimate it (bounded by max_samples).
        timeout: solver search time limit in seconds, excluding model construction.
        X_known / y_known: opt-in informed reconstruction with N-1 fully known
        examples and labels. Exactly one missing row is reconstructed and appended.
        n_samples may be omitted in this mode (N = len(X_known) + 1).
        missing_example: target feature vector, or None without a solution,
        added to the result only in informed mode.
        target_ratio: likelihood/proximity range ratio (default epsilon / 2).
        Returns status, duration, reconstructed_data (None without a solution),
        max_max_depth, plus DP count and sample-size diagnostics.
        """
        validate_solver_options(timeout, n_threads, seed)
        for name, value in (("n_samples", n_samples), ("max_samples", max_samples)):
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0):
                raise ValueError(f"{name} must be a positive integer.")
        if max_samples is None:
            raise ValueError("max_samples must be a positive integer.")
        informed = X_known is not None or y_known is not None
        # Internal row layout: known examples first, one unknown example last.
        X_partial_expe = y_partial_expe = ex_id = None
        if informed:
            if X_known is None or y_known is None:
                raise ValueError("Supply X_known and y_known together.")
            known_X = np.asarray(X_known)
            known_y = np.asarray(y_known)
            n_features = self.estimators[0].n_features_in_
            if known_X.ndim != 2 or known_X.shape[1] != n_features:
                raise ValueError("X_known must be a 2-D array with the forest's feature count.")
            if known_y.shape != (len(known_X),):
                raise ValueError("y_known must contain one label per known example.")
            expected_n = len(known_X) + 1
            if n_samples is None:
                n_samples = expected_n
            elif n_samples != expected_n:
                raise ValueError("Informed reconstruction supports exactly one missing example: n_samples must equal len(X_known) + 1.")
            if not np.isin(known_X, [0, 1]).all():
                raise ValueError("Known examples must contain binary features.")
            if not np.isin(known_y, range(self.clf.n_classes_)).all():
                raise ValueError("Known labels must be valid integer class indices.")
            ex_id = len(known_X)
            X_partial_expe = np.concatenate((known_X.astype(int), np.zeros((1, n_features), dtype=int)))
            y_partial_expe = np.concatenate((known_y.astype(int), np.zeros(1, dtype=int)))
            if target_ratio is not None and (isinstance(target_ratio, bool) or not isinstance(target_ratio, (int, float, np.number)) or not np.isfinite(target_ratio) or target_ratio <= 0):
                raise ValueError("target_ratio must be a finite positive number.")
        elif target_ratio is not None:
            raise ValueError("target_ratio applies only to informed reconstruction.")

        if n_threads == -1:
            n_threads = 0
        start = time.time()
        model = cp_model.CpModel()
        if verbosity:
            print('Constructing CP model.')
        nb_noise = self._format_nb()
        card_c = self.clf.n_classes_
        N_trees = self.clf.n_estimators
        N_leaves = 2 ** self.clf.estimators_[0].tree_.max_depth
        bound = round(np.ceil(12 / self.eps_v))
        p = self._log_liste_laplace()
        one_hot_encoded_groups = self.ohe_groups
        M = self.clf.estimators_[0].n_features_in_
        trees_branches = self._parse_forest(self.clf, verbosity=verbosity)
        N_min = 0
        N_max = 0
        if n_samples is None:
            N_avg, N_min, N_max = self._interval_N()
            N_min = max([1, N_min])
            N_max = min([N_max, max_samples])
            if N_min > N_max:
                raise ValueError("Estimated size exceeds max_samples; increase max_samples or supply n_samples.")
            if N_avg > max_samples:
                print('Warning: N_avg is quite large (%d) compared to max_samples (%d). Consider increasing max_samples if no solution is found.' % (N_avg, max_samples))
            if verbosity:
                print('N_avg', N_avg, 'N_max :', N_max, 'N_min :', N_min)
            N = model.NewIntVar(N_min, N_max, 'N')
            nb = [[[model.NewIntVar(0, N_max, 'nb_%d' % t) for c in range(card_c)] for v in range(N_leaves)] for t in range(N_trees)]
            delta = [[[model.NewIntVar(-bound, bound, 'delta_%d' % t) for c in range(card_c)] for v in range(N_leaves)] for t in range(N_trees)]
            liste_p = []
            liste_bool = []
            x = [[model.NewBoolVar('x_%d_%d' % (k, j)) for j in range(M)] for k in range(N_max)]
            y = [[[[model.NewBoolVar('y') for c in range(card_c)] for k in range(N_max)] for v in range(N_leaves)] for t in range(N_trees)]
            z = [[model.NewBoolVar('Z_%d_%d' % (i, c)) for c in range(card_c)] for i in range(N_max)]
            delta_bool_list = [[[[model.NewBoolVar('delta_val') for i in range(0, bound + 1)] for c in range(card_c)] for v in range(N_leaves)] for t in range(N_trees)]
            abs_delta = [[[model.NewIntVar(0, bound, 'delta') for c in range(card_c)] for v in range(N_leaves)] for t in range(N_trees)]
            for t in range(N_trees):
                model.Add(sum((nb[t][v][c] for v in range(N_leaves) for c in range(card_c))) == N)
                for c in range(card_c):
                    for v in range(N_leaves):
                        model.Add(delta[t][v][c] == nb_noise[t][v][c] - nb[t][v][c])
                        model.AddAbsEquality(abs_delta[t][v][c], delta[t][v][c])
                        delta_bool_constraint = []
                        ortools_version = str(ortools.__version__).split('.')
                        if int(ortools_version[0]) <= 9 and int(ortools_version[1]) <= 8:
                            model.AddMapDomain(abs_delta[t][v][c], delta_bool_list[t][v][c], offset=0)
                        else:
                            model.add_map_domain(abs_delta[t][v][c], delta_bool_list[t][v][c], offset=0)
                        delta_bool_constraint.extend(delta_bool_list[t][v][c])
                        liste_p.extend(p)
                        liste_bool.extend(delta_bool_constraint)
                        model.Add(nb[t][v][c] >= 0)
            for k in range(N_max):
                model.Add(sum((z[k][c] for c in range(card_c))) == 1)
            for k in range(N_max):
                for c in range(card_c):
                    model.Add(sum((y[t][v][k][c] for t in range(N_trees) for v in range(N_leaves))) == 0).OnlyEnforceIf(z[k][c].Not())
            ex_k_not_classified_by_leaf_v_in_tree_t = [[[model.NewBoolVar('ex_k_not_classified_by_leaf_v_in_tree_t') for k in range(N_max)] for v in range(N_leaves)] for t in range(N_trees)]
            for idx_tree, liste_branches in enumerate(trees_branches):
                for idx_branch, branche in enumerate(liste_branches):
                    for k in range(N_max):
                        for feature in branche[0]:
                            model.Add(cp_model.LinearExpr.Sum(y[idx_tree][idx_branch][k]) == 0).OnlyEnforceIf(ex_k_not_classified_by_leaf_v_in_tree_t[idx_tree][idx_branch][k])
                            if feature > 0:
                                model.Add(x[k][abs(feature) - 1] == 1).OnlyEnforceIf(ex_k_not_classified_by_leaf_v_in_tree_t[idx_tree][idx_branch][k].Not())
                            if feature < 0:
                                model.Add(x[k][abs(feature) - 1] == 0).OnlyEnforceIf(ex_k_not_classified_by_leaf_v_in_tree_t[idx_tree][idx_branch][k].Not())
            active = [model.NewBoolVar(f"active_{k}") for k in range(N_max)]
            for k in range(N_max):
                model.Add(N > k).OnlyEnforceIf(active[k])
                model.Add(N <= k).OnlyEnforceIf(active[k].Not())
                for t in range(N_trees):
                    model.Add(sum(y[t][v][k][c] for v in range(N_leaves)
                                  for c in range(card_c)) == active[k])

            for t in range(N_trees):
                for v in range(N_leaves):
                    for c in range(card_c):
                        model.Add(sum((y[t][v][k][c] for k in range(N_max))) == nb[t][v][c])
            for k in range(N_max):
                for w in range(len(one_hot_encoded_groups)):
                    model.Add(cp_model.LinearExpr.Sum([x[k][i] for i in one_hot_encoded_groups[w]]) == 1)
        else:
            if verbosity:
                print("For n_samples.")

            N = n_samples


            informed = (X_partial_expe is not None and
            y_partial_expe is not None and
            ex_id is not None) # Whether we are in the informed adversary experiment setting

            if informed:
                assert(X_partial_expe.shape[1] == M)
                assert(X_partial_expe.shape[0] == N)
                assert(y_partial_expe.shape[0] == N)
                assert(0 <= ex_id and ex_id < N)

                # Map sklearn leaf node id -> position v used in nb_noise / nb / delta
                num_leaves = self._get_numeros_leaves(self.clf.estimators_[0].tree_)
                leaf_pos = {leaf_id: idx for idx, leaf_id in enumerate(num_leaves)}
                assert len(num_leaves) == N_leaves

                # known_leaf_counts[t][v][c] = number of known examples in leaf v, class c
                known_leaf_counts = [[[0 for _ in range(card_c)]
                                    for _ in range(N_leaves)]
                                    for _ in range(N_trees)]

                for t, est in enumerate(self.clf.estimators_):
                    leaf_ids = est.apply(X_partial_expe)  # shape (N,)
                    for k in range(N):
                        if k == ex_id:
                            continue  # unknown example, handled by y-vars
                        v_leaf_id = leaf_ids[k]
                        v = leaf_pos[v_leaf_id]
                        c = int(y_partial_expe[k])
                        known_leaf_counts[t][v][c] += 1
            else:
                known_leaf_counts = None

            # Variables definitions
            nb = [[[model.NewIntVar(0, N, 'nb') for c in range(card_c)] for v in range(N_leaves)] for t in range(N_trees)] #'nb_%d_%d_%d' % (t, v, c))
            delta = [[[model.NewIntVar(-bound, bound, 'delta') for c in range(card_c)] for v in range(N_leaves)] for t in range(N_trees)] #'delta_%d_%d_%d' % (t, v, c)

            liste_p = []
            liste_bool = []

            x = [[model.NewBoolVar('x') for j in range(M)] for k in range(N)] #'x_%d_%d' % (k,j)
            if informed:
                # Only keep y-vars for the unknown example ex_id
                y = [[[[None for c in range(card_c)] for k in range(N)]
                    for v in range(N_leaves)]
                    for t in range(N_trees)]
                for t in range(N_trees):
                    for v in range(N_leaves):
                        for c in range(card_c):
                            y[t][v][ex_id][c] = model.NewBoolVar('y')
            else:
                y = [[[[model.NewBoolVar('y') for c in range(card_c)] for k in range(N)]
                    for v in range(N_leaves)]
                    for t in range(N_trees)]
            z = [[model.NewBoolVar('Z') for c in range(card_c)] for i in range(N)] #'Z_%d_%d' % (i, c)

            # Assume knowledge of part of the dataset's examples (informed adversary experiment)
            if informed:
                for k in range(N): # fix all examples but one
                    if k != ex_id: # ex_id
                        for j in range(M):
                            model.Add( x[k][j] == X_partial_expe[k][j] )
                        for c in range(card_c):
                            if y_partial_expe[k] == c:
                                model.Add(z[k][c] == 1)
                            else:
                                model.Add(z[k][c] == 0)
            # ----------------------------------------------------------------------------------------------

            delta_bool_list = [[[[model.NewBoolVar('delta_val') for i in range(0, bound+1)] for c in range(card_c)] for v in range(N_leaves)] for t in range(N_trees)] # f'delta_val_{i}_{t}_{v}_{c}'
            abs_delta = [[[model.NewIntVar(0, bound, 'abs_delta') for c in range(card_c)] for v in range(N_leaves)] for t in range(N_trees)] # 'delta_%d_%d_%d' % (t, v, c)

            if verbosity:
                print("Created variables.")

            for t in range(N_trees):
                #Constraint ensuring that all trees have N training examples
                model.Add(sum(nb[t][v][c] for v in range(N_leaves) for c in range(card_c)) == N)

                for c in range(card_c):
                    for v in range(N_leaves):
                        # Constraint that computes the discrepancies between the noised value and the estimated count values
                        model.Add(delta[t][v][c] == nb_noise[t][v][c]-nb[t][v][c])
                        model.AddAbsEquality(abs_delta[t][v][c], delta[t][v][c])

                        # Constraint defining bool_proba and delta_val as a function of delta[t][v][c]
                        delta_bool_constraint = []
                        ortools_version = str(ortools.__version__).split(".")
                        if int(ortools_version[0]) <= 9 and int(ortools_version[1]) <= 8:
                            model.AddMapDomain(abs_delta[t][v][c], delta_bool_list[t][v][c], offset = 0)
                        else:
                            model.add_map_domain(abs_delta[t][v][c], delta_bool_list[t][v][c], offset = 0)

                        delta_bool_constraint.extend(delta_bool_list[t][v][c])
                        liste_p.extend(p)
                        liste_bool.extend(delta_bool_constraint)

                        # Constraint ensuring that nb[t][v][c] is positif or null
                        model.Add(nb[t][v][c] >= 0)

            if verbosity:
                print("Created tree constraints.")

            # Each example is assigned to only one class
            for k in range(N):
                model.Add(sum(z[k][c] for c in range(card_c)) == 1)

            # An example appears only in the counts of its class
            #for k in range(N):
            #    for c in range(card_c):
            #        model.Add(sum(y[t][v][k][c] for t in range(N_trees) for v in range(N_leaves)) == 0).OnlyEnforceIf(z[k][c].Not())

            for t in range(N_trees):
                for k in range(N):
                    for c in range(card_c):
                        if informed and k != ex_id:
                            # For known examples, z[k][c] is already fixed from y_partial_expe
                            continue
                        model.Add(sum(y[t][v][k][c] for v in range(N_leaves)) == z[k][c])


            if verbosity:
                print("Created other constraints.")


            #The values of the features align with the splits of the branch
            if informed:
                # Only allocate for the unknown example
                ex_k_not_classified_by_leaf_v_in_tree_t = [[[None for _ in range(N)]
                                                            for _ in range(N_leaves)]
                                                        for _ in range(N_trees)]
                for t in range(N_trees):
                    for v in range(N_leaves):
                        ex_k_not_classified_by_leaf_v_in_tree_t[t][v][ex_id] = model.NewBoolVar('ex_k_not_classified')
            else:
                ex_k_not_classified_by_leaf_v_in_tree_t = [[[model.NewBoolVar('ex_k_not_classified') for _ in range(N)]
                                                            for _ in range(N_leaves)]
                                                        for _ in range(N_trees)]

            for idx_tree, liste_branches in enumerate(trees_branches):
                for idx_branch, branche in enumerate(liste_branches):
                    #print("idx_branch :", idx_branch, "branche :", branche)
                    for k in range(N):
                        if informed and k != ex_id:
                            # Known examples have fixed features; we don't need per-branch boolean machinery for them
                            continue
                        for feature in branche[0]:
                            model.Add(cp_model.LinearExpr.Sum(y[idx_tree][idx_branch][k]) == 0).OnlyEnforceIf(ex_k_not_classified_by_leaf_v_in_tree_t[idx_tree][idx_branch][k])
                            if feature > 0:
                                model.Add(x[k][abs(feature)-1] == 1).OnlyEnforceIf(ex_k_not_classified_by_leaf_v_in_tree_t[idx_tree][idx_branch][k].Not())
                            if feature < 0:
                                model.Add(x[k][abs(feature)-1] == 0).OnlyEnforceIf(ex_k_not_classified_by_leaf_v_in_tree_t[idx_tree][idx_branch][k].Not())

            if verbosity:
                print("Created other constraints bis.")

            # The counts correspond to the number of assigned examples
            for t in range(N_trees):
                for v in range(N_leaves):
                    for c in range(card_c):
                        if informed:
                            # known_leaf_counts includes all k != ex_id
                            base = known_leaf_counts[t][v][c]
                            # unknown example contributes at most 1 here
                            # (if y[t][v][ex_id][c] is None, that means this class cannot occur here)
                            y_var = y[t][v][ex_id][c]
                            if y_var is not None:
                                model.Add(base + y_var == nb[t][v][c])
                            else:
                                # No variable for this (t,v,c) for ex_id: only known examples contribute
                                model.Add(nb[t][v][c] == base)
                        else:
                            model.Add(sum(y[t][v][k][c] for k in range(N)) == nb[t][v][c])

            if verbosity:
                print("Created other constraints ter.")

            #OHE Constraint
            for k in range(N):
                for w in range(len(one_hot_encoded_groups)): # for each group of binary attributes one-hot encoding the same attribute
                    model.Add(cp_model.LinearExpr.Sum([x[k][i] for i in one_hot_encoded_groups[w]]) == 1)

        if verbosity:
            print('Beginning search.')
        solver = cp_model.CpSolver()
        solver.parameters.log_search_progress = verbosity
        solver.parameters.max_time_in_seconds = timeout
        solver.parameters.num_workers = n_threads
        solver.parameters.random_seed = seed
        if informed:
            # For informed adversary experiment
            # Add a regularization term encouraging proximity of the reconstructed example to the others
            list_l1_bools = []
            list_l1_coeffs = []
            for k in range(N):
                if k != ex_id: # ex_id
                    for j in range(M):
                        if X_partial_expe[k][j] == 1:
                            list_l1_bools.append(x[ex_id][j])
                        elif X_partial_expe[k][j] == 0:
                            list_l1_bools.append(x[ex_id][j].Not())
                        else:
                            raise ValueError("X_partial_expe should be binary.")
                        list_l1_coeffs.append(1)

            n_leaves_total = len(liste_bool)/len(p)
            likelihood_obj_range = np.abs(max(liste_p)*n_leaves_total -min(liste_p)*n_leaves_total)
            proximity_obj_range = sum(list_l1_coeffs)
            if target_ratio is None:
                target_ratio = self.eps/2 # default target ratio
            if verbosity:
                print("Likelihood objective value range:", likelihood_obj_range)
                print("Proximity objective value range:", proximity_obj_range)
                # Ensure the ratio (Likelihood objective value range/ Proximity objective value range) is around epsilon
                # => For large epsilons, gives more importance to likelihood, for small epsilons, gives more importance to proximity
                print("Current ratio:", (likelihood_obj_range / proximity_obj_range if proximity_obj_range else float("inf")), "target ratio:", target_ratio)
            if proximity_obj_range > 0 and likelihood_obj_range > 0:
                if (likelihood_obj_range / proximity_obj_range) > target_ratio:
                    # Need to give more importance to proximity
                    if verbosity:
                        print("Scaling proximity objective...")
                    current_ratio = proximity_obj_range / likelihood_obj_range
                    scaling_factor = 1/(current_ratio * target_ratio)
                    current_ratio = likelihood_obj_range / proximity_obj_range
                    list_l1_coeffs = [np.round(val * scaling_factor) for val in list_l1_coeffs]
                else:
                    # Need to give more importance to likelihood
                    if verbosity:
                        print("Scaling likelihood objective...")
                    current_ratio = likelihood_obj_range / proximity_obj_range
                    scaling_factor = target_ratio / current_ratio
                    liste_p = [np.round(val * scaling_factor) for val in liste_p]
            likelihood_obj_range = np.abs(max(liste_p)*n_leaves_total -min(liste_p)*n_leaves_total)
            proximity_obj_range = sum(list_l1_coeffs)
            if verbosity:
                print("[SCALED] Likelihood objective value range:", likelihood_obj_range)
                print("[SCALED] Proximity objective value range:", proximity_obj_range)
                print("Updated ratio:", (likelihood_obj_range / proximity_obj_range if proximity_obj_range else float("inf")), "target ratio:", target_ratio)
            liste_bool.extend(list_l1_bools) # incorporate the proximity term in the objective
            liste_p.extend(list_l1_coeffs) # incorporate the proximity term in the objective
        model.Maximize(cp_model.LinearExpr.WeightedSum(liste_bool, liste_p))
        status = solver.Solve(model)
        end = time.time()
        duration = end - start
        if solver.StatusName(status) == 'OPTIMAL' or solver.StatusName(status) == 'FEASIBLE':
            if verbosity:
                print('Value of the maximized objective function:', solver.ObjectiveValue())
            N = solver.Value(N)
            if verbosity:
                print('N :', N)
            x = [[solver.Value(x[k][i]) for i in range(M)] for k in range(N)]
            z = [[solver.Value(z[k][c]) for c in range(card_c)] for k in range(N)]
            values_nb = [[[solver.Value(nb[t][v][c]) for c in range(card_c)] for v in range(N_leaves)] for t in range(N_trees)]
            self.result_dict = {'status': solver.StatusName(status), 'nb_recons': values_nb, 'duration': duration, 'reconstructed_data': x, 'N_min': N_min, 'N_max': N_max, 'N': N}

        else:
            self.result_dict = {'status': solver.StatusName(status), 'duration': duration, 'reconstructed_data': None}
        if informed:
            reconstructed = self.result_dict["reconstructed_data"]
            self.result_dict["missing_example"] = reconstructed[-1] if reconstructed is not None else None
        self.result_dict["max_max_depth"] = max(t.tree_.max_depth for t in self.estimators)
        return self.result_dict
