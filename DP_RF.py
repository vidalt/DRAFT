import numpy as np
from diffprivlib.models import RandomForestClassifier as DP_RandomForestClassifier

class DP_RF:
    def __init__(self, n_estimators, max_depth, random_state, verbosity=0, seed_noise=0):
        self.N_trees = n_estimators
        self.depth = max_depth
        self.seed = random_state
        self.seed_noise = seed_noise
        self.verbosity = verbosity

        self.maximum_depth = max_depth
        self.clf = DP_RandomForestClassifier(n_estimators=n_estimators, max_depth=self.maximum_depth, bounds=(0, 1), classes=[0, 1], random_state=self.seed)

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        if X.ndim != 2 or not np.isin(X, [0, 1]).all():
            raise ValueError("DP_RF supports binary features only.")
        if y.ndim != 1 or len(y) != len(X) or not np.isin(y, [0, 1]).all():
            raise ValueError("DP_RF expects one binary label per example.")
        self.clf.fit(X, y)
        self.estimators_ = self.clf.estimators_
        self.n_classes_ = self.clf.n_classes_
        self.n_estimators = self.clf.n_estimators

        def split_feature(X, y, feature, threshold):
            mask = X[:, feature] <= threshold
            return X[mask], X[~mask], y[mask], y[~mask]

        def fill_tree(tree, X, y, node_id):
            if tree.feature[node_id] == -2:
                return
            feature, threshold = tree.feature[node_id], tree.threshold[node_id]
            left_X, right_X, left_y, right_y = split_feature(X, y, feature, threshold)
            left_child, right_child = tree.children_left[node_id], tree.children_right[node_id]
            for c in range(len(tree.value[0][0])):
                if left_child != -1:
                    tree.value[left_child][0][c] = (left_y == c).sum()
                if right_child != -1:
                    tree.value[right_child][0][c] = (right_y == c).sum()
            fill_tree(tree, left_X, left_y, left_child)
            fill_tree(tree, right_X, right_y, right_child)

        for tree in self.clf.estimators_:
            for c in range(len(tree.tree_.value[0][0])):
                tree.tree_.value[0][0][c] = len(y[y == c])
            fill_tree(tree.tree_, X, y, 0)

    def add_noise(self,eps):
        if not np.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be finite and strictly positive.")
        eps_v = eps / self.N_trees
        N_leaves = 2 ** self.clf.estimators_[0].tree_.max_depth
        card_c = self.clf.n_classes_
        noise_rng = np.random.RandomState(self.seed_noise)
        delta = [noise_rng.laplace(scale=1.0 / eps_v) for _ in range(self.N_trees * N_leaves * card_c)]
        for tree in self.clf.estimators_:
            for i in range(len(tree.tree_.value)):
                if tree.tree_.children_left[i] == -1:
                    for j in range(len(tree.tree_.value[i][0])):
                        tree.tree_.value[i][0][j] += int(delta.pop())
                else:
                    for j in range(len(tree.tree_.value[i][0])):
                        tree.tree_.value[i][0][j] = 0

    def predict(self,X):
        # fixes compatibility breaking between sklearn 1.3.2 and 1.4.0
        # predict uses confidence scores which for sklearn>=1.4 are supposed to be already normalized
        # so we normalize them here (per leaf) and then re-set them to their raw count to be displayed (as in sklearn<=1.3.2)
        compatibility_fix = True
        if compatibility_fix:
            from copy import deepcopy
            old_counts = []
            for tree in self.clf.estimators_:
                old_counts.append(deepcopy(tree.tree_.value))
                for i in range(len(tree.tree_.value)):
                    if tree.tree_.children_left[i] == -1:
                        old_vals = tree.tree_.value[i][0]
                        #print("old = ", old_vals)
                        min_val = min(old_vals)
                        if min_val < 0:
                            old_vals += abs(min_val)
                        #print("redresssed = ", old_vals)
                        if sum(old_vals) == 0:
                            new_vals = [1/self.clf.n_classes_ for i in range(self.clf.n_classes_)] # uniform
                            #print("uniform = ", new_vals)
                        else:
                            new_vals = (old_vals)/sum(old_vals)
                            #print("new = ", new_vals)
                        tree.tree_.value[i][0] = new_vals
                        #print("===========")

        try:
            return self.clf.predict(np.asarray(X))
        finally:
            if compatibility_fix:
                for t, tree in enumerate(self.clf.estimators_):
                    for i in range(len(tree.tree_.value)):
                        if tree.tree_.children_left[i] == -1:
                            tree.tree_.value[i][0] = old_counts[t][i][0]

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
