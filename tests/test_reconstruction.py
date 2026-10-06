"""Behavior checks for the unified public APIs and supported forest types."""
import numpy as np
import pandas as pd
import pytest
from ortools.sat.python import cp_model
from sklearn.ensemble import RandomForestClassifier
from DRAFT import DRAFT
from DP_RF import DP_RF
from DP_RF_solver import DRAFT_DP
from utils import average_error, print_reconstruction_results


@pytest.fixture
def binary_data():
    X = pd.DataFrame([[0, 0], [0, 1], [1, 0], [1, 1]], columns=['a', 'b'])
    y = pd.Series([0, 0, 1, 1])
    return X, y


@pytest.mark.parametrize('bootstrap', [False, True])
def test_sklearn_reconstruction(binary_data, bootstrap):
    X, y = binary_data
    clf = RandomForestClassifier(n_estimators=20, bootstrap=bootstrap, random_state=0).fit(X, y)
    result = DRAFT(clf).fit(timeout=10, n_threads=1)
    assert result['status'] in {'OPTIMAL', 'FEASIBLE'}
    assert np.asarray(result['reconstructed_data']).shape == X.shape
    assert average_error(result['reconstructed_data'], X)[0] == 0


@pytest.mark.parametrize('n_samples', [4, None])
def test_dp_reconstruction_matches_tree_counts(binary_data, n_samples):
    X, y = binary_data
    clf = DP_RF(n_estimators=3, max_depth=2, random_state=0)
    clf.fit(X, y)
    clf.add_noise(1000)
    before = [tree.tree_.value.copy() for tree in clf.estimators_]
    clf.predict(X)
    for tree, counts in zip(clf.estimators_, before):
        np.testing.assert_array_equal(tree.tree_.value, counts)
    result = DRAFT_DP(clf, 1000).fit(n_samples, timeout=10, n_threads=1, max_samples=8)
    assert result['status'] in {'OPTIMAL', 'FEASIBLE'}
    reconstruction = np.asarray(result['reconstructed_data'])
    assert reconstruction.shape == X.shape
    # Returned class counts in each leaf must sum to actual reconstructed rows.
    for estimator, counts in zip(clf.estimators_, result['nb_recons']):
        leaf_ids = estimator.apply(reconstruction)
        leaves = clf._get_numeros_leaves(estimator.tree_)
        assert [int(np.sum(leaf_ids == leaf)) for leaf in leaves] == [sum(c) for c in counts]


def test_non_dp_random_structure(binary_data):
    X, y = binary_data
    clf = DP_RF(n_estimators=10, max_depth=3, random_state=0)
    clf.fit(X, y)
    result = DRAFT(clf).fit(timeout=10, n_threads=1)
    assert result['status'] in {'OPTIMAL', 'FEASIBLE'}
    assert average_error(result['reconstructed_data'], X)[0] == 0


def test_numerical_repeated_fit():
    X = np.array([[0.1, 0], [0.4, 0], [0.7, 1], [0.9, 1]])
    clf = RandomForestClassifier(n_estimators=3, bootstrap=False, random_state=0).fit(X, [0, 0, 1, 1])
    attack = DRAFT(clf, numerical_attributes=[[0, 0.0, 1.0]])
    for _ in range(2):
        result = attack.fit(timeout=10, n_threads=1)
        assert result['status'] in {'OPTIMAL', 'FEASIBLE'}
        reconstruction = np.asarray(result['reconstructed_data'])
        assert ((0 <= reconstruction[:, 0]) & (reconstruction[:, 0] <= 1)).all()


def test_no_solution_is_not_evaluated(binary_data, capsys):
    X, y = binary_data
    clf = RandomForestClassifier(n_estimators=5, random_state=0).fit(X, y)
    result = DRAFT(clf).fit(timeout=1e-9, n_threads=1, verbosity=True)
    assert result['status'] == 'UNKNOWN'
    assert result['reconstructed_data'] is None
    print_reconstruction_results('timeout', result, X)
    assert 'No reconstruction was produced' in capsys.readouterr().out


@pytest.mark.parametrize('options', [{'timeout':0}, {'timeout':float('nan')}, {'n_threads':0}, {'seed':-1}])
def test_invalid_options(binary_data, options):
    X, y = binary_data
    clf = RandomForestClassifier(n_estimators=1, random_state=0).fit(X, y)
    with pytest.raises(ValueError):
        DRAFT(clf).fit(**options)
    dp = DP_RF(n_estimators=2, max_depth=2, random_state=0)
    dp.fit(X, y)
    dp.add_noise(10)
    with pytest.raises(ValueError):
        DRAFT_DP(dp, 10).fit(4, **options)


def test_dp_timeout_returns_none(binary_data):
    X, y = binary_data
    clf = DP_RF(n_estimators=3, max_depth=2, random_state=0)
    clf.fit(X, y)
    clf.add_noise(10)
    result = DRAFT_DP(clf, 10).fit(4, timeout=1e-9, n_threads=1)
    assert result['status'] == 'UNKNOWN'
    assert result['reconstructed_data'] is None


def test_ohe_constraints():
    X = np.array([[0, 1], [1, 0], [0, 1], [1, 0]])
    y = np.array([0, 1, 0, 1])
    forest = RandomForestClassifier(n_estimators=10, bootstrap=False, random_state=0).fit(X, y)
    ordinary = DRAFT(forest, one_hot_encoded_groups=[[0, 1]]).fit(timeout=10, n_threads=1)
    protected = DP_RF(n_estimators=3, max_depth=2, random_state=0)
    protected.fit(X, y)
    protected.add_noise(1000)
    private = DRAFT_DP(protected, 1000, one_hot_encoded_groups=[[0, 1]]).fit(4, timeout=10, n_threads=1)
    for result in (ordinary, private):
        assert result['status'] in {'OPTIMAL', 'FEASIBLE'}
        assert np.asarray(result['reconstructed_data']).sum(axis=1).tolist() == [1]*4


def test_prediction_failure_restores_counts(binary_data, monkeypatch):
    X, y = binary_data
    clf = DP_RF(n_estimators=3, max_depth=2, random_state=0)
    clf.fit(X, y)
    clf.add_noise(10)
    before = [t.tree_.value.copy() for t in clf.estimators_]
    def fail(_):
        raise RuntimeError('prediction failed')
    monkeypatch.setattr(clf.clf, 'predict', fail)
    with pytest.raises(RuntimeError, match='prediction failed'):
        clf.predict(X)
    for tree, counts in zip(clf.estimators_, before):
        np.testing.assert_array_equal(tree.tree_.value, counts)


def test_unknown_size_deactivates_excess_rows(binary_data):
    X, y = binary_data
    forest = DP_RF(n_estimators=3, max_depth=2, random_state=0)
    forest.fit(X, y)
    forest.add_noise(10)
    result = DRAFT_DP(forest, 10).fit(timeout=10, n_threads=1, max_samples=12)
    assert result['status'] in {'OPTIMAL', 'FEASIBLE'}
    reconstruction = np.asarray(result['reconstructed_data'])
    assert len(reconstruction) == result['N']
    assert result['N_min'] <= result['N'] <= result['N_max']
    assert result['N_min'] < result['N_max']
    for estimator, counts in zip(forest.estimators_, result['nb_recons']):
        leaves = forest._get_numeros_leaves(estimator.tree_)
        actual = estimator.apply(reconstruction)
        assert [int(np.sum(actual == leaf)) for leaf in leaves] == [sum(c) for c in counts]


@pytest.mark.parametrize("supply_count", [False, True])
def test_informed_reconstruction_accepts_only_known_rows(binary_data, supply_count):
    X, y = binary_data
    forest = DP_RF(n_estimators=8, max_depth=3, random_state=0)
    forest.fit(X, y)
    forest.add_noise(1000)
    known_X = X.to_numpy()[[0, 1, 3]].copy()
    known_y = y.to_numpy()[[0, 1, 3]].copy()
    original_X, original_y = known_X.copy(), known_y.copy()
    result = DRAFT_DP(forest, 1000).fit(
        4 if supply_count else None, timeout=10, n_threads=1,
        X_known=known_X, y_known=known_y)
    assert result['status'] in {'OPTIMAL', 'FEASIBLE'}
    recovered = np.asarray(result['reconstructed_data'])
    assert result['N'] == 4
    np.testing.assert_array_equal(recovered[:-1], known_X)
    np.testing.assert_array_equal(recovered[-1], X.to_numpy()[2])
    np.testing.assert_array_equal(result['missing_example'], recovered[-1])
    np.testing.assert_array_equal(known_X, original_X)
    np.testing.assert_array_equal(known_y, original_y)


@pytest.mark.parametrize('extra', [
    {'X_known':np.zeros((3,2))},
    {'X_known':np.zeros((4,2)), 'y_known':np.zeros(4)},
    {'X_known':np.zeros((2,2)), 'y_known':np.zeros(2)},
    {'X_known':np.zeros((3,2)), 'y_known':np.zeros(2)},
    {'X_known':np.zeros((3,3)), 'y_known':np.zeros(3)},
    {'X_known':np.full((3,2), np.nan), 'y_known':np.zeros(3)},
    {'X_known':np.zeros((3,2)), 'y_known':np.full(3, 2)},
    {'X_known':np.zeros((3,2)), 'y_known':np.zeros(3), 'target_ratio':0},
    {'target_ratio':1},
])
def test_informed_invalid_inputs(binary_data, extra):
    X, y = binary_data
    forest = DP_RF(n_estimators=2, max_depth=2, random_state=0)
    forest.fit(X, y)
    forest.add_noise(10)
    with pytest.raises(ValueError):
        DRAFT_DP(forest, 10).fit(4, **extra)


@pytest.mark.parametrize('ratio', [0.01, 1000])
def test_informed_proximity_scaling_options(binary_data, ratio):
    X, y = binary_data
    forest = DP_RF(n_estimators=3, max_depth=2, random_state=0)
    forest.fit(X, y)
    forest.add_noise(10)
    result = DRAFT_DP(forest, 10).fit(timeout=10, n_threads=1,
        X_known=X.iloc[[0,1,3]], y_known=y.iloc[[0,1,3]], target_ratio=ratio)
    assert result['status'] in {'OPTIMAL', 'FEASIBLE'}
    np.testing.assert_array_equal(np.asarray(result['reconstructed_data'])[:-1], X.to_numpy()[[0,1,3]])


def test_informed_single_row_does_not_divide_by_zero():
    forest = DP_RF(n_estimators=3, max_depth=2, random_state=0)
    forest.fit(np.array([[1, 0]]), np.array([1]))
    forest.add_noise(1000)
    result = DRAFT_DP(forest, 1000).fit(timeout=10, n_threads=1,
        X_known=np.empty((0,2)), y_known=np.empty(0))
    assert result['status'] in {'OPTIMAL', 'FEASIBLE'}
    assert len(result['reconstructed_data']) == 1
    assert result['missing_example'] == result['reconstructed_data'][0]


def test_informed_timeout_has_no_missing_example(binary_data, monkeypatch):
    X, y = binary_data
    forest = DP_RF(n_estimators=2, max_depth=2, random_state=0)
    forest.fit(X, y)
    forest.add_noise(10)
    monkeypatch.setattr(cp_model.CpSolver, 'Solve', lambda self, model: cp_model.UNKNOWN)
    result = DRAFT_DP(forest, 10).fit(X_known=X.iloc[:-1], y_known=y.iloc[:-1])
    assert result['reconstructed_data'] is None
    assert result['missing_example'] is None
