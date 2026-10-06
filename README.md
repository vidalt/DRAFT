# DRAFT: Dataset Reconstruction Attacks From Tree ensembles

Reconstruct training data from a trained random forest using **DRAFT** (ICML 2024), or from the supported differentially private forest using **DRAFT-DP** (SaTML 2026).

This version brings both attacks together with a shared Python API and illustrated notebooks. It contains the reusable reconstruction code and examples, without the research experiment runners, result archives, plotting scripts, or Slurm configuration.

**Looking to reproduce a paper?** Use the [original ICML 2024 code snapshot](https://github.com/vidalt/DRAFT/tree/22e832e3726a9f18bbbeaed108a220ef4db237fb) or the [SaTML 2026 research code snapshot](https://github.com/vidalt/DRAFT-DP/tree/e7067660d21e7d9208dc1758c473b882e52c4313). The snapshot captures DRAFT immediately before this refactor; consult its README for experiment instructions.

## Demo notebooks

Our main attack methodology is designed to reconstruct the entire training set of a given random forest. It encompasses traditional, non-DP random forests, but can also handle random forests protected using a state-of-the-art differential privacy mecanism. We illustrate both cases in our demo: [Open the complete notebook with saved outputs](demo-DRAFT.ipynb).

However, most prior works tackling reconstruction attacks consider a simpler setup where all but one of the training examples are known, and the attacker’s objective is to reconstruct the remaining example. We therefore also provide a specific version of our attack tackling this special case, and an illustrative notebook: [Open the complete notebook with saved outputs](demo-informed-DP.ipynb).

## Installation

Use Python 3.10 or newer in a virtual environment, from a checkout of this repository:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[notebook]"
python -m jupyter lab demo-DRAFT.ipynb
```

On Windows, activate the environment with `.venv\Scripts\activate`.

For Python-only usage, install with `python -m pip install -e .`. OR-Tools CP-SAT is the default solver and needs no commercial license. The optional MILP formulation requires `python -m pip install -e ".[milp]"` and an appropriate Gurobi license.

The notebook and bundled CSV files are used from the repository root. The Python modules are installable; the notebook and datasets are repository examples, not installed package data.

## Quick start

```python
from sklearn.ensemble import RandomForestClassifier
from DRAFT import DRAFT
from DP_RF import DP_RF
from DP_RF_solver import DRAFT_DP
from utils import load_dataset, print_reconstruction_results
from datasets_infos import datasets_ohe_vectors, predictions

X_train, X_test, y_train, y_test = load_dataset(
    "data/compas.csv", predictions["compas"],
    train_size=100, test_size=100, seed=0,
)
groups = datasets_ohe_vectors["compas"]
options = dict(timeout=60, n_threads=2, seed=0)

# Ordinary scikit-learn random forest.
forest = RandomForestClassifier(n_estimators=100, random_state=0)
forest.fit(X_train, y_train)
result = DRAFT(forest, one_hot_encoded_groups=groups).fit(**options)
print_reconstruction_results("DRAFT", result, X_train)

# The supported random-structure random forest with noisy leaf counts.
forest_dp = DP_RF(n_estimators=10, max_depth=5, random_state=0)
forest_dp.fit(X_train, y_train)
forest_dp.add_noise(10)
result_dp = DRAFT_DP(
    forest_dp, epsilon=10, one_hot_encoded_groups=groups,
).fit(n_samples=len(X_train), **options)
print_reconstruction_results("DRAFT-DP", result_dp, X_train)
```

Training data is supplied to the target model for training and to the evaluation helper for measuring reconstruction error. Neither attack receives the original training features. In this example, DRAFT-DP is given the sample count and privacy budget.

## Shared API

```python
DRAFT(random_forest, one_hot_encoded_groups=None,
      ordinal_attributes=None, numerical_attributes=None)
DRAFT_DP(random_forest, epsilon, one_hot_encoded_groups=None)

DRAFT(...).fit(timeout=60, verbosity=False, n_threads=-1, seed=0,
               bagging=None, method="cp-sat")
DRAFT_DP(...).fit(n_samples=None, timeout=60, verbosity=False,
                  n_threads=-1, seed=0, max_samples=400)
```

Both attacks accept the same solver options:

| Option | Meaning |
| --- | --- |
| `timeout` | Solver search time limit, in seconds; model construction takes additional time. |
| `verbosity` | Print parsing and solver progress. |
| `n_threads` | Positive thread count, or `-1` for all available threads. |
| `seed` | Solver random seed. Multiple threads can still affect reproducibility. |

Both return a dictionary with `status`, `duration`, `reconstructed_data`, and `max_max_depth`. `duration` includes construction and search. If no feasible solution is found before timeout, `reconstructed_data` is `None`.

DRAFT-DP additionally returns `nb_recons` (reconstructed per-leaf class counts), `N` (sample count), and `N_min`/`N_max` (estimated-size bounds; zero when a fixed count was supplied) when a feasible solution exists.

### Supported models and features

| Attack | Target | Features | Additional options |
| --- | --- | --- | --- |
| `DRAFT` | `sklearn.ensemble.RandomForestClassifier` | Binary, one-hot categorical, ordinal, numerical | Bagging is inferred from the forest; `method="milp"` supports binary features without bagging. |
| `DRAFT` | Unnoised `DP_RF` | Binary / one-hot categorical | Useful for the notebook's before/after comparison. |
| `DRAFT_DP` | Noised `DP_RF` using this repository's mechanism | Binary / one-hot categorical | Supply `n_samples`, or infer it with `n_samples=None` and an explicit `max_samples` cap. |

DRAFT-DP does not accept an arbitrary differentially private random forest: its likelihood model is specific to the random-tree structure and integer-truncated Laplace leaf-count noise used by `DP_RF`. This corresponds to a method from the literature, introduced in the following paper:
> G. Jagannathan, K. Pillaipakkamnatt, and R. N. Wright, “A practical differentially private random decision tree classifier,” Trans. Data Priv., vol. 5, no. 1, pp. 273–295, 2012.

`DP_RF` expects binary labels `0` and `1`. Its `fit()` first constructs unnoised counts; call `add_noise(epsilon)` once on a fresh trained copy before using the protected model. The notebook retains a separate unnoised copy for comparison.

For DRAFT, ordinal and numerical metadata use `[feature_index, lower_bound, upper_bound]` triples. Numerical reconstruction identifies split-defined intervals and returns their midpoints. One-hot groups contain zero-based feature indices. See `datasets_infos.py` for examples.

`average_error(reconstructed_data, training_data)` evaluates the optimal row matching. For mixed features, pass `dataset_ordinal=` and `dataset_numerical=` explicitly to normalize their distances by the declared domains. The notebook's highlighted tables use binary mismatch costs.

## Files

| File | Purpose |
| --- | --- |
| `DRAFT.py` | Random forest reconstruction, including CP-SAT and optional MILP. |
| `DP_RF_solver.py` | Reconstruction against the supported DP random forest. |
| `DP_RF.py` | The target random-structure forest and leaf-count noise mechanism. |
| `demo-DRAFT.ipynb` | End-to-end demo with highlighted training/reconstruction tables. |
| `demo-informed-DP.ipynb` | Standalone DP-only demo with one missing example and saved target-only outputs. |
| `utils.py` | Dataset loading, evaluation, and notebook display helpers. |
| `datasets_infos.py`, `data/` | Example feature metadata and CSV datasets from the original DRAFT repository. |
| `tests/` | Small regression tests for the public APIs. |

## Validation

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

The prepared bundle was checked on Python 3.12 with NumPy 2.3.5, SciPy 1.17.0, pandas 2.2.3, scikit-learn 1.8.0, OR-Tools 9.14.6206, and diffprivlib 0.6.6. These are the tested versions; dependency ranges in `pyproject.toml` are not a claim that every combination was tested. MILP remains available but was not runtime-tested in this environment.

## Papers

- Julien Ferry, Ricardo Fukasawa, Timothée Pascal, and Thibaut Vidal. **Trained Random Forests Completely Reveal your Dataset.** ICML 2024. [Proceedings](https://proceedings.mlr.press/v235/ferry24a.html).
- **Training Set Reconstruction from Differentially Private Forests: How Effective is DP?** SaTML 2026. [Preprint](https://arxiv.org/abs/2502.05307) · [Research code](https://github.com/vidalt/DRAFT-DP).

Please cite the corresponding paper when using either method. The MIT license notices from both source repositories are retained in [LICENSE](LICENSE).

## Informed-adversary reconstruction

An **informed adversary** knows the feature vectors and class labels of **all training examples except one**. Given the protected forest and its privacy budget, the attack tries to reconstruct that remaining example. This is a stronger knowledge assumption than the standard reconstruction examples above, which do not supply training examples to the attack.

In such a case, it is sufficient to supply **the known examples and their labels**, through `X_known` and `y_known`. 

| Argument | Meaning |
| --- | --- |
| `X_known` | An `(N-1, n_features)` array containing only the known binary feature vectors. |
| `y_known` | A length-`N-1` array containing their known labels (`0` or `1` for DP_RF). |
| `n_samples` | Optional in informed mode: inferred as `len(X_known) + 1`; if supplied, it must equal this value. |
| `target_ratio` | Optional target ratio of likelihood-objective range to proximity-objective range; defaults to `epsilon / 2`. |

**Exactly one missing example is supported.** Multiple missing examples require extending the formulation and are not supported by this API.

[Open the standalone DP informed-adversary notebook with saved outputs](demo-informed-DP.ipynb), or launch it with `python -m jupyter lab demo-informed-DP.ipynb`. 

For standard reconstruction, omit the known-example inputs:

```python
result = DRAFT_DP(clf_dp, epsilon, one_hot_encoded_groups=ohe_vector).fit(
    n_samples=len(X_train), timeout=60, n_threads=2, seed=seed,
)
```

For informed reconstruction with a trained protected forest `clf_dp`:

```python
import numpy as np
from DP_RF_solver import DRAFT_DP

# For this demonstration only, remove row 0 to simulate the adversary's knowledge.
# In actual usage, provide the N-1 examples and labels you already know.
known_rows = np.arange(len(X_train)) != 0
known_X = X_train.to_numpy()[known_rows]
known_y = y_train.to_numpy()[known_rows]

informed_result = DRAFT_DP(
    clf_dp, epsilon, one_hot_encoded_groups=ohe_vector,
).fit(
    X_known=known_X,
    y_known=known_y,
    timeout=60,
    n_threads=2,
    seed=seed,
)

if informed_result["missing_example"] is not None:
    print("Reconstructed missing example:", informed_result["missing_example"])
else:
    print("No feasible reconstruction:", informed_result["status"])
```

The known examples and labels are fixed, and only the missing example contributes new leaf assignments. The objective combines the original noisy-count likelihood with a proximity term favoring similarity to known examples, using the original relative scaling rule. Larger `target_ratio` values favor likelihood relative to proximity; coefficient rounding can affect the precise balance.

In informed mode, the result additionally contains `missing_example`, the reconstructed feature vector, or `None` without a feasible solution. `reconstructed_data` contains the known rows in their supplied order followed by the missing row. A feasible solution does not guarantee exact recovery.
