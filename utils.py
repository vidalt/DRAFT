import random
from collections.abc import Mapping as _Mapping
from numbers import Integral as _Integral

import numpy as np
from scipy.optimize import linear_sum_assignment

# Helper function
def print_reconstruction_results(
    experiment_name,
    dict_res,
    X_train,
    display_reconstruction=False,
):
    """Print reconstruction metrics and optionally display a highlighted table.

    When ``display_reconstruction`` is true, the complete reconstruction is
    shown in a scrollable HTML table. Each reconstructed value is compared with
    the corresponding value in its matched training example: matching cells are
    green and differing cells are red. For equally sized datasets, this is the
    same optimal one-to-one matching used to calculate the error. Hovering over
    a cell shows the expected value from the training data. If the solver did
    not produce a reconstruction, its status is reported and the table is
    skipped.
    """
    if not isinstance(display_reconstruction, (bool, np.bool_)):
        raise TypeError("display_reconstruction must be a boolean")

    # Report the solver outcome before attempting to read its reconstruction.
    duration = dict_res.get('duration')
    if duration is not None:
        print(
            "(%s) Complete solving duration : %.3f seconds."
            % (experiment_name, duration)
        )

    x_sol = dict_res.get('reconstructed_data')
    if x_sol is None:
        status = dict_res.get('status', 'not provided')
        print(
            "(%s) No reconstruction was produced (solver status: %s)."
            % (experiment_name, status)
        )
        if str(status).upper() == 'UNKNOWN':
            print(
                "The solver may have reached its time limit before finding a "
                "feasible solution; try increasing timeout and rerun it."
            )
        return

    # Evaluate and display the reconstruction rate
    e_mean, list_matching = average_error(x_sol, X_train)

    print("(%s) Reconstruction Error: %.6f" % (experiment_name, e_mean))

    if display_reconstruction:
        _display_reconstruction_table(
            experiment_name,
            x_sol,
            X_train,
            list_matching,
        )

def _as_array(data, name):
    """Convert a supported dataset representation to a non-empty 2-D array."""
    if isinstance(data, _Mapping):
        data = list(data.values())
    elif hasattr(data, "to_numpy"):
        data = data.to_numpy()

    try:
        array = np.asarray(data)
    except ValueError as exc:
        raise ValueError(f"{name} must be a rectangular two-dimensional dataset") from exc

    if array.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional; got shape {array.shape}")
    if array.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one example")
    if array.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one attribute")
    return array


def _display_row_matching(x_sol, x_train, evaluation_matching):
    """Return a training-row match for every original reconstruction row."""
    if x_sol.shape[0] == len(evaluation_matching):
        return np.asarray(evaluation_matching, dtype=int)

    # ``average_error`` resizes unequal datasets before matching. For display,
    # preserve every original reconstruction row and match the rectangular data
    # directly. If there are more reconstructed than training rows, associate
    # any unmatched extras with their closest training row.
    cost = matrice_matching(x_sol, x_train)
    reconstructed_ids, training_ids = linear_sum_assignment(cost)
    matching = np.full(x_sol.shape[0], -1, dtype=int)
    matching[reconstructed_ids] = training_ids

    unmatched = np.flatnonzero(matching < 0)
    if unmatched.size:
        matching[unmatched] = np.argmin(cost[unmatched], axis=1)
    return matching


def _values_are_equal(value, expected):
    """Compare scalar cell values while treating two missing values as equal."""
    if value is expected:
        return True

    try:
        if bool(np.isnan(value)) and bool(np.isnan(expected)):
            return True
    except (TypeError, ValueError):
        pass

    try:
        return bool(value == expected)
    except (TypeError, ValueError):
        return False


def _human_readable_columns(data, n_features):
    """Return source column names when available, otherwise generic labels."""
    columns = getattr(data, "columns", None)
    if columns is None or len(columns) != n_features:
        return [f"Feature {i + 1}" for i in range(n_features)]
    return [str(column).replace(":", " = ") for column in columns]


def _table_headers(columns):
    """Build the dark, sticky column headers shared by notebook tables."""
    from html import escape

    return "".join(
        '<th style="position:sticky;top:0;background:#f3f4f6;'
        'color:#111827;font-weight:700;padding:6px 10px;'
        'border:1px solid #d1d5db;z-index:2;'
        'white-space:nowrap">%s</th>' % escape(column)
        for column in columns
    )


def display_dataset(data, title="Dataset"):
    """Display an entire dataset in a neutral, scrollable notebook table."""
    from html import escape
    from IPython.display import HTML, display

    values = _as_array(data, "data")
    columns = _human_readable_columns(data, values.shape[1])
    headers = _table_headers(columns)
    source_index = getattr(data, "index", None)

    rows = []
    for row_id, row in enumerate(values):
        source_label = source_index[row_id] if source_index is not None else row_id + 1
        background = "#ffffff" if row_id % 2 == 0 else "#f9fafb"
        cells = "".join(
            '<td style="background:%s;color:#111827;padding:6px 10px;'
            'border:1px solid #d1d5db;text-align:center;'
            'white-space:nowrap">%s</td>'
            % (background, escape(str(value)))
            for value in row
        )
        rows.append(
            '<tr><th title="Source row: %s" '
            'style="position:sticky;left:0;background:#f3f4f6;color:#111827;'
            'font-weight:700;padding:6px 10px;border:1px solid #d1d5db;'
            'z-index:1">%d</th>%s</tr>'
            % (
                escape(str(source_label), quote=True),
                row_id + 1,
                cells,
            )
        )

    table = (
        '<div style="margin:10px 0">'
        '<div style="margin-bottom:6px"><strong>%s</strong> &mdash; '
        '%d examples</div>'
        '<div style="max-height:430px;max-width:100%%;overflow:auto;'
        'border:1px solid #d1d5db;border-radius:6px">'
        '<table style="border-collapse:collapse;font-family:system-ui,sans-serif;'
        'font-size:13px"><thead><tr>'
        '<th style="position:sticky;top:0;left:0;background:#e5e7eb;'
        'color:#111827;font-weight:700;padding:6px 10px;'
        'border:1px solid #d1d5db;z-index:3">Example</th>'
        '%s</tr></thead><tbody>%s</tbody></table></div></div>'
        % (
            escape(str(title)),
            values.shape[0],
            headers,
            "".join(rows),
        )
    )
    display(HTML(table))


def _display_reconstruction_table(
    experiment_name,
    reconstructed_data,
    training_data,
    evaluation_matching,
):
    """Display reconstructed rows ordered by their matched ``X_train`` row."""
    from html import escape
    from IPython.display import HTML, display

    reconstructed = _as_array(reconstructed_data, "reconstructed_data")
    training = _as_array(training_data, "training_data")
    matching = _display_row_matching(
        reconstructed,
        training,
        evaluation_matching,
    )
    reconstruction_ids = np.arange(reconstructed.shape[0])
    display_order = np.argsort(matching, kind="stable")
    reconstructed = reconstructed[display_order]
    matching = matching[display_order]
    reconstruction_ids = reconstruction_ids[display_order]
    matched_training = training[matching]

    columns = _human_readable_columns(training_data, reconstructed.shape[1])

    training_index = getattr(training_data, "index", None)
    correct_cells = 0
    rows = []
    for reconstructed_id, reconstructed_row, expected_row, matched_position in zip(
        reconstruction_ids,
        reconstructed,
        matched_training,
        matching,
    ):
        matched_position = int(matched_position)
        matched_label = (
            training_index[matched_position]
            if training_index is not None
            else matched_position + 1
        )
        cells = []
        for value, expected in zip(reconstructed_row, expected_row):
            is_correct = _values_are_equal(value, expected)
            correct_cells += int(is_correct)
            background = "#d1fae5" if is_correct else "#fee2e2"
            foreground = "#065f46" if is_correct else "#991b1b"
            status = (
                f"Correctly reconstructed; expected {expected}"
                if is_correct
                else f"Incorrect; expected {expected}"
            )
            cells.append(
                '<td title="%s" style="background:%s;color:%s;'
                'padding:6px 10px;border:1px solid #d1d5db;'
                'text-align:center;white-space:nowrap">%s</td>'
                % (
                    escape(status, quote=True),
                    background,
                    foreground,
                    escape(str(value)),
                )
            )

        rows.append(
            '<tr><th title="X_train source index: %s; reconstruction row: %d" '
            'style="position:sticky;left:0;background:#f3f4f6;color:#111827;'
            'font-weight:700;padding:6px 10px;border:1px solid #d1d5db;'
            'z-index:1">%d</th>%s</tr>'
            % (
                escape(str(matched_label), quote=True),
                int(reconstructed_id) + 1,
                matched_position + 1,
                "".join(cells),
            )
        )

    headers = _table_headers(columns)
    total_cells = reconstructed.size
    table = (
        '<div style="margin:10px 0">'
        '<div style="margin-bottom:6px"><strong>%s reconstruction</strong> '
        '&mdash; <span style="color:#065f46">green: correct</span>, '
        '<span style="color:#991b1b">red: incorrect</span> '
        '(%d/%d cells correct). Rows follow their matched X_train order; '
        'hover over a cell to see its comparison.</div>'
        '<div style="max-height:430px;max-width:100%%;overflow:auto;'
        'border:1px solid #d1d5db;border-radius:6px">'
        '<table style="border-collapse:collapse;font-family:system-ui,sans-serif;'
        'font-size:13px"><thead><tr>'
        '<th style="position:sticky;top:0;left:0;background:#e5e7eb;'
        'color:#111827;font-weight:700;padding:6px 10px;'
        'border:1px solid #d1d5db;z-index:3">X_train row</th>'
        '%s</tr></thead><tbody>%s</tbody></table></div></div>'
        % (
            escape(str(experiment_name)),
            correct_cells,
            total_cells,
            headers,
            "".join(rows),
        )
    )
    display(HTML(table))


def _as_vector(individual, name):
    """Convert one example to a non-empty 1-D array."""
    if isinstance(individual, _Mapping):
        individual = list(individual.values())
    elif hasattr(individual, "to_numpy"):
        individual = individual.to_numpy()

    vector = np.asarray(individual)
    if vector.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; got shape {vector.shape}")
    if vector.size == 0:
        raise ValueError(f"{name} must contain at least one attribute")
    return vector


def _normalise_attributes(attrs, n_features, name, require_positive_range=True):
    """Validate ``(index, lower_bound, upper_bound)`` attribute metadata."""
    if attrs is None:
        return {}

    normalised = {}
    for position, attr in enumerate(attrs):
        try:
            attr_id, lower_bound, upper_bound = attr
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{name}[{position}] must contain exactly "
                "(attribute_index, lower_bound, upper_bound)"
            ) from exc

        if not isinstance(attr_id, _Integral):
            raise TypeError(f"{name}[{position}] has a non-integer attribute index")
        attr_id = int(attr_id)
        if not 0 <= attr_id < n_features:
            raise ValueError(
                f"{name}[{position}] refers to attribute {attr_id}, but the dataset "
                f"has {n_features} attributes"
            )
        if attr_id in normalised:
            raise ValueError(f"attribute {attr_id} is listed more than once in {name}")

        try:
            invalid_range = (
                upper_bound <= lower_bound
                if require_positive_range
                else upper_bound < lower_bound
            )
        except TypeError as exc:
            raise TypeError(f"{name}[{position}] bounds must be comparable") from exc
        if invalid_range:
            relation = "greater than" if require_positive_range else "at least"
            raise ValueError(
                f"{name}[{position}] upper bound must be {relation} its lower bound"
            )

        normalised[attr_id] = (lower_bound, upper_bound)
    return normalised


def _normalise_ohe_groups(ohe_groups, n_features):
    """Validate groups of positional one-hot-encoded column indices."""
    if ohe_groups is None:
        return []

    normalised = []
    for group_position, group in enumerate(ohe_groups):
        group = list(group)
        if not group:
            raise ValueError(f"ohe group {group_position} must not be empty")
        if any(not isinstance(index, _Integral) for index in group):
            raise TypeError(f"ohe group {group_position} contains a non-integer index")

        group = [int(index) for index in group]
        if len(set(group)) != len(group):
            raise ValueError(f"ohe group {group_position} contains duplicate indices")
        for index in group:
            if not 0 <= index < n_features:
                raise ValueError(
                    f"ohe group {group_position} refers to attribute {index}, but "
                    f"the dataset has {n_features} attributes"
                )
        normalised.append(group)
    return normalised


def _distance(ind1, ind2, non_binary_attrs):
    """Compute a distance after inputs and metadata have been validated."""
    total = 0.0
    for attr_id, (value_1, value_2) in enumerate(zip(ind1, ind2)):
        if attr_id in non_binary_attrs:
            lower_bound, upper_bound = non_binary_attrs[attr_id]
            total += abs(value_1 - value_2) / (upper_bound - lower_bound)
        elif value_1 != value_2:
            total += 1.0
    return float(total / len(ind1))


def load_dataset(dataset, label, train_size, test_size, seed):
    """Load a CSV file and return reproducible training and test splits.

    The function first samples ``train_size + test_size`` rows from the CSV,
    then uses ``label`` as the target column and randomly splits the sample.
    Both the sampling and splitting steps use ``seed``.

    Parameters
    ----------
    dataset : str or path-like
        Path to the CSV file.
    label : hashable
        Name of the target column. All other columns are returned as features.
    train_size : int
        Number of rows in the training split.
    test_size : int
        Number of rows in the test split.
    seed : int or None
        Random seed used to sample and split the rows.

    Returns
    -------
    tuple
        ``(X_train, X_test, y_train, y_test)`` as pandas DataFrames and Series.
    """
    from sklearn.model_selection import train_test_split
    import pandas as pd
    df = pd.read_csv(dataset)
    df = df.sample(n=train_size+test_size, random_state = seed, ignore_index= True)

    y = df[label]
    X = df.drop(labels = [label], axis = 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= test_size , shuffle = True, random_state = seed)
    return X_train, X_test, y_train, y_test


def dist_individus(ind1, ind2, non_binary_attrs=None):
    """Return the mean per-attribute distance between two examples.

    Attributes absent from ``non_binary_attrs`` contribute zero when equal and
    one when different (Hamming distance).  Each non-binary attribute must be
    described by ``(column_index, lower_bound, upper_bound)`` and contributes
    ``abs(value_1 - value_2) / (upper_bound - lower_bound)``.  Consequently the
    result is in ``[0, 1]`` when values respect all declared domains.

    Parameters
    ----------
    ind1, ind2 : one-dimensional array-like
        Examples with the same number of attributes.
    non_binary_attrs : iterable of triples, optional
        Positional column indices and their numerical domains.
    """
    ind1 = _as_vector(ind1, "ind1")
    ind2 = _as_vector(ind2, "ind2")
    if ind1.shape != ind2.shape:
        raise ValueError(
            f"ind1 and ind2 must have the same shape; got {ind1.shape} and {ind2.shape}"
        )

    attrs = _normalise_attributes(
        non_binary_attrs,
        ind1.size,
        "non_binary_attrs",
        require_positive_range=True,
    )
    return _distance(ind1, ind2, attrs)


def matrice_matching(x_sol, x_train, non_binary_attrs=None):
    """Build the pairwise reconstruction/training distance matrix.

    The returned array has shape ``(len(x_sol), len(x_train))``.  Entry ``i, j``
    is the value returned by :func:`dist_individus` for reconstructed example
    ``i`` and training example ``j``.  The two datasets may have different row
    counts, but they must have the same number of columns.
    """
    x_sol_array = _as_array(x_sol, "x_sol")
    x_train_array = _as_array(x_train, "x_train")
    if x_sol_array.shape[1] != x_train_array.shape[1]:
        raise ValueError(
            "x_sol and x_train must have the same number of attributes; got "
            f"{x_sol_array.shape[1]} and {x_train_array.shape[1]}"
        )

    attrs = _normalise_attributes(
        non_binary_attrs,
        x_sol_array.shape[1],
        "non_binary_attrs",
        require_positive_range=True,
    )
    cost = np.empty((x_sol_array.shape[0], x_train_array.shape[0]), dtype=float)
    for sol_id, reconstructed in enumerate(x_sol_array):
        for train_id, original in enumerate(x_train_array):
            cost[sol_id, train_id] = _distance(reconstructed, original, attrs)
    return cost


def _normalise_average_error_arguments(
    seed,
    return_all_distances,
    dataset_ordinal,
    dataset_numerical,
):
    """Support the positional APIs used by both historical implementations."""
    if seed is not None and not isinstance(seed, _Integral):
        if dataset_ordinal is not None:
            raise TypeError(
                "dataset_ordinal was supplied both positionally and by keyword"
            )
        dataset_ordinal = seed
        seed = 42

        if not isinstance(return_all_distances, (bool, np.bool_)):
            if dataset_numerical is not None:
                raise TypeError(
                    "dataset_numerical was supplied both positionally and by keyword"
                )
            dataset_numerical = return_all_distances
            return_all_distances = False

    if seed is not None and not isinstance(seed, _Integral):
        raise TypeError("seed must be an integer or None")
    if not isinstance(return_all_distances, (bool, np.bool_)):
        raise TypeError("return_all_distances must be a boolean")

    return (
        None if seed is None else int(seed),
        bool(return_all_distances),
        dataset_ordinal,
        dataset_numerical,
    )


def average_error(
    x_sol,
    x_train,
    seed=42,
    return_all_distances=False,
    dataset_ordinal=None,
    dataset_numerical=None,
):
    """Compute the minimum-matching reconstruction error.

    A minimum-cost one-to-one assignment is computed after making the number of
    reconstructed rows equal to the number of training rows:

    * If ``x_sol`` is smaller, reconstructed rows are appended by sampling
      ``x_sol`` with replacement.
    * If ``x_sol`` is larger, reconstructed rows are sampled without replacement
      down to ``len(x_train)``.
    * ``x_train`` is never sampled or modified.

    ``seed`` controls only this resizing and a local random-number generator is
    used, so the global Python random state is not changed.  Both inputs must be
    non-empty two-dimensional datasets with the same number of attributes.
    Dictionaries are interpreted in insertion order through their values;
    pandas objects and other array-like inputs are also accepted.

    By default, attributes use a zero/one mismatch cost.  Attributes listed in
    ``dataset_ordinal`` or ``dataset_numerical`` use their normalized absolute
    difference instead.  Both metadata arguments contain triples of
    ``(column_index, lower_bound, upper_bound)``.

    Parameters
    ----------
    x_sol, x_train : mapping, pandas.DataFrame, or two-dimensional array-like
        Reconstructed examples and actual training examples.
    seed : int or None, default=42
        Seed used when the reconstruction must be resized.  For compatibility,
        the third positional argument may instead be DRAFT-style ordinal
        metadata; in that case the default seed is used.
    return_all_distances : bool, default=False
        If true, return each matched-pair distance as a third result.
    dataset_ordinal, dataset_numerical : iterable of triples, optional
        Non-binary attribute definitions.  Keyword arguments are recommended.

    Returns
    -------
    average_distance : float
        Mean distance across the optimal matched pairs.
    matching : list of int
        For each row of the resized ``x_sol``, the matched row index in the
        original ``x_train``.
    all_distances : list of float, optional
        Distances aligned with ``matching``; returned only when
        ``return_all_distances`` is true.
    """
    (
        seed,
        return_all_distances,
        dataset_ordinal,
        dataset_numerical,
    ) = _normalise_average_error_arguments(
        seed,
        return_all_distances,
        dataset_ordinal,
        dataset_numerical,
    )

    x_sol_array = _as_array(x_sol, "x_sol")
    x_train_array = _as_array(x_train, "x_train")
    if x_sol_array.shape[1] != x_train_array.shape[1]:
        raise ValueError(
            "x_sol and x_train must have the same number of attributes; got "
            f"{x_sol_array.shape[1]} and {x_train_array.shape[1]}"
        )

    ordinal_attrs = [] if dataset_ordinal is None else list(dataset_ordinal)
    numerical_attrs = [] if dataset_numerical is None else list(dataset_numerical)
    non_binary_attrs = ordinal_attrs + numerical_attrs

    n_sol = x_sol_array.shape[0]
    n_train = x_train_array.shape[0]
    rng = random.Random(seed)
    if n_sol < n_train:
        additional_indices = rng.choices(range(n_sol), k=n_train - n_sol)
        x_sol_array = np.concatenate(
            (x_sol_array, x_sol_array[additional_indices]),
            axis=0,
        )
    elif n_sol > n_train:
        selected_indices = rng.sample(range(n_sol), n_train)
        x_sol_array = x_sol_array[selected_indices]

    cost = matrice_matching(
        x_sol_array,
        x_train_array,
        non_binary_attrs=non_binary_attrs,
    )
    row_ind, col_ind = linear_sum_assignment(cost)
    all_distances = [float(cost[row_id, col_id]) for row_id, col_id in zip(row_ind, col_ind)]
    average_distance = float(np.mean(all_distances))
    matching = col_ind.tolist()

    if return_all_distances:
        return average_distance, matching, all_distances
    return average_distance, matching
