# Example datasets

These CSV files are copied unchanged from the original DRAFT repository at commit `22e832e3726a9f18bbbeaed108a220ef4db237fb`:

https://github.com/vidalt/DRAFT/tree/22e832e3726a9f18bbbeaed108a220ef4db237fb/data

- `compas.csv`: the notebook's default binary / one-hot example.
- `adult.csv` and `default_credit.csv`: alternative binary / one-hot examples.
- `default_credit_numerical.csv`: mixed numerical / ordinal example for ordinary DRAFT.

Feature groups, targets, and domains are in `datasets_infos.py`. The default notebook demonstrates binary-feature attacks. Changing it to a mixed-feature dataset requires supplying the ordinal/numerical domains to DRAFT and its evaluation helper; DRAFT-DP supports binary features only.
