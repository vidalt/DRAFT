import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from utils import * 

bootstrap_sampling = False
# Load data and subsample a training dataset with 40 examples
df = pd.read_csv("data/compas.csv")
df = df.sample(n=50, random_state = 42, ignore_index= True)
X_train, X_test, y_train, y_test = data_splitting(df, "recidivate-within-two-years:1", test_size=10, seed=42)

# Train a Random Forest (without bootstrap sampling)
clf = RandomForestClassifier(bootstrap = bootstrap_sampling, random_state = 42)
clf = clf.fit(X_train, y_train)
accuracy_train = clf.score(X_train, y_train)
accuracy_test = clf.score(X_test, y_test)
print("accuracy_train=", accuracy_train, "accuracy_test=",accuracy_test)

# Reconstruct the Random Forest's training set
from DRAFT import DRAFT

extractor = DRAFT(clf)
dict_res = extractor.fit(bagging=bootstrap_sampling, method="cp-sat", timeout=60, verbosity=False, n_jobs=-1, seed=42) # 'status':solve_status, 'duration': duration, 'reconstructed_data':x_sol

duration = dict_res['duration']
x_sol = dict_res['reconstructed_data']

# Evaluate and display the reconstruction rate
e_mean, list_matching = average_error(x_sol,X_train.to_numpy())

print("Complete solving duration :", duration)
print("Reconstruction Error: ", e_mean)