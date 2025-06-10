# Load packages
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from utils import * 

# Load data and subsample a toy training dataset with 10 examples
df = pd.read_csv("data/compas.csv")
df = df.sample(n=50, random_state = 0, ignore_index= True)
X_train, X_test, y_train, y_test = data_splitting(df, "recidivate-within-two-years:1", test_size=40, seed=42)

# Train a Random Forest
clf = RandomForestClassifier()
clf = clf.fit(X_train, y_train)

# Reconstruct the Random Forest's training set
from DRAFT import DRAFT

extractor = DRAFT(clf)
dict_res = extractor.fit(timeout=60) 

# Retrieve solving time and reconstructed data
duration = dict_res['duration']
x_sol = dict_res['reconstructed_data']

# Evaluate and display the reconstruction rate
e_mean, list_matching = average_error(x_sol,X_train.to_numpy())

print("Complete solving duration :", duration)
print("Reconstruction Error: ", e_mean)