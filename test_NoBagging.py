import unittest
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from utils import * 
# Import local functions to test
from DRAFT import DRAFT

class test_NoBagging(unittest.TestCase):
    # Load dataset
    dataset = 'data/compas.csv'
    THIS_DIR = Path(__file__).parent
    datasetFile = THIS_DIR.parent / dataset
    df = pd.read_csv(dataset)

    # Subsample and divide it
    df = df.sample(n=50, random_state = 42, ignore_index= True)
    X_train, X_test, y_train, y_test = data_splitting(df, "recidivate-within-two-years:1", test_size=10, seed=42)

    # Train a Random Forest (without bootstrap sampling)
    clf = RandomForestClassifier(bootstrap = False, random_state = 42)
    clf = clf.fit(X_train, y_train)
    accuracy_train = clf.score(X_train, y_train)
    accuracy_test = clf.score(X_test, y_test)

    # Create the DRAFT reconstructor
    extractor = DRAFT(clf)

    def test_CPModel(self):
        # Perform the reconstruction using the CP formulation and checks its resulting reconstruction error
        dict_res = self.extractor.fit(bagging=False, method="cp-sat", timeout=60, verbosity=False, n_jobs=-1, seed=42) # 'status':solve_status, 'duration': duration, 'reconstructed_data':x_sol
        duration = dict_res['duration']
        x_sol = dict_res['reconstructed_data']
        # Evaluate and display the reconstruction rate
        e_mean, list_matching = average_error(x_sol,self.X_train.to_numpy())
        self.assertTrue(e_mean < 0.005) # Expect even less

    def test_MILPModel(self):
        from gurobipy import GurobiError
        try:
            # Perform the reconstruction using the MILP formulation and checks its resulting reconstruction error
            dict_res = self.extractor.fit(bagging=False, method="milp", timeout=60, verbosity=False, n_jobs=-1, seed=42) # 'status':solve_status, 'duration': duration, 'reconstructed_data':x_sol
            duration = dict_res['duration']
            x_sol = dict_res['reconstructed_data']
            # Evaluate and display the reconstruction rate
            e_mean, list_matching = average_error(x_sol,self.X_train.to_numpy())
            self.assertTrue(e_mean < 0.005) # Expect even less
        except GurobiError:
            print("Warning: Gurobi license not found:"
                  " cannot run integration test that solves MILP.")