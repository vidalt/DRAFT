import unittest
from pathlib import Path
# Import local functions to test
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from utils import * 
from DRAFT import DRAFT

class test_NoBagging(unittest.TestCase):
    # Load dataset
    dataset = 'data/compas.csv'
    THIS_DIR = Path(__file__).parent
    datasetFile = THIS_DIR.parent / dataset
    df = pd.read_csv(dataset)

    # Subsample and divide it
    df = df.sample(n=40, random_state = 42, ignore_index= True)
    X_train, X_test, y_train, y_test = data_splitting(df, "recidivate-within-two-years:1", 20, 42)
    X_train = X_train.to_numpy() # 50 examples

    # Train a Random Forest (without bootstrap sampling)
    clf = RandomForestClassifier(bootstrap = False, random_state = 42)
    clf = clf.fit(X_train, y_train)
    accuracy_train = clf.score(X_train, y_train)
    accuracy_test = clf.score(X_test, y_test)

    # Create the DRAFT reconstructor
    extractor = DRAFT(clf)

    def test_CPModel(self):
        dict_res = self.extractor.fit(bagging=False, method="cp-sat", timeout=60, verbosity=False, n_jobs=-1, seed=42) # 'status':solve_status, 'duration': duration, 'reconstructed_data':x_sol
        duration = dict_res['duration']
        x_sol = dict_res['reconstructed_data']
        e_mean, list_matching = ecart_moyen(x_sol,list(self.X_train))
        self.assertTrue(e_mean < 0.005) # Expect even less
    
    def test_MILPModel(self):
        dict_res = self.extractor.fit(bagging=False, method="milp", timeout=60, verbosity=False, n_jobs=-1, seed=42) # 'status':solve_status, 'duration': duration, 'reconstructed_data':x_sol
        duration = dict_res['duration']
        x_sol = dict_res['reconstructed_data']
        e_mean, list_matching = ecart_moyen(x_sol,list(self.X_train))
        self.assertTrue(e_mean < 0.005) # Expect even less