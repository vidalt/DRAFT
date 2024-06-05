datasets_ohe_vectors = {"compas": [[1,2,3,4,5], [9,10,11,12]],
                        "adult": [[3,4,5,6,7,8]],# [11,12,13,14,15], [16, 17, 18]],
                        "adult_larger":[[0,1,2], [5,6,7,8,9,10], [12,13,14], [15,16,17,18,19], [20,21,22,23,24,25], [26,27,28,29,30], [31,32,33,34,35,36], [37,38,39,40]],
                        "default_credit":[[0,1,2,3], [5,6,7]]}

predictions = {"compas":"recidivate-within-two-years:1", "adult":"income", "adult_larger":"income", "default_credit":"DEFAULT_PAYEMENT", "default_credit_non_binary":"DEFAULT_PAYEMENT"}

# Below: for ordinal and numerical attributes support
# Each element in the list is the index of an ordinal (integer attribute whose values can be sorted) feature, then the lower and upper bounds on values of such feature
dataset_ordinal = {"compas": [],
                     "adult": [],
                     "adult_larger": [],
                     "default_credit": []}

# Each element in the list is the index of a numerical (continuous) feature, then the lower and upper bounds on values of such feature
dataset_numerical = {"compas": [],
                     "adult": [],
                     "adult_larger": [],
                     "default_credit": []}

