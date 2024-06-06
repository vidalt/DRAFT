datasets_ohe_vectors = {"compas": [[1,2,3,4,5], [9,10,11,12]],
                        "adult": [[3,4,5,6,7,8]],# [11,12,13,14,15], [16, 17, 18]],
                        "adult_larger":[[0,1,2], [5,6,7,8,9,10], [12,13,14], [15,16,17,18,19], [20,21,22,23,24,25], [26,27,28,29,30], [31,32,33,34,35,36], [37,38,39,40]],
                        "default_credit":[[0,1,2,3], [5,6,7]],
                        "default_credit_numerical":[[15,16,17,18],[19,20,21]],
                        "default_credit_numerical_2":[[0,1,2,3], [5,6,7]]}

predictions = {"compas":"recidivate-within-two-years:1", "adult":"income", "adult_larger":"income", 
               "default_credit":"DEFAULT_PAYEMENT", "default_credit_numerical":"DEFAULT_PAYEMENT",
               "default_credit_numerical_2":"DEFAULT_PAYEMENT"}

# Below: for ordinal and numerical attributes support
# Each element in the list is the index of an ordinal (integer attribute whose values can be sorted) feature, then the lower and upper bounds on values of such feature
datasets_ordinal_attrs = {"compas": [],
                     "adult": [],
                     "adult_larger": [],
                     "default_credit": [],
                     "default_credit_numerical": [[1, 21, 79]],
                     "default_credit_numerical_2": [[4,-1,-1]]}

# Each element in the list is the index of a numerical (continuous) feature, then the lower and upper bounds on values of such feature
datasets_numerical_attrs = {"compas": [],
                     "adult": [],
                     "adult_larger": [],
                     "default_credit": [],
                     "default_credit_numerical": [[0, 10000, 1000000], [2, -165580, 964511], [3, -69777, 983931], [4, -157264, 1664089], [5, -170000, 891586], [6, -81334, 927171], [7, -339603, 961664], [8, 0, 873552], [9, 0, 1684259], [10, 0, 896040], [11, 0, 621000], [12, 0, 426529], [13, 0, 528666]],
                     "default_credit_numerical_2":[]}

