def data_splitting(data, label, test_size, seed):
    """
    Splits data between train and test sets using the label column as prediction and test_size examples for the test set (can be a proportion as well).
    """
    from sklearn.model_selection import train_test_split
    y = data[label]
    X = data.drop(labels = [label], axis = 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= test_size , shuffle = False, random_state = seed)
    return X_train, X_test, y_train, y_test

def dist_individus(ind1,ind2):
    """
    Computes the manhattan distance between two examples.
    """
    nbfd = 0 #Compteur nombre de features différentes
    m = len(ind1)
    for i in range(m):
        if ind1[i]!=ind2[i]:
            nbfd += 1
    return nbfd/m


def matrice_matching(x_sol,x_train):
    """
    Computes the distance matrix (using manhattan distance) between two datasets (i.e., the distance between each pair of reconstructed and actual examples).
    """
    import numpy as np
    n = len(x_sol)
    Matrice_match = np.empty([n,n])
    for i in range(n):
        for j in range(n):
            Matrice_match[i][j] = dist_individus(x_sol[i],x_train[j])
    return Matrice_match

def average_error(x_sol,x_train):
    """
    Computes the average reconstruction error between the proposed reconstruction x_sol and the actual training set x_train.
    Both must have the same shape.
    As described in our paper, we first perform a minimum cost matching to determine which reconstructed example corresponds to which actual example.
    We then compute the average error over all attributes of all (matched) examples and return it.
    """
    import numpy as np
    assert(np.asarray(x_sol).shape == np.asarray(x_train).shape)
    from scipy.optimize import linear_sum_assignment
    cost = matrice_matching(x_sol,x_train)
    row_ind, col_ind = linear_sum_assignment(cost)
    moyenne = 0
    for i in range(len(x_train)):
        moyenne += dist_individus(x_sol[i], x_train[col_ind[i]])
    moyenne = moyenne/len(x_train)
    return moyenne, col_ind.tolist()

def generate_random_sols(N,M, dataset_ohe_groups=[], n_sols=10, seed=42):
    """
    Generates n_sols random reconstructions of shape (N,M) that conform with the one-hot encoding information provided through dataset_ohe_groups.
    """
    import numpy as np
    np.random.seed(seed)
    randlist = []
    for i in range(n_sols):
        temporary_random = np.random.randint(2,size = (N,M))
        for j in range(N):
            for w in dataset_ohe_groups:
                list_draw = [1] + [0]*(len(w) - 1) # exactly one zero
                drawn = np.random.choice(np.array(list_draw), len(list_draw), replace=False) # random order
                for drawn_index, w_index in enumerate(w):
                    temporary_random[j][w_index] = drawn[drawn_index]

        randlist.append(temporary_random.tolist())
    return randlist

def check_ohe(X, ohe_vectors, verbose = True):
    ''' 
    Debugging function: use to check whether the stated one-hot encoding is verified on a given dataset

    Arguments
    ---------
    X: np array of shape [n_examples, n_attriutes]
        The one-hot encoded dataset to be verified
    
    ohe_vectors: list, optional
        List of lists, where each sub-list contains the IDs of a group of attributes corresponding to a one-hot encoding of the same original feature

    verbose: boolean, optional (default True)
        If an example for which the encoding is not correct is found, whether to print it or not

    Returns
    -------
    output: boolean
        False if an example for which the provided one-hot encoding is not verified 
            (i.e., for some subgroup of binary features one-hot encoding the same original attribute, their sum is not 1)
        True otherwise
    '''
    for a_ohe_group in ohe_vectors:
        for an_example in range(X.shape[0]):
            check_sum = sum(X[an_example][a_ohe_group])
            if check_sum != 1:
                if verbose:
                    print("Found non-verified OHE: example %d, ohe group: " %(an_example), a_ohe_group)
                    print("Example is: ", X[an_example], "with incorrect subset: ", X[an_example][a_ohe_group])
                return False
    return True