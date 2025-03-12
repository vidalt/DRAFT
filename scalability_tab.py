import json
import matplotlib.pyplot as plt
import pandas as pd
from os.path import exists
import numpy as np 
from matplotlib.lines import Line2D


method="bench_scalability"
datasets = ["compas", "adult", "default_credit"]
n_estimators = 100
max_depth_t = None
val_dataset_size = [25, 50, 100, 200, 300, 400, 500, 750, 1000, 1500]#, 2000, 3000]
val_seed = [i for i in range(5)]
for dataset in datasets:
    if dataset == "adult":
        val_dataset_size = [25, 50, 100, 200, 300, 400, 500, 750]#, 2000, 3000]
    elif dataset == "default_credit":
        val_dataset_size = [25, 50, 100, 200, 300, 400, 500]#, 2000, 3000]
    bagging = False 
    folder = "results/results_%s" %method
    keys = ["reconstr_error", "random_error", "solve_time", "acc_train", "acc_test"] #"generalization_error"
    global_res_avg= {}
    global_res_min =  {}
    global_res_max=  {}
    global_res_std=  {}
    for k in keys:
        global_res_avg[k] = []
        global_res_min[k] = []
        global_res_max[k] = []
        global_res_std[k] = []

    for train_size in val_dataset_size:
        local_res = {}
        for k in keys:
            local_res[k] = []

        for seed in val_seed:
            filename = str(dataset) + "_" + str(n_estimators) +"_"+ str(max_depth_t)+ "_" + str(seed) + "_bagging-" + str(bagging) + "_" + str(method) + "_" + str(train_size)

            path_to_file = f"./{folder}/{filename}.json"
            file_exists = exists(path_to_file)
            if file_exists:
                f = open(path_to_file)
                data = json.load(f)
                #metrics = data["values"]["accuracy %s" %mode]
                local_res["reconstr_error"].append(data["values"]["mean-error"])
                local_res["random_error"].append(data["values"]["random_error"])
                local_res["solve_time"].append(data["values"]["solve_duration_time"])
                local_res["acc_train"].append(data["values"]["accuracy train"])
                local_res["acc_test"].append(data["values"]["accuracy test"])
                #local_res["generalization_error"].append(data["values"]["accuracy train"] - data["values"]["accuracy test"])
                assert(data["values"]["train_size"] == train_size)
            else :
                print("missing file %s" %path_to_file)
        for a_metric in global_res_avg.keys():
            global_res_avg[a_metric].append(np.average(local_res[a_metric]))
            global_res_min[a_metric].append(min(local_res[a_metric]))
            global_res_max[a_metric].append(max(local_res[a_metric]))
            global_res_std[a_metric].append(np.std(local_res[a_metric]))

    '''print(global_res_avg)
    print(global_res_min)
    print(global_res_max)
    print(global_res_std)'''

    from scipy.optimize import curve_fit

    def power(x, a, b, c):
        return a + b * x ** c

    popt, pcov = curve_fit(power, val_dataset_size, global_res_avg["solve_time"])

    best_func = "%.2f+%.2f*x**%.2f" %(popt[0], popt[1], popt[2])

    import csv 
    with open('tables/scalability_%s.csv' %(dataset), 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(["=== Dataset " + str(dataset) + ", method " + str(method) + " ==="])
        csv_writer.writerow(["", 'Reconstr. Error', '', 'Reconstruction', 'Times (s)', '', '', '', ''])
        csv_writer.writerow(["#examples", 'Avg', 'Std', 'Avg', 'Std', 'Min', 'Max', 'RF Error'])
        for i, train_size in enumerate(val_dataset_size):
            #print("Max. depth " + str(one_depth_val) + " avg solving time is " + str(np.average(times_per_max_depth[one_depth_val])) + ", std is " + str(np.std(times_per_max_depth[one_depth_val])) + "min is " + str(np.min(times_per_max_depth[one_depth_val])) + ", max is " + str(np.max(times_per_max_depth[one_depth_val])))
            csv_writer.writerow([train_size, "%.1f" %global_res_avg["reconstr_error"][i], "%.1f" %global_res_std["reconstr_error"][i], "%.1f" %global_res_avg["solve_time"][i], "%.1f" %global_res_std["solve_time"][i], "%.1f" %global_res_min["solve_time"][i], "%.1f" %global_res_max["solve_time"][i], "%.3f" %global_res_avg["acc_train"][i], "%.3f" %global_res_avg["acc_test"][i]])
        csv_writer.writerow(["best_fct_fit=", best_func])