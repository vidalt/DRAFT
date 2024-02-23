import json
import matplotlib.pyplot as plt
import pandas as pd
from os.path import exists
import numpy as np 
from matplotlib.lines import Line2D

figures_sizes = (7.0,5.0)
method="cp-sat" #"milp" or "cp-sat" or "bench"
for dataset in ["compas", "default_credit", "adult"]:
    for bagging in [False, True]:
        #if method == "milp":
        #    if bagging or not(dataset == "compas"):
        #        continue
        print("==== EXPERIMENT: " + str(dataset) + " " + str(bagging) + " ====")
        # Experiment (locate the right folder)
        folder = "results/results_%s" %method

        # Combinations of hyperparameters
        val_trees = [1, 10, 30, 50, 80,100]
        val_depths = ['None']
        val_seed = [i for i in range(0,5)]

        params_list = []

        # Random generation of colors for the plots
        random_colors = {}
        import random
        from random import randint
        random.seed(40)
        colors_list = []
        n = len(val_depths) + len(val_trees) + 1
        for i in range(n):
            colors_list.append('#%06X' % randint(0, 0xFFFFFF))
        i = 0
        for max_depth in val_depths:
            random_colors[max_depth] = colors_list[i]
            i += 1
        random_colors["random_baseline"] = colors_list[i]
        i+=1
        for n_trees in val_trees:
            random_colors[n_trees] = colors_list[i]
            i += 1

        for t in val_trees:
            for d in val_depths:
                params_list.append([t, d])

        '''val_depths.remove('None')
        val_depths.remove('10')'''


        expe_suffix = dataset + '_' + method + '_bagging=' + str(bagging)
        def plot_f_depth_single_seed(params_list, seed, show=False):

            results_dict = {}
            missing_cnt = 0
            unknown_cnt = 0
            longest_time_res = -1

            for params in params_list:
                '''if int(params[0]) >= 90 and (params[1] == 'None' or params[1] == '10'):
                    continue'''
                
                n_estimators = params[0]
                max_depth = params[1]

                filename = str(dataset) + "_" + str(n_estimators) +"_"+ str(max_depth)+ "_" + str(seed) + "_bagging-" + str(bagging) + "_" + str(method)
                path_to_file = f"./{folder}/{filename}.json"

                file_exists = exists(path_to_file)
                if file_exists:
                    f = open(path_to_file)
                    data = json.load(f)

                    mean_error = data["values"]["solve_duration_time"]
                    solve_duration = data["values"]["solve_duration_time"]
                    solve_status = data["values"]["solve_status"]
                    random_baseline = data["values"]["random_error"]

                    if solve_duration > longest_time_res:
                        longest_solve_status = solve_status
                        longest_time_res = solve_duration

                    if solve_status == "UNKNOWN": # Then ignore the result
                        #print("Oups, for this configuration the solver struggled: ", str(n_estimators) + " trees, max. depth " + str(max_depth) + ", seed " + str(seed) + "(time elapsed: " + str(solve_duration) + ")")
                        unknown_cnt+=1
                    else:
                        if not n_estimators in results_dict.keys():
                            results_dict[n_estimators] = {}
                        results_dict[n_estimators][max_depth] = mean_error
                    
                    #if "random_baseline" in results_dict[n_estimators].keys():
                    #    assert(random_baseline == results_dict[n_estimators]["random_baseline"])
                    #else:
                        #results_dict[n_estimators]["random_baseline"] = random_baseline
            

                else :
                    print("missing file %s" %path_to_file)
                    missing_cnt +=1


            print("missing %d files" %missing_cnt)
            print("ignored %d UNKNOWN runs" %unknown_cnt)
            #print("longest run:", longest_time_res, " seconds, status", longest_solve_status)
            
            return results_dict

        all_seeds_results = []
        # First plot per-fold results
        for seed in val_seed:
            local_results = plot_f_depth_single_seed(params_list, seed)
            all_seeds_results.append(local_results)

        # Then compute averages
        average_results = {}
        std_results = {}
        max_results = {}
        timeouts = {}

        for n_trees in all_seeds_results[0].keys(): # iterate over each curve (random + different depth values)
            depth_errors_list_avg = []
            depth_errors_list_std = []
            depth_error_list_max = []
            for one_depth_val in val_depths: # iterate over n_estimators (x axis, i.e., #trees)
                acc_results_local = []
                for one_seed_results in all_seeds_results:
                    try:
                        acc_results_local.append(one_seed_results[n_trees][one_depth_val])
                    except KeyError:
                        continue
                #assert(len(acc_results_local) == len(val_seed))
                depth_errors_list_avg.extend(acc_results_local)
                depth_errors_list_std.extend(acc_results_local)
                depth_error_list_max.extend(acc_results_local)
                
                #depth_times_list_avg.append(np.average(depth_times_local))
                #depth_times_list_std.append(np.std(depth_times_local))

            average_results[n_trees] = np.average(depth_errors_list_avg)
            std_results[n_trees] = np.std(depth_errors_list_std)
            max_results[n_trees] = np.max(depth_error_list_max)
            timeouts[n_trees] = (len(val_depths)*len(val_seed)) - len(depth_error_list_max)

        import csv 
        with open('tables/time_table_None_depth_%s_%s_Bagging_%s.csv' %(dataset, method, str(bagging)), 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(["=== Dataset " + str(dataset) + ", method " + str(method) + " ==="])
            csv_writer.writerow(["", 'Reconstruction Times (s)', '', ''])
            csv_writer.writerow(["#trees", 'Avg', 'Std', 'Max', '#timeouts'])
            for n_trees in average_results.keys():
                #print("Max. depth " + str(one_depth_val) + " avg solving time is " + str(np.average(times_per_max_depth[one_depth_val])) + ", std is " + str(np.std(times_per_max_depth[one_depth_val])) + "min is " + str(np.min(times_per_max_depth[one_depth_val])) + ", max is " + str(np.max(times_per_max_depth[one_depth_val])))
                csv_writer.writerow([n_trees, "%.1f" %average_results[n_trees], "%.1f" %std_results[n_trees], "%.1f" %max_results[n_trees], timeouts[n_trees]])