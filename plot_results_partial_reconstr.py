import json
import matplotlib.pyplot as plt
import pandas as pd
from os.path import exists
import numpy as np 
from matplotlib.lines import Line2D

figures_sizes = (7.0,4.0)
 #"milp" or "cp-sat" or "bench"
for mode in ["attributes", "examples"]:
    method="bench_partial_%s"  %mode
    for dataset in ["compas", "default_credit", "adult"]:
        for bagging in [True]:
            #if method == "milp":
            #   if bagging or not(dataset == "compas"):
            #       continue
            print("==== EXPERIMENT: " + str(dataset) + " " + str(mode) + " reconstr. ====")
            # Experiment (locate the right folder)
            folder = "results_partial_%s" %mode

            # Combinations of hyperparameters
            val_trees = [100]
            val_depths = ['None']
            val_seed = [i for i in range(0,5)]
            params_list = []

            if mode == "examples":
                known_proportion_list = [0.0, 0.01, 0.05]
                known_proportion_list.extend([np.round(0.1*b,2) for b in range(1,10)])
                known_proportion_list.extend([0.95, 0.99])
                for t in val_trees:
                    for d in val_depths:
                        for kp in known_proportion_list:
                            params_list.append([t, d, kp])
                method_param_list = known_proportion_list
            elif mode == "attributes":
                nattrs_datasets = {"compas":14-4-3, "adult":19-5, "default_credit":21-3-2} # All attributes OHEncoding the same original feature count only once
                for t in val_trees:
                    for d in val_depths:
                                for nattrs in range(nattrs_datasets[dataset]):
                                    params_list.append([t, d, nattrs])
                method_param_list = range(nattrs_datasets[dataset])

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

            '''val_depths.remove('None')
            val_depths.remove('10')'''


        
            def plot_f_depth_single_seed(params_list, seed, show=False):

                results_dict = {}
                results_dict_random = {}
                missing_cnt = 0
                unknown_cnt = 0
                longest_time_res = -1

                for params in params_list:
                    '''if int(params[0]) >= 90 and (params[1] == 'None' or params[1] == '10'):
                        continue'''

                    n_estimators = params[0]
                    max_depth = params[1]
                    method_param = params[2]

                    filename = str(dataset) + "_" + str(n_estimators) +"_"+ str(max_depth)+ "_" + str(seed) + "_bagging-" + str(bagging) + "_" + str(method) + "_" + str(method_param)
                    path_to_file = f"./{folder}/{filename}.json"

                    file_exists = exists(path_to_file)
                    if file_exists:
                        f = open(path_to_file)
                        data = json.load(f)

                        mean_error = data["values"]["mean-error"]  
                        solve_duration = data["values"]["solve_duration_time"]
                        solve_status = data["values"]["solve_status"]
                        random_baseline = data["values"]["random_error"] 

                        if solve_duration > longest_time_res:
                            longest_time_res = solve_duration

                        if solve_status == "UNKNOWN": # Then ignore the result
                            #print("Oups, for this configuration the solver struggled: ", str(n_estimators) + " trees, max. depth " + str(max_depth) + ", seed " + str(seed) + "(time elapsed: " + str(solve_duration) + ")")
                            unknown_cnt+=1
                        else:
                            results_dict[method_param] = mean_error
                        results_dict_random[method_param] = random_baseline
            
                    else :
                        #print("missing file %s" %path_to_file)
                        missing_cnt +=1

                '''if missing_cnt > 0:
                    print("[seed:" + str(seed) + "] missing %d files" %missing_cnt)
                if unknown_cnt > 0:
                    print("[seed:" + str(seed) + "] ignored %d UNKNOWN runs" %unknown_cnt)'''
                #print("longest run:", longest_time_res, " seconds")
                
                return results_dict, results_dict_random

            all_seeds_results = []
            # First plot per-fold results
            for seed in val_seed:
                local_results, local_results_random = plot_f_depth_single_seed(params_list, seed)
                all_seeds_results.append({"method":local_results, "random":local_results_random})

            # Then compute averages
            errors_list_avg = []
            errors_list_std = []
            random_errors_list_avg = []
            random_errors_list_std = []
            method_param_list_definitive = []
            for unknown_val in method_param_list: # iterate over n_estimators (x axis, i.e., #trees)
                acc_results_local = []
                random_acc_results_local = []
                for one_seed_results in all_seeds_results:
                    try:
                        random_acc_results_local.append(one_seed_results["random"][unknown_val])
                        acc_results_local.append(one_seed_results["method"][unknown_val])
                    except KeyError:
                        continue
                if len(acc_results_local) < 2:
                    print("Dataset " + dataset + ", removing x=" + str(unknown_val) + "(not enough value)")
                    continue
                else:
                    #assert(len(random_acc_results_local) == len(val_seed))
                    errors_list_avg.append(np.average(acc_results_local))
                    errors_list_std.append(np.std(acc_results_local))
                    random_errors_list_avg.append(np.average(random_acc_results_local))
                    random_errors_list_std.append(np.std(random_acc_results_local))
                    method_param_list_definitive.append(unknown_val)

            plt.figure(figsize=figures_sizes)
            # Accuracy plot
            plt.plot(method_param_list_definitive, errors_list_avg,c="#41BA05") #label='max depth'+one_depth_val+"(average & std)",
            plt.fill_between(method_param_list_definitive, np.asarray(errors_list_avg) - np.asarray(errors_list_std), np.asarray(errors_list_avg) + np.asarray(errors_list_std), color="#41BA05", alpha=0.2)
            #plt.plot(method_param_list_definitive, random_errors_list_avg,c=random_colors["random_baseline"]) #label='max depth'+one_depth_val+"(average & std)",
            #plt.fill_between(method_param_list_definitive, np.asarray(random_errors_list_avg) - np.asarray(random_errors_list_std), np.asarray(random_errors_list_avg) + np.asarray(random_errors_list_std), color=random_colors["random_baseline"], alpha=0.2)

            if mode == "examples":
                plt.xlabel("Proportion of known examples")
            elif mode == "attributes":
                plt.xlabel("#known attributes")
            plt.ylabel("Reconstruction Error (for unknown %s)" %mode)
            ax = plt.gca()
            #ax.set_ylim([-0.01, 0.05])
            #plt.legend(loc='best')
            expe_suffix = dataset + '_' + method + '_bagging=' + str(bagging)
            plt.savefig('./figures/%s_average_acc.pdf' %expe_suffix, bbox_inches='tight')
            plt.clf()

            '''legendFig = plt.figure("Legend plot")
            legend_elements = []
            for n_trees in val_trees:
                legend_elements.append(Line2D([0], [0], marker=None, color=random_colors[n_trees], lw=1, label='#trees = '+str(n_trees))) # linestyle = 'None',
            val_depth = 'random_baseline'
            legend_elements.append(Line2D([0], [0], marker=None, color=random_colors[val_depth], lw=1, label='Random Baseline')) # linestyle = 'None',
            legendFig.legend(handles=legend_elements, loc='center', ncol=4)
            legendFig.savefig('./figures/average_acc_legend_partial.pdf', bbox_inches='tight')
            plt.clf()'''

            '''
            # Solving times plot
            for one_depth_val in all_seed_times[0].keys():
                val_trees_local = val_trees
                if len(average_times[one_depth_val]) < len(val_trees):
                    last_index = (len(val_trees)-len(average_times[one_depth_val]))
                    val_trees_local = val_trees[:-last_index]
                plt.plot(val_trees_local, average_times[one_depth_val],c=random_colors[one_depth_val]) #label='max depth'+one_depth_val+"(average & std)",
                plt.fill_between(val_trees_local, np.asarray(average_times[one_depth_val]) - np.asarray(std_times[one_depth_val]), np.asarray(average_times[one_depth_val]) + np.asarray(std_times[one_depth_val]), color=random_colors[one_depth_val], alpha=0.2)

            plt.xlabel("#trees")
            plt.ylabel("Solving time (s)")
            #plt.legend(loc='best')
            plt.savefig('./figures/%s_average_time.pdf' %expe_suffix, bbox_inches='tight')
            plt.clf()'''