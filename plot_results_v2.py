import json
import matplotlib.pyplot as plt
import pandas as pd
from os.path import exists
import numpy as np 
from matplotlib.lines import Line2D

figures_sizes = (7.0,4.0)
for method in ["milp", "cp-sat", "bench"]:
    for dataset in ["compas", "default_credit", "adult"]:
        for bagging in [True, False]:
            if method == "milp" and bagging: # bagging not supported for the MILP formulation
                continue
            elif method == 'bench' and not bagging: # bench (Appendix B) only uses bagging at it aims at studying it
                continue
            print("==== EXPERIMENT: " + str(dataset) + " " + str(bagging) + " ====")
            # Experiment (locate the right folder)
            folder = "results_%s" %method

            # Combinations of hyperparameters
            val_trees = [1, 5, 10, 20,30,40,50,60,70,80,90,100]
            val_depths = ['2', '3','4','5', '10', 'None']
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
                            if not n_estimators in results_dict.keys():
                                results_dict[n_estimators] = {}
                            results_dict[n_estimators][max_depth] = mean_error
                        
                        if "random_baseline" in results_dict[n_estimators].keys():
                            assert(random_baseline == results_dict[n_estimators]["random_baseline"])
                        else:
                            results_dict[n_estimators]["random_baseline"] = random_baseline
                

                    else :
                        print("missing file %s" %path_to_file)
                        missing_cnt +=1


                print("missing %d files" %missing_cnt)
                print("ignored %d UNKNOWN runs" %unknown_cnt)
                print("longest run:", longest_time_res, " seconds")
                
                return results_dict

            all_seeds_results = []
            # First plot per-fold results
            for seed in val_seed:
                local_results = plot_f_depth_single_seed(params_list, seed)
                all_seeds_results.append(local_results)

            # Then compute averages
            average_results = {}
            std_results = {}

            for one_depth_val in all_seeds_results[0][1].keys(): # iterate over each curve (random + different depth values)
                depth_errors_list_avg = []
                depth_errors_list_std = []
                for n_trees in val_trees: # iterate over n_estimators (x axis, i.e., #trees)
                    acc_results_local = []
                    for one_seed_results in all_seeds_results:
                        try:
                            acc_results_local.append(one_seed_results[n_trees][one_depth_val])
                        except KeyError:
                            continue
                    #assert(len(acc_results_local) == len(val_seed))
                    depth_errors_list_avg.append(np.average(acc_results_local))
                    depth_errors_list_std.append(np.std(acc_results_local))
                    #depth_times_list_avg.append(np.average(depth_times_local))
                    #depth_times_list_std.append(np.std(depth_times_local))

                average_results[one_depth_val] = depth_errors_list_avg
                std_results[one_depth_val] = depth_errors_list_std

            plt.figure(figsize=figures_sizes)
            # Accuracy plot
            for one_depth_val in all_seeds_results[0][1].keys():
                val_trees_local = val_trees
                if len(average_results[one_depth_val]) < len(val_trees):
                    last_index = (len(val_trees)-len(average_results[one_depth_val]))
                    print("depth " + str(one_depth_val) + " diff is " + str(last_index))
                    val_trees_local = val_trees[:-last_index]

                plt.plot(val_trees_local, average_results[one_depth_val],c=random_colors[one_depth_val]) #label='max depth'+one_depth_val+"(average & std)",
                plt.fill_between(val_trees_local, np.asarray(average_results[one_depth_val]) - np.asarray(std_results[one_depth_val]), np.asarray(average_results[one_depth_val]) + np.asarray(std_results[one_depth_val]), color=random_colors[one_depth_val], alpha=0.2)

            plt.xlabel("#trees")
            plt.ylabel("Reconstruction Error")
            ax = plt.gca()
            ax.set_ylim([-0.01, 0.35])
            #plt.legend(loc='best')
            plt.savefig('./figures/%s_average_acc.pdf' %expe_suffix, bbox_inches='tight')
            plt.clf()
            legendFig = plt.figure("Legend plot")
            legend_elements = []
            for val_depth in val_depths:
                legend_elements.append(Line2D([0], [0], marker=None, color=random_colors[val_depth], lw=1, label='Max. Depth '+str(val_depth))) # linestyle = 'None',
            val_depth = 'random_baseline'
            legend_elements.append(Line2D([0], [0], marker=None, color=random_colors[val_depth], lw=1, label='Random Baseline')) # linestyle = 'None',
            legendFig.legend(handles=legend_elements, loc='center', ncol=4)
            legendFig.savefig('./figures/average_acc_legend.pdf', bbox_inches='tight')
            plt.clf()

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