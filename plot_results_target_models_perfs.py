import json
import matplotlib.pyplot as plt
import pandas as pd
from os.path import exists
import numpy as np 
from matplotlib.lines import Line2D


method="cp-sat"
figures_sizes = (7.0,5.0)

for mode in ['train', 'test']:
    for dataset in ["adult", "compas", "default_credit"]:
        for bagging in [True, False]:
            folder = "results/results_%s" %method

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
               
            expe_suffix = dataset + '_' + method + '_bagging=' + str(bagging)
            def plot_f_depth_single_seed(params_list, seed, show=False):

                mean_error_gs = []
                parameters = []
                missing_cnt = 0

                for params in params_list:
                    '''if int(params[0]) >= 90 and (params[1] == 'None' or params[1] == '10'):
                        continue'''
                    filename = str(dataset) + "_" + str(params[0]) +"_"+ str(params[1])+ "_" + str(seed) + "_bagging-" + str(bagging) + "_" + str(method)

                    path_to_file = f"./{folder}/{filename}.json"
                    file_exists = exists(path_to_file)
                    if file_exists:

                        f = open(path_to_file)
                        data = json.load(f)
                        metrics = data["values"]["accuracy %s" %mode]
                        parameters.append({'n_estimators': params[0], 'max_depth': str(params[1])})
                        mean_error_gs.append(metrics)   

                    else :
                        print("missing file %s" %path_to_file)
                        missing_cnt +=1
                        #parameters.append({'n_estimators': params[0], 'max_depth': str(params[1])})
                        #mean_error_gs.append(None)
                        #solve_time_gs.append(None)

                print("missing %d files" %missing_cnt)
                gs_results_df = pd.DataFrame(parameters).fillna("None")
                gs_results_df["mean error"] = mean_error_gs

                seed_results = {}
                for a_depth in val_depths:
                    sub_df = gs_results_df[gs_results_df["max_depth"]==a_depth]
                    seed_results[a_depth] = list(sub_df["mean error"])

                return seed_results

                #px.line(gs_results_df, x= x, y="mean error", color=color, height=750).show()

            all_seeds_results = []
            # First plot per-fold results
            for seed in val_seed:
                local_results = plot_f_depth_single_seed(params_list, seed)
                all_seeds_results.append(local_results)

            # Then compute averages
            average_results = {}
            std_results = {}

            for one_depth_val in all_seeds_results[0].keys():
                depth_errors_list_avg = []
                depth_errors_list_std = []
                for n_trees_index in range(len(val_trees)):
                    acc_results_local = []
                    for one_seed_results in all_seeds_results:
                        try:
                            acc_results_local.append(one_seed_results[one_depth_val][n_trees_index])                
                        except IndexError:
                            continue
                    depth_errors_list_avg.append(np.average(acc_results_local))
                    depth_errors_list_std.append(np.std(acc_results_local))

                average_results[one_depth_val] = depth_errors_list_avg
                std_results[one_depth_val] = depth_errors_list_std

            # Accuracy plot
            for one_depth_val in all_seeds_results[0].keys():
                val_trees_local = val_trees
                if len(average_results[one_depth_val]) < len(val_trees):
                    last_index = (len(val_trees)-len(average_results[one_depth_val]))
                    print("depth " + str(one_depth_val) + " diff is " + str(last_index))
                    val_trees_local = val_trees[:-last_index]

                plt.plot(val_trees_local, average_results[one_depth_val],c=random_colors[one_depth_val]) #label='max depth'+one_depth_val+"(average & std)",
                plt.fill_between(val_trees_local, np.asarray(average_results[one_depth_val]) - np.asarray(std_results[one_depth_val]), np.asarray(average_results[one_depth_val]) + np.asarray(std_results[one_depth_val]), color=random_colors[one_depth_val], alpha=0.2)

            plt.xlabel("#trees")
            plt.ylabel("Target Model Accuracy")
            ax = plt.gca()
            #ax.set_ylim([0.0, 0.25])
            #plt.legend(loc='best')
            plt.savefig('./figures/%s_target_models_perfs_%s.pdf' %(expe_suffix, mode), bbox_inches='tight')
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