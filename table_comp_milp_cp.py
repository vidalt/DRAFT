import json
import matplotlib.pyplot as plt
import pandas as pd
from os.path import exists
import numpy as np 
from matplotlib.lines import Line2D


dataset= "compas"
bagging=False

for method in["cp-sat", "milp"]:

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

        mean_error_gs = []
        solve_time_gs = []
        random_error = []
        parameters = []
        missing_cnt = 0
        longest_time_res = -1

        for params in params_list:
            '''if int(params[0]) >= 90 and (params[1] == 'None' or params[1] == '10'):
                continue'''
            filename = str(dataset) + "_" + str(params[0]) +"_"+ str(params[1])+ "_" + str(seed) + "_bagging-" + str(bagging) + "_" + str(method)

            path_to_file = f"./{folder}/{filename}.json"
            file_exists = exists(path_to_file)
            if file_exists:

                f = open(path_to_file)
                data = json.load(f)
                metrics = data["values"]["mean-error"]
                time_res = data["values"]["solve_duration_time"]
                solve_status = data["values"]["solve_status"]
                random_baseline = data["values"]["random_error"]
                if time_res > longest_time_res:
                    longest_time_res = time_res
                parameters.append({'n_estimators': params[0], 'max_depth': str(params[1])})
                if solve_status == "UNKNOWN":
                    print("Oups, for this configuration the solver struggled: ", str(params[0]) + " trees, max. depth " + str(params[1]) + ", seed " + str(seed) + "(time elapsed: " + str(time_res) + ")")
                    mean_error_gs.append(random_baseline)           
                else:
                    mean_error_gs.append(metrics)
                    solve_time_gs.append(time_res)
                
                random_error.append(random_baseline)

                    

            else :
                print("missing file %s" %path_to_file)
                missing_cnt +=1
                #parameters.append({'n_estimators': params[0], 'max_depth': str(params[1])})
                #mean_error_gs.append(None)
                #solve_time_gs.append(None)

        print("missing %d files" %missing_cnt)
        print("longest run:", longest_time_res, " seconds")
        gs_results_df = pd.DataFrame(parameters).fillna("None")
        gs_results_df["mean error"] = mean_error_gs
        gs_results_df["solve time"] = solve_time_gs
        gs_results_df["random error"] = random_error

        seed_results = {}
        seed_times = {}
        for a_depth in val_depths:
            sub_df = gs_results_df[gs_results_df["max_depth"]==a_depth]
            if show:
                plt.plot(sub_df["n_estimators"], sub_df["mean error"], c=random_colors[a_depth], label=str(a_depth), marker='x')
            seed_results[a_depth] = list(sub_df["mean error"])
            seed_times[a_depth] = list(sub_df["solve time"])

        random_baseline = gs_results_df.iloc[1]["random error"]
        for index, row  in gs_results_df.iterrows(): # just double checking
            assert(row["random error"] == random_baseline)

        seed_results['random_baseline'] = [random_baseline for _ in val_trees]
        if show:
            plt.plot(val_trees, [random_baseline for _ in val_trees], linestyle='dashed', label='Random Baseline')
            plt.legend(loc='best')
            plt.show()

        return seed_results, seed_times

        #px.line(gs_results_df, x= x, y="mean error", color=color, height=750).show()

    all_seeds_results = []
    all_seed_times = []
    # First plot per-fold results
    for seed in val_seed:
        local_results, local_times = plot_f_depth_single_seed(params_list, seed)
        all_seeds_results.append(local_results)
        all_seed_times.append(local_times)

    # Then compute averages
    average_results = {}
    std_results = {}
    average_times = {}
    std_times = {}
    times_per_max_depth = {}

    for one_depth_val in all_seeds_results[0].keys():
        depth_errors_list_avg = []
        depth_errors_list_std = []
        depth_times_list_avg = []
        depth_times_list_std = []
        if one_depth_val != 'random_baseline':
            times_per_max_depth[one_depth_val] = []
        for n_trees_index in range(len(val_trees)):
            acc_results_local = []
            depth_times_local = []
            for one_seed_results in all_seeds_results:
                try:
                    acc_results_local.append(one_seed_results[one_depth_val][n_trees_index])
                except IndexError:
                    continue
            for one_time_results in all_seed_times:
                if one_depth_val != 'random_baseline':
                        depth_times_local.append(one_time_results[one_depth_val][n_trees_index])
            if one_depth_val != 'random_baseline':
                times_per_max_depth[one_depth_val].extend(depth_times_local)
            depth_errors_list_avg.append(np.average(acc_results_local))
            depth_errors_list_std.append(np.std(acc_results_local))
            depth_times_list_avg.append(np.average(depth_times_local))
            depth_times_list_std.append(np.std(depth_times_local))

        average_results[one_depth_val] = depth_errors_list_avg
        std_results[one_depth_val] = depth_errors_list_std
        average_times[one_depth_val] = depth_times_list_avg
        std_times[one_depth_val] = depth_times_list_std
        if one_depth_val != 'random_baseline':
            assert(len(times_per_max_depth[one_depth_val]) == len(val_trees) * len(val_seed))

    import csv 
    with open('tables/comp_table_time_table_%s_%s.csv' %(dataset, method), 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(["=== Dataset " + str(dataset) + ", method " + str(method) + " ==="])
        csv_writer.writerow(["Max.depth", 'Reconstruction Times (s)', '', '', ''])
        csv_writer.writerow(["Max.depth", 'Avg', 'Std', 'Min', 'Max'])
        for one_depth_val in times_per_max_depth.keys():
            #print("Max. depth " + str(one_depth_val) + " avg solving time is " + str(np.average(times_per_max_depth[one_depth_val])) + ", std is " + str(np.std(times_per_max_depth[one_depth_val])) + "min is " + str(np.min(times_per_max_depth[one_depth_val])) + ", max is " + str(np.max(times_per_max_depth[one_depth_val])))    
            csv_writer.writerow([one_depth_val, "%.1f" %np.average(times_per_max_depth[one_depth_val]), "%.1f" %np.std(times_per_max_depth[one_depth_val]), "%.1f" %np.min(times_per_max_depth[one_depth_val]), "%.1f" %np.max(times_per_max_depth[one_depth_val])])

    '''
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
    plt.ylabel("Reconstruction Error")
    ax = plt.gca()
    #ax.set_ylim([0.0, 0.25])
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
    plt.clf()
    '''