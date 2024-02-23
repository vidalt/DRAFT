
from sklearn.tree import DecisionTreeClassifier
import matplotlib 
import re

for removed_feat in [0,1,2,3]:
    plot_extension = 'pdf'

    X = [[0,0,0,1],
        [1,0,0,0],
        [0,1,0,0],
        [1,0,1,1]]
    y = [0, 0, 1, 1]
    features = ["$f_1$", "$f_2$", "$f_3$", "$f_4$"] #, "$X_3$","$X_4$"

    for an_ex in X:
        del an_ex[removed_feat]
    del features[removed_feat]

    clf = DecisionTreeClassifier(random_state=42, max_depth=3, min_samples_leaf=1)
    clf.fit(X, y) 
    print("DT accuracy = ", clf.score(X, y))

    from sklearn.tree import plot_tree
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(8,5))

    plot_tree(clf, ax=ax, feature_names = features, fontsize=14)

    def replace_text(obj):
        if type(obj) == matplotlib.text.Annotation:
            txt = obj.get_text()
            txt = txt.split("\n")
            newtxt = ""
            first = True
            for line in txt:
                '''if " <= 0.5" in line:
                    line = line.replace(" <= 0.5","")'''
                if not ("gini" in line) and not("samples" in line):
                    if "value" in line:
                        #line = line.replace("value =", "per-class #examples:")
                        line = re.split('[ | ]', line)
                        line_els = [l.replace('[','').replace(']','').replace(',','') for l in line]
                        cards = line_els[line_els.index('=')+1:]
                        line = ''
                        for i, c in enumerate(cards):
                            if i > 0:
                                line += '\n'
                            line += 'Class %d:' %i + ' %s example' %c
                            if int(c) > 1:
                                line += 's'
                    if first:
                        newtxt = newtxt + line
                        first = False
                    else:
                        newtxt = newtxt + "\n" + line
            obj.set_text(newtxt)
        return obj
        
    ax.properties()['children'] = [replace_text(i) for i in ax.properties()['children']]

    fig.savefig("./figures/tree_toy_example_no_%d.%s" %(removed_feat+1, plot_extension), bbox_inches='tight')
    plt.clf()

    def retrieve_branches(number_nodes, children_left_list, children_right_list, nodes_features_list, nodes_value_list):
        """Retrieve decision tree branches"""
        
        # Calculate if a node is a leaf
        is_leaves_list = [(False if cl != cr else True) for cl, cr in zip(children_left_list, children_right_list)]
        
        # Store the branches paths
        paths = []
        
        for i in range(number_nodes):
            if is_leaves_list[i]:
                # Search leaf node in previous paths
                end_node = [path[-1] for path in paths]

                # If it is a leave node yield the path
                if i in end_node:
                    output = paths.pop(np.argwhere(i == np.array(end_node))[0][0])
                    output = output[:-1]
                    yield (output, list(nodes_value_list[i][0]))

            else:
                
                # Origin and end nodes
                origin, end_l, end_r = i, children_left_list[i], children_right_list[i]

                # Iterate over previous paths to add nodes
                for index, path in enumerate(paths):
                    if origin == path[-1]:
                        path[-1] = -nodes_features_list[origin]
                        paths[index] = path + [end_l]
                        path[-1] = nodes_features_list[origin]
                        paths.append(path + [end_r])

                # Initialize path in first iteration
                if i == 0:
                    paths.append([-nodes_features_list[i], children_left[i]])
                    paths.append([nodes_features_list[i], children_right[i]])


    t = clf.tree_


    n_nodes = t.node_count
    children_left = t.children_left # For all nodes in the tree, list of their left children (or -1 for leaves)
    children_right = t.children_right # For all nodes in the tree, list of their right children (or -1 for leaves)
    nodes_features = t.feature # For all nodes in the tree, list of their used feature (or -2 for leaves)
    nodes_value = t.value # For all nodes in the tree, list of their value (support for both classes)
    nodes_features += 1

    all_branches = list(retrieve_branches(n_nodes, children_left, children_right, nodes_features, nodes_value))
    print(all_branches)