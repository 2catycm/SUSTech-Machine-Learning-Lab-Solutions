# 画图

# 'os' module provides functions for interacting with the operating system 
# 'Numpy' is used for mathematical operations on large, multi-dimensional arrays and matrices
import numpy as np
# 'Pandas' is used for data manipulation and analysis
import pandas as pd
# 'Matplotlib' is a data visualization library for 2D and 3D plots, built on numpy
from matplotlib import pyplot as plt
# 'Seaborn' is based on matplotlib; used for plotting statistical graphics
import seaborn as sns
# to suppress warnings
import warnings

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn import tree
from sklearn.tree import _tree
from sklearn.base import is_classifier # 用于判断是回归树还是分类树
from dtreeviz.colors import adjust_colors # 用于分类树颜色（色盲友好模式）
import seaborn as sns #用于回归树颜色
from matplotlib.colors import Normalize # 用于标准化RGB数值
import graphviz # 插入graphviz库
import os

warnings.filterwarnings("ignore") 
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

def draw_tree(model, X, y,  feature_names, target_names):
    dt = model
    yvec, xmat, vnames = y, X, feature_names
    # categorical_dict = {i:t for i, t in enumerate(target_names)}
    categorical_dict = target_names
    
    # class_names = sorted(list(set(yvec)))
    class_names = target_names
    dot_data = classification_tree_to_dot(model, vnames, class_names, categorical_dict)
    graph = graphviz.Source(dot_data)  
    return graph

# def get_yvec_xmat_vnames(target, df):

#     yvec = df[target]

#     # 将拥有n个不同数值的变量转换为n个0/1的变量，变量名字中有"_isDummy_"作为标注
#     xmat = pd.get_dummies(df.loc[:, df.columns != target], prefix_sep = "_isDummy_")

#     vnames = xmat.columns

#     return yvec, xmat, vnames
# # 得到更好看的树！

# def get_categorical_dict(df):
#     # store all the values of categorical value
#     df_categorical = df.select_dtypes(include=['object', 'bool', 'category'])
#     categorical_dict = {}
#     for i in df_categorical.columns:
#         # store in descending order
#         categorical_dict[i]= sorted(list(set(df[i].astype('str'))))
#     return categorical_dict

def tree_to_dot(tree, target, df):
    """ 把树变成dot data,用于输入graphviz然后绘制
    
    参数
        tree: DecisionTree的输出
        target: 目标变量名字
        df: 表单

    输出
        graphvic_str: dot data
    """
    # get yvec, vnames and categorical_dict of the df
    yvec, xmat, vnames = get_yvec_xmat_vnames(target, df)
    categorical_dict = get_categorical_dict(df)

    if is_classifier(tree):
        # 如果是分类树
        # classes should be in descending order
        class_names = sorted(list(set(yvec)))
        return classification_tree_to_dot(tree, vnames, class_names, categorical_dict)
    else:
        return regression_tree_to_dot(tree, vnames, categorical_dict)

    
def classification_tree_to_dot(tree, feature_names, class_names, categorical_dict):
    """ 把分类树转化成dot data

    参数
        tree: DecisionTreeClassifier的输出
        feature_names: vnames, 除去目标变量所有变量的名字
        class_names: 目标变量所有的分类
        categorical_dict: 储存所有名称及分类的字典

    输出
        graphvic_str: the dot data
    """    
    tree_ = tree.tree_
    
    # store colors that distinguish discrete chunks of data
    if len(class_names) <= 10:
        # get the colorblind friendly colors
        color_palette = adjust_colors(None)['classes'][len(class_names)]
    else:
        color_palette = sns.color_palette("coolwarm",len(class_names)).as_hex()

    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    
    # initialize the dot data string
    graphvic_str = 'digraph Tree {node [shape=oval, penwidth=0.1, width=1, fontname=helvetica] ; edge [fontname=helvetica] ;'
    #print(graphvic_str)

    def recurse(node, depth, categorical_dict):
         # store the categorical_dict information of each side
        categorical_dict_L = categorical_dict.copy()
        categorical_dict_R = categorical_dict.copy()
        # non local statement of graphvic_str
        nonlocal graphvic_str
        # variable is not dummy by default
        is_dummy = False
        # get the threshold
        threshold = tree_.threshold[node]
        
        # get the feature name
        name = feature_name[node]
        # judge whether a feature is dummy or not by the indicator "_isDummy_"
        if "_isDummy_" in str(name) and name.split('_isDummy_')[0] in list(categorical_dict.keys()):
            is_dummy = True
            # if the feature is dummy, the threshold is the value following name
            name, threshold = name.split('_isDummy_')[0], name.split('_isDummy_')[1]
        
        # get the data distribution of current node
        value = tree_.value[node][0]
        # get the total amount
        n_samples = tree_.n_node_samples[node]
        # calculate the weight
        weights = [i/sum(value) for i in value]
        # get the largest class
        class_name = class_names[np.argmax(value)]
        
        # pair the color and weight
        fillcolor_str = ""
        for i, j in enumerate(color_palette): 
            fillcolor_str += j + ";" + str(weights[i]) + ":"
        fillcolor_str = '"' + fillcolor_str[:-1] + '"'
        
        if tree_.feature[node] != _tree.TREE_UNDEFINED: 
            # if the node is not a leaf
            graphvic_str += ('{} [style=wedged, label=<{}<br/>{}>, fillcolor ='+fillcolor_str+'] ;').format(node,n_samples,name)
            #print(('{} [style=wedged, label=<{}<br/>{}>, fillcolor ='+fillcolor_str+'] ;').format(node,n_samples,name))
            if is_dummy:
                # if the feature is dummy and if its total categories > 5
                categorical_dict_L[name] = [str(i) for i in categorical_dict_L[name] if i != threshold]
                categorical_dict_R[name] = [str(threshold)]
                if len(categorical_dict[name])>5:
                    # only show one category on edge
                    threshold_left = "not " + threshold
                    threshold_right = threshold
                else:
                    # if total categories <= 5, list all the categories on edge
                    threshold_left = ", ".join( categorical_dict_L[name])
                    threshold_right = threshold
            else:
                # if the feature is not dummy, then it is numerical
                threshold_left = "<="+ str(round(threshold,3))
                threshold_right = ">"+ str(round(threshold,3))
            graphvic_str += ('{} -> {} [labeldistance=2.5, labelangle=45, headlabel="{}"] ;').format(node,tree_.children_left[node],threshold_left)
            graphvic_str += ('{} -> {} [labeldistance=2.5, labelangle=-45, headlabel="{}"] ;').format(node,tree_.children_right[node],threshold_right)
            #print(('{} -> {} [labeldistance=2.5, labelangle=45, headlabel="{}"] ;').format(node,tree_.children_left[node],threshold_left))
            #print(('{} -> {} [labeldistance=2.5, labelangle=-45, headlabel="{}"] ;').format(node,tree_.children_right[node],threshold_right))

            recurse(tree_.children_left[node], depth + 1,categorical_dict_L)
            recurse(tree_.children_right[node], depth + 1,categorical_dict_R)
        else:
            # the node is a leaf
            graphvic_str += ('{} [shape=box, style=striped, label=<{}<br/>{}>, fillcolor ='+fillcolor_str+'] ;').format(node,n_samples,class_name)
            #print(('{} [shape=box, style=striped, label=<{}<br/>{}>, fillcolor ='+fillcolor_str+'] ;').format(node,n_samples,class_name))

    recurse(0, 1,categorical_dict)
    return graphvic_str + "}"

def regression_tree_to_dot(tree, feature_names, categorical_dict):
    """ 把回归树转换成dot data

    参数
        tree: DecisionTreeClassifier的输出
        feature_names: vnames, 除去目标变量所有变量的名字
        categorical_dict: 储存所有名称及分类的字典

    输出
        graphvic_str: the dot data
    """    
    # get the criterion of regression tree: mse or mae
    criterion = tree.get_params()['criterion']
    
    tree_ = tree.tree_
    
    value_list = tree_.value[:,0][:,0]
    
    # Normalize data to produce heatmap colors
    cmap = cm.get_cmap('coolwarm')
    norm = Normalize(vmin=min(value_list), vmax=max(value_list))
    rgb_values = (cmap(norm(value_list))*255).astype(int)
    hex_values = ['#%02x%02x%02x' % (i[0], i[1], i[2]) for i in rgb_values]
    
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    
    # initialize the dot data string
    graphvic_str = 'digraph Tree {node [shape=oval, width=1, color="black", fontname=helvetica] ;edge [fontname=helvetica] ;'
    #print(graphvic_str)

    def recurse(node, depth, categorical_dict):
        # store the categorical_dict information of each side
        categorical_dict_L = categorical_dict.copy()
        categorical_dict_R = categorical_dict.copy()
        # non local statement of graphvic_str
        nonlocal graphvic_str
        
        # variable is not dummy by default
        is_dummy = False
        # get the threshold
        threshold = tree_.threshold[node]
        
        # get the feature name
        name = feature_name[node]
        # judge whether a feature is dummy or not by the indicator "_isDummy_"
        if "_isDummy_" in str(name) and name.split('_isDummy_')[0] in list(categorical_dict.keys()):
            is_dummy = True
            # if the feature is dummy, the threshold is the value following name
            name, threshold = name.split('_isDummy_')[0], name.split('_isDummy_')[1]
        
        # get the regression value
        value = round(tree_.value[node][0][0],3)
        # get the impurity
        impurity = criterion+ "=" + str(round(tree_.impurity[node],3))
        # get the total amount
        n_samples = tree_.n_node_samples[node]

        
        # pair the color with node
        fillcolor_str = '"'+hex_values[node]+'"'

        if tree_.feature[node] != _tree.TREE_UNDEFINED: 
            # if the node is not a leaf
            graphvic_str += ('{} [style="filled", label=<{}<br/>{}<br/>{}>, fillcolor ='+fillcolor_str+'] ;').format(node,n_samples,impurity,name)
            #print(('{} [style="filled", label=<{}<br/>{}<br/>{}>, fillcolor ='+fillcolor_str+'] ;').format(node,n_samples,impurity,name))
            if is_dummy:
                # if the feature is dummy and if its total categories > 5
                categorical_dict_L[name] = [str(i) for i in categorical_dict_L[name] if i != threshold]
                categorical_dict_R[name] = [str(threshold)]
                
                if len(categorical_dict[name])>5:
                    # only show one category on edge
                    threshold_left = "not " + threshold
                    threshold_right = threshold
                else:
                    # if total categories <= 5, list all the categories on edge
                    threshold_left = ", ".join(categorical_dict_L[name])
                    threshold_right = threshold
            else:
                # if the feature is not dummy, then it is numerical
                threshold_left = "<="+ str(round(threshold,3))
                threshold_right = ">"+ str(round(threshold,3))
            graphvic_str += ('{} -> {} [labeldistance=2.5, labelangle=45, headlabel="{}"] ;').format(node,tree_.children_left[node],threshold_left)
            graphvic_str += ('{} -> {} [labeldistance=2.5, labelangle=-45, headlabel="{}"] ;').format(node,tree_.children_right[node],threshold_right)
            #print(('{} -> {} [labeldistance=2.5, labelangle=45, headlabel="{}"] ;').format(node,tree_.children_left[node],threshold_left))
            #print(('{} -> {} [labeldistance=2.5, labelangle=-45, headlabel="{}"] ;').format(node,tree_.children_right[node],threshold_right))

            recurse(tree_.children_left[node], depth + 1,categorical_dict_L)
            recurse(tree_.children_right[node], depth + 1,categorical_dict_R)
        else:
            # the node is a leaf
            graphvic_str += ('{} [shape=box, style=filled, label=<{}<br/>{}<br/>{}>, fillcolor ='+fillcolor_str+'] ;').format(node,n_samples,impurity,value)
            #print(('{} [shape=box, style=filled, label=<{}<br/>{}<br/>{}>, fillcolor ='+fillcolor_str+'] ;').format(node,n_samples,impurity,value))

    recurse(0, 1,categorical_dict)
    return graphvic_str + "}"

