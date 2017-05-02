#%% IMPORT MODULES

import pandas as pd
import numpy as np
import random
import pydotplus 
from enum import Enum
import pydotplus
from sklearn import tree, preprocessing

#%% PARSER ACTION ENUMERATION

class Action(Enum):
    LEFT = 0
    RIGHT = 1
    SHIFT = 2
    

if __name__ == "__main__":
    lines = pd.read_csv("../data/features.txt", sep=", ")
    
        
    X = lines.iloc[:, :-1]
    Y = lines.iloc[:, -1]
    
    X_t = pd.get_dummies(X, columns = X.columns)
    
    le = preprocessing.LabelEncoder()
    le.fit(list(set(Y)))
    Y_t = le.transform(Y)
    #%%
    clf = tree.DecisionTreeClassifier()
    
    clf = clf.fit(X_t, Y_t)
    
    #%%
    dot_data = tree.export_graphviz(clf, out_file=None,
                         feature_names=X_t.columns,  
                         class_names=Y,  
                         filled=True, rounded=True, label="root" ,
                         special_characters=True, impurity=False)  
    graph = pydotplus.graph_from_dot_data(dot_data) 
    graph.write_pdf("tree.pdf") 
