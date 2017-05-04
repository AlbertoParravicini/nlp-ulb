# Use the features built by the hand-made oracle  
# to train a decision tree. 


#%% IMPORT MODULES

import pandas as pd
import pydotplus 
from enum import Enum
from sklearn import tree, preprocessing
import pickle

#%% PARSER ACTION ENUMERATION

class Action(Enum):
    LEFT = 0
    RIGHT = 1
    SHIFT = 2
    

if __name__ == "__main__":
    lines = pd.read_csv("../data/feattemp.txt", sep=", ")
       
    X = lines.iloc[:, :-1]
    Y = lines.iloc[:, -1]
    
    # 1-hot encoding
    X_t = pd.get_dummies(X, columns = X.columns)
    # Encode targets.
    le = preprocessing.LabelEncoder()
    le.fit(list(set(Y)))
    Y_t = le.transform(Y)
    
    #%% TRAIN TREE
    clf = tree.DecisionTreeClassifier()
    
    clf = clf.fit(X_t, Y_t)
    
    #%% SAVE THE MODEL
    with open('tree.pickle', 'wb') as f:
        pickle.dump(clf, f)
    with open("enc.pickle", "wb") as f:
        pickle.dump(le, f)
    with open("1h.pickle", "wb") as f:
        pickle.dump(X.columns, f)
    
    
    #%% DRAW TREE
    dot_data = tree.export_graphviz(clf, out_file=None,
                         feature_names=X_t.columns,  
                         class_names=Y,  
                         filled=True, rounded=True, label="root" ,
                         special_characters=True, impurity=False)  
    graph = pydotplus.graph_from_dot_data(dot_data) 
    graph.write_pdf("tree.pdf") 
