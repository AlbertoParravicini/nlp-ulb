#%% IMPORT STUFF

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update(plt.rcParamsDefault)
plt.style.use('seaborn-pastel')

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.svm import SVC

from sklearn.metrics import r2_score, mean_squared_error

from bokeh.plotting import figure, output_file, show
from bokeh.charts import Bar


import spacy
nlp = spacy.load('en')

#%% LOAD DATA

bin_df = pd.read_hdf("../data/compressed_large.h5")


#%% BUILD TRAIN AND TEST SETS
test_size = 0.2

X = bin_df.iloc[:, :-1].as_matrix()
y = bin_df.iloc[:, -1].as_matrix()

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
kfolds = StratifiedKFold(n_splits=10)

#%% BUILD A NAIVE BAYES MODEL; EVALUATE IT WITH KFOLD XVALIDATION
alphas = [3]
for a in alphas:
    model = MultinomialNB(alpha=3)
    
    scores = cross_val_score(model, x_train, y_train, cv=kfolds, n_jobs=1, verbose=2)
    print("ALPHA:", a, " -- ", np.mean(scores))

#%% TRAIN NAIVE BAYES 

model.fit(x_train, y_train)

#%% PREDICT WITH NAIVE BAYES

pred = model.predict(x_test)

print("ACCURACY:", sum(pred == y_test) / len(pred))
print('RMSE: {}'.format(np.sqrt(mean_squared_error(pred, y_test))))

#%% SAVE PRED
pred_table = pd.DataFrame({"pred": pred, "real": y_test})
pred_table.to_csv("../data/prediction_nb_large.csv")

pred_table_melt = pd.melt(pred_table, value_vars=["pred", "real"], var_name='type', value_name='value')

p = Bar(pred_table_melt, label="value", group="type", color="type",
        title="Predicted sentiment scores distribution, versus real values", width=800)
show(p)

#%% SVM: used with the dense vectors. 

model = SVC()
#scores = cross_val_score(model, x_train, y_train, cv=kfolds, n_jobs=1, verbose=2)

#print(np.mean(scores))


param_grid = {
                "C": [0.1],
                "gamma": [1/1000]
             }
 
grid_svm = GridSearchCV(model, param_grid, cv=4, verbose=2, n_jobs=1)
grid_svm.fit(x_train, y_train)
    
print("\n-------- BEST ESTIMATOR --------\n")
print(grid_svm.best_estimator_)
print("\n-------- BEST PARAMS --------\n")
print(grid_svm.best_params_)
print("\n-------- BEST SCORE --------\n")
print(grid_svm.best_score_)

#%% TRAIN SVM

model.fit(x_train, y_train)

#%% PREDICT WITH SVM

pred = model.predict(x_test)

print("ACCURACY:", sum(pred == y_test) / len(pred))
print('RMSE: {}'.format(np.sqrt(mean_squared_error(pred, y_test))))
