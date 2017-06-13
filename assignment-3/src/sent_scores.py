#%% IMPORT STUFF

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV

from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import ElasticNet, LogisticRegressionCV, RidgeClassifierCV
from sklearn.preprocessing import StandardScaler

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB

from sklearn.svm import SVC

from bokeh.plotting import figure, output_file, show
from bokeh.charts import Bar

#%% LOAD DATA

df_sent = pd.read_hdf("../data/df_sent.h5")
df_sent.drop('original_text', axis=1, inplace=True)

#%% PLOT SCORE DISTRIBUTION


output_file("plot.html")
p = Bar(df_sent, label='score', values='pos', agg='mean',
        title="Average Sentiment Scores")

# COMPOUND
df_sent_melt = pd.melt(df_sent, id_vars=["score", "text"], value_vars=["compound", "compound_o"], var_name='type', value_name='value')

p = Bar(df_sent_melt, label=["score"], values="value", 
        title="Compound sentiment scores, grouped by review score", width=800, color='type', agg="mean", group="type")
show(p)

# NEGATIVE
df_sent_melt = pd.melt(df_sent, id_vars=["score", "text"], value_vars=["neg", "neg_o"], var_name='type', value_name='value')

p = Bar(df_sent_melt, label=["score"], values="value", 
        title="Negative sentiment scores, grouped by review score", width=800, color='type', agg="mean", group="type")
show(p)

# NEUTRAL
df_sent_melt = pd.melt(df_sent, id_vars=["score", "text"], value_vars=["neu", "neu_o"], var_name='type', value_name='value')

p = Bar(df_sent_melt, label=["score"], values="value", 
        title="Neutral sentiment scores, grouped by review score", width=800, color='type', agg="mean", group="type")
show(p)

# POSITIVE
df_sent_melt = pd.melt(df_sent, id_vars=["score", "text"], value_vars=["pos", "pos_o"], var_name='type', value_name='value')

p = Bar(df_sent_melt, label=["score"], values="value", 
        title="Positive sentiment scores, grouped by review score", width=800, color='type', agg="mean", group="type")
show(p)

#%% BUILD TRAIN AND TEST SETS
test_size = 0.2

X = df_sent.iloc[:, 2:].as_matrix()
y = df_sent.iloc[:, 0].as_matrix()

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
kfolds = StratifiedKFold(n_splits=10)


# Scale data
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)



#%% BUILD A GLM; EVALUATE IT WITH KFOLD XVALIDATION
model = LogisticRegressionCV(Cs=20, cv=10,  solver="lbfgs")

#scores = cross_val_score(model, x_train, y_train, cv=kfolds, n_jobs=1, verbose=2)
#print(np.mean(scores))

model.fit(x_train, y_train)
print(model.C_)
model.score(x_train, y_train)

#%% TRAIN GLM

model.fit(x_train, y_train)
#%% PREDICT WITH GLM

pred = model.predict(x_test)

print("ACCURACY:", sum(np.ceil(pred) == y_test) / len(pred))
print('RMSE: {}'.format(np.sqrt(mean_squared_error(pred, y_test))))
pd.Series(pred).value_counts()


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

pd.Series(pred).value_counts()
