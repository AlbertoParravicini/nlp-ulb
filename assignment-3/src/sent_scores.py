"""
Script that handles multi-class classification, using the sentiment scores given by VADER.

Different models were tried, with a simple logistic regression giving the best results.
"""



#%% IMPORT STUFF

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV

from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import ElasticNet, LogisticRegressionCV, RidgeClassifierCV, SGDClassifier
from sklearn.preprocessing import StandardScaler

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB

from sklearn.svm import SVC

from bokeh.plotting import figure, output_file, show
from bokeh.io import gridplot
from bokeh.charts import Bar

#%% LOAD DATA

df_sent = pd.read_hdf("../data/df_sent_large.h5")
df_sent.drop('original_text', axis=1, inplace=True)

#%% PLOT SCORE DISTRIBUTION


output_file("plot.html")

# COMPOUND
df_sent_melt = pd.melt(df_sent, id_vars=["score", "text"], value_vars=["compound", "compound_o"], var_name='type', value_name='value')

p1 = Bar(df_sent_melt, label=["score"], values="value", 
        title="Compound sentiment scores, grouped by review score", width=800, color='type', agg="mean", group="type")
p1.title.text_font_size = "18pt"
p1.xaxis.axis_label_text_font_size = "14pt"
p1.xaxis.major_label_text_font_size = "14pt"
p1.yaxis.axis_label_text_font_size = "14pt"
p1.yaxis.major_label_text_font_size = "14pt"
# NEGATIVE
df_sent_melt = pd.melt(df_sent, id_vars=["score", "text"], value_vars=["neg", "neg_o"], var_name='type', value_name='value')

p2 = Bar(df_sent_melt, label=["score"], values="value", 
        title="Negative sentiment scores, grouped by review score", width=800, color='type', agg="mean", group="type")
p2.title.text_font_size = "18pt"
p2.xaxis.axis_label_text_font_size = "14pt"
p2.xaxis.major_label_text_font_size = "14pt"
p2.yaxis.axis_label_text_font_size = "14pt"
p2.yaxis.major_label_text_font_size = "14pt"
# NEUTRAL
df_sent_melt = pd.melt(df_sent, id_vars=["score", "text"], value_vars=["neu", "neu_o"], var_name='type', value_name='value')

p3 = Bar(df_sent_melt, label=["score"], values="value", 
        title="Neutral sentiment scores, grouped by review score", width=800, color='type', agg="mean", group="type")
p3.title.text_font_size = "18pt"
p3.xaxis.axis_label_text_font_size = "14pt"
p3.xaxis.major_label_text_font_size = "14pt"
p3.yaxis.axis_label_text_font_size = "14pt"
p3.yaxis.major_label_text_font_size = "14pt"
# POSITIVE
df_sent_melt = pd.melt(df_sent, id_vars=["score", "text"], value_vars=["pos", "pos_o"], var_name='type', value_name='value')

p4 = Bar(df_sent_melt, label=["score"], values="value", 
        title="Positive sentiment scores, grouped by review score", width=800, color='type', agg="mean", group="type")
p4.title.text_font_size = "18pt"
p4.xaxis.axis_label_text_font_size = "14pt"
p4.xaxis.major_label_text_font_size = "14pt"
p4.yaxis.axis_label_text_font_size = "14pt"
p4.yaxis.major_label_text_font_size = "14pt"
p = gridplot([[p1, p3], [p2, p4]])
show(p)

# END OF PLOTS



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


#%% SGD: train a classifier using stochastic gradient descent.
# "hinge" gives an SVM, "log" gives a logistic regression.
# Many different parameters are possible,
# below there is just a small sample of what was tested.
# Still, the results are worse than the standard logistic regression.

model = SGDClassifier()

param_grid = {
                "loss": ["hinge", "log"],
                "penalty": ["l2"],
                "alpha": [10**-1],
                "class_weight": ["balanced", None]
                
             }
 
grid_sgd = GridSearchCV(model, param_grid, cv=4, verbose=2, n_jobs=1)
grid_sgd.fit(x_train, y_train)
    
print("\n-------- BEST ESTIMATOR --------\n")
print(grid_sgd.best_estimator_)
print("\n-------- BEST PARAMS --------\n")
print(grid_sgd.best_params_)
print("\n-------- BEST SCORE --------\n")
print(grid_sgd.best_score_)
df_res = pd.DataFrame(grid_sgd.cv_results_)

#%% TRAIN SGD

model.fit(x_train, y_train)

#%% PREDICT WITH SGD

pred = model.predict(x_test)

print("ACCURACY:", sum(pred == y_test) / len(pred))
print('RMSE: {}'.format(np.sqrt(mean_squared_error(pred, y_test))))

pd.Series(pred).value_counts()
