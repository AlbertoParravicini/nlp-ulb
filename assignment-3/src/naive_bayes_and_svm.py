"""
Script that handles multi-class classification, using sparse or dense embeddings.
It is recommended to use Naive Bayes with sparse embeddings, ot SVM with dense embeddings.

Different models were tried, with the 2 above giving the best results.
"""


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

import itertools
from sklearn.metrics import confusion_matrix

import spacy
#%%
nlp = spacy.load('en')

#%% LOAD DATA

bin_df = pd.read_hdf("../data/bin_df.h5")



#%% BUILD TRAIN AND TEST SETS
test_size = 0.2

bin_df.insert(0, "id", bin_df.index)

X = bin_df.iloc[:, :-1].as_matrix()
y = bin_df.iloc[:, -1].as_matrix()

#%%
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
#%%
test_index = x_test[:, 0]

x_train = x_train[:, 1:]
x_test = x_test[:, 1:]
kfolds = StratifiedKFold(n_splits=10)



#%% BUILD A NAIVE BAYES MODEL; EVALUATE IT WITH KFOLD XVALIDATION
alphas = [3]
for a in alphas:
    model = MultinomialNB(alpha=3)
    
    scores = cross_val_score(model, x_train, y_train, cv=kfolds, n_jobs=1, verbose=2)
    print("ALPHA:", a, " -- ", np.mean(scores))

#%% TRAIN NAIVE BAYES 
model = MultinomialNB(alpha=3)
model.fit(x_train, y_train)

#%% PREDICT WITH NAIVE BAYES

pred = model.predict(x_test)

print("ACCURACY:", sum(pred == y_test) / len(pred))
print('RMSE: {}'.format(np.sqrt(mean_squared_error(pred, y_test))))

#%% SAVE PRED
pred_table = pd.DataFrame({"id": test_index, "pred": pred, "real": y_test})
pred_table.to_csv("../data/prediction_nb.csv")

pred_table_melt = pd.melt(pred_table, value_vars=["pred", "real"], var_name='type', value_name='value')

p = Bar(pred_table_melt, label="value", group="type", color="type",
        title="Predicted sentiment scores distribution, versus real values", width=800)
show(p)

#%% SVM: used with the dense vectors. 
# UNCOMMENT DEPENDING ON WHICH DATASET IS USED.

#model = SVC()
##scores = cross_val_score(model, x_train, y_train, cv=kfolds, n_jobs=1, verbose=2)
#
##print(np.mean(scores))
#
#
#param_grid = {
#                "C": [0.1],
#                "gamma": [1/1000]
#             }
# 
#grid_svm = GridSearchCV(model, param_grid, cv=4, verbose=2, n_jobs=1)
#grid_svm.fit(x_train, y_train)
#    
#print("\n-------- BEST ESTIMATOR --------\n")
#print(grid_svm.best_estimator_)
#print("\n-------- BEST PARAMS --------\n")
#print(grid_svm.best_params_)
#print("\n-------- BEST SCORE --------\n")
#print(grid_svm.best_score_)
#
##%% TRAIN SVM
#
#model.fit(x_train, y_train)
#
##%% PREDICT WITH SVM
#
#pred = model.predict(x_test)
#
#print("ACCURACY:", sum(pred == y_test) / len(pred))
#print('RMSE: {}'.format(np.sqrt(mean_squared_error(pred, y_test))))



#%% CONFUSION MATRIX
# Code from http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, list(range(1,6)),
                      title='Confusion matrix')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=list(range(1,6)), normalize=True,
                      title='Normalized confusion matrix')

plt.show()

#%% Consider score=5 as "true" class, and compute metrics.

pred_5 = pd.DataFrame({"pred": [1 if x == 5 else 0 for x in pred], "real": [1 if x == 5 else 0 for x in y_test]})

cm = confusion_matrix(pred_5.real, pred_5.pred)
tp = cm[1, 1]
tn = cm[0, 1]
fn = cm[1, 0]
fp = cm[0, 1]

plt.figure()
plot_confusion_matrix(cm, classes=list(range(2)), normalize=False,
                      title='Confusion matrix, for score 5')

plt.show()

accuracy = (tp + tn) / (tp + tn + fp + fn)
sensitivity = (tp) / (tp + fn)
specificity = tn / (tn + fp)
bal_acc = 1 - 0.5 * (fp / (tn + fp) + fn / (tp + fn))
err = (fp + fn) / (tp + tn + fp + fn)
precision = tp / (tp + fp)
