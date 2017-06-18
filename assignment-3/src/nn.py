"""
Script that handles multi-class classification, using the sentiment scores given by VADER.
It uses Keras
"""



#%% IMPORT STUFF

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV

from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

from keras.layers import Dense, Dropout, BatchNormalization, Activation
from keras.models import Sequential
from keras.regularizers import l1
from keras.callbacks import ModelCheckpoint

from bokeh.plotting import figure, output_file, show
from bokeh.io import gridplot
from bokeh.charts import Bar
import keras.utils as utils

#%% LOAD DATA

df_sent = pd.read_hdf("../data/df_sent_large.h5")
df_sent.drop('original_text', axis=1, inplace=True)


#%% BUILD TRAIN AND TEST SETS
X = df_sent.iloc[:, 2:].as_matrix()
y = df_sent.iloc[:, 0].as_matrix()

# Convert labels to categorical one-hot encoding
y_bin = utils.to_categorical(y-1, num_classes=5)

X = StandardScaler().fit_transform(X)
X_tr, X_val, y_tr, y_val = train_test_split(X, y_bin, random_state = 3)
#%% BUILD NN

model = Sequential()
model.add(Dense(40, input_shape = (X_tr.shape[1], ), kernel_regularizer=l1(0.001)))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dense(y_bin.shape[1], activation='softmax'))
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

#%% TRAIN NN
filepath="../models/best-weights.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

hist = model.fit(X_tr, y_tr, validation_data = (X_val, y_val), epochs=2000, batch_size=32, callbacks=callbacks_list)


