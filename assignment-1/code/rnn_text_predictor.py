#%%
################################
# LSTM #########################
################################
# Just a test to see how an LSTM would handle text classification.
# Results are encouraging, but
# reaching good levels of accuracy takes many many epochs!
"""
Created on Mon Mar 20 15:10:54 2017

@author: albyr
"""

import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, MaxPool1D
from keras.layers import Embedding
from keras.layers import LSTM
from keras.preprocessing import text
from keras.callbacks import ModelCheckpoint

from keras.utils.np_utils import to_categorical

import timeit
import string
import re


import pip
installed_packages = pip.get_installed_distributions()
flat_installed_packages = [package.project_name for package in installed_packages]
if 'Unidecode' in flat_installed_packages:
    import unicodedata


##########################################
# TEXT PREPROCESSING #####################
##########################################

def preprocess_string(input_lines):
    # Put all sentences to lowercase.
    lines = [x.lower() for x in input_lines]
    # If the package "unidecode" is installed,
    # replace unicode non-ascii characters (e.g. accented characters) with their closest ascii alternative.
    if 'Unidecode' in flat_installed_packages:
        lines = [unicodedata.normalize("NFKD", x) for x in lines]
    # Remove any character except a-z and whitespaces.
    lines = [re.sub(r"([^\sa-z])+", "", x) for x in lines]
    # Remove whitespaces at the start and end of each sentence.
    lines = [x.strip() for x in lines]
    # Substitute single and multiple whitespaces with a double underscore.
    lines = [re.sub(r"[\s]+", "_", x) for x in lines]
    # Also add a double underscore at the start and at the end of each sentence.
    lines = ["__" + x + "__" for x in lines]
    
    return lines

##########################################
# LOAD DATA ##############################
##########################################

def load_data(version):
    filename = "../data/training." + version + ".txt"
    
    start_time = timeit.default_timer()
    with open(filename, encoding="utf8") as f:
        lines = f.readlines()
    
    lines = lines[:int(len(lines)/300)]
    lines = preprocess_string(lines)
    
    labels = [version] * len(lines) 
    
    end_time = timeit.default_timer()
    print("! -> EXECUTION TIME OF TEXT PREPROCESSING:", (end_time - start_time), "\n")
    
    return [lines, labels]
    
    
versions = ["GB", "US", "AU"]
lines = []
labels = []
for v in versions:
    [lines_temp, labels_temp] = load_data(v)
    lines += lines_temp
    labels += labels_temp
    
# max_length = max([len(x) for x in lines])
# # Pad each string to the max length
# lines = [x + "_" * (max_length - len(x)) for x in lines]
max_length = 100
lines = [x + "_" * (max_length - len(x)) if len(x) < max_length else x[:max_length] for x in lines]
    

#%%
# Turn lines and labels to numeric
print("turning input to numeric")
vocabulary=list(string.ascii_lowercase[:26] + "_")
v_dict = {char: num for num, char in enumerate(vocabulary)}


lines_num = [[v_dict[c] for c in x] for x in lines]
lines_to_predict_num = [list(np.roll(v, -1)) for v in lines_num]


# 1 hot encoding for the output labels
lines_to_predict_num_1_hot = [to_categorical(x, num_classes=None) for x in lines_to_predict_num]


#%%
# Build train and test sets
print("building train and test sets")
split_factor = 0.8
train_size = int(len(lines_num) * split_factor)
train_indices = np.random.choice(len(lines_num), train_size, replace=False)

train_set = [lines_num[i] for i in train_indices]
train_to_pred = [lines_to_predict_num_1_hot[i] for i in train_indices]

test_indices = np.array(list(set(range(len(lines_num))) - set(train_indices)))
test_set = [lines_num[i] for i in test_indices]
test_to_pred = [lines_to_predict_num_1_hot[i] for i in test_indices]

print(len(train_set), len(train_to_pred), len(test_set), len(test_to_pred))


train_set = np.reshape(train_set, (len(train_set), max_length, 1))
test_set = np.reshape(test_set, (len(test_set), max_length, 1))
train_to_pred_array = np.asarray(train_to_pred)

#%%
# create the model
print("creating the model")
model = Sequential()
model.add(LSTM(4, input_shape=(None,1)))
model.add(Dense(27, activation="softmax"))
model.compile(loss='categorical_crossentropy', optimizer='adam')
model.fit(train_set, train_to_pred_array, nb_epoch=100, batch_size=1, verbose=2)
print(model.summary())

# checkpoint
filepath="weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]


# Final evaluation of the model
scores = model.evaluate(test_set, test_to_pred, verbose=1)
print("Accuracy: %.2f%%" % (scores[1]*100))

filepath = "model_1_lstm.h5"
model.save(filepath)

