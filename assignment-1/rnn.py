#%%

# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 15:10:54 2017

@author: albyr
"""

import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM
from keras.preprocessing import text

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
    lines = [re.sub(r"[\s]+", "__", x) for x in lines]
    # Also add a double underscore at the start and at the end of each sentence.
    lines = ["__" + x + "__" for x in lines]

    return lines

# LOAD DATA ############
########################
def load_data(version):
    filename = "./data/training." + version + ".txt"
    
    start_time = timeit.default_timer()
    with open(filename, encoding="utf8") as f:
        lines = f.readlines()
    
    lines = lines[:int(len(lines)/400)]
    lines = preprocess_string(lines)
    
    labels = [version] * len(lines) 
    
    end_time = timeit.default_timer()
    print("! -> EXECUTION TIME OF TEXT PREPROCESSING:", (end_time - start_time), "\n")
    
    print(lines[:20])
    print(labels[:20])
    return [lines, labels]
    
    
versions = ["GB", "US", "AU"]
lines = []
labels = []
for v in versions:
    [lines_temp, labels_temp] = load_data(v)
    lines += lines_temp
    labels += labels_temp
    
labels = [1 if x == "GB" else 0 for x in labels]
#%%

model = Sequential()
model.add(Embedding(27, output_dim=256))
model.add(LSTM(128))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

train_indices = np.random.choice(range(len(lines)), np.floor(len(lines)*0.8)).reshape(-1)
x_train = [lines[i] for i in train_indices]
y_train = [labels[i] for i in train_indices]

x_test = [lines[i] for i in list(set(range(len(lines))) - set(train_indices))]
y_test = [labels[i] for i in list(set(range(len(lines))) - set(train_indices))]

vocabulary=list(string.ascii_lowercase[:26] + "_")
v_dict = {char: num for num, char in enumerate(vocabulary)}

x_train_encoded = [[v_dict[c] for c in x] for x in x_train]
x_test_encoded = [[v_dict[c] for c in x] for x in x_test]
#%%

model.fit(x_train_encoded, y_train, batch_size=16, epochs=10)
score = model.evaluate(x_test_encoded, y_test, batch_size=16)