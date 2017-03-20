import pandas as pd
import numpy as np
import re
import timeit
import string

from language_model import preprocess_string

def perplexity(string, language_model, log=True, vocabulary=list(string.ascii_lowercase[:26] + "_")):
    """
    Computes the perplexity of a given string, for the specified language model.

    Given a sentence composed by character [c_1, c_2, ..., c_n],
    perplexity is defined as P(c_1, c_2, ..., c_n)^(-1/n).

    :param string: the input string on which perplexity is computed
    :param language_model: language model used to compute perplexity.
        It is a matrix in which entry [i, j, k] is P(k | j, i).
    :param log: returns perplexity in log-space.
    :param vocabulary: the vocabulary that is used to evaluate the perplexity.
    :return: the perplexity of the sentence.
    """
    v_dict = {char: num for num, char in enumerate(vocabulary)}

    perp = 0
    for i in range(len(string) - 2):
        perp += np.log2(language_model[v_dict[string[i-2]], v_dict[string [i-1]], v_dict[string[i]]])
    perp *= -(1/len(string))

    return perp if log==True else 2**perp

def analyze_results(results, true_cond, perc=True):
    """
    :param results: a list of tuples of the form (real_label, predicted_label)
    :param true_cond: label that should be considered as true condition
    :param perc: if true, give the results as % instead than absolute values
    :return: a dictionary with keys [TP, FN, FP, TN]
    """
    tp = sum([item[0] == true_cond and item[1] == true_cond for item in results])
    fn = sum([item[0] == true_cond and item[1] != true_cond for item in results])   
    fp = sum([item[0] != true_cond and item[1] == true_cond for item in results])
    tn = sum([item[0] != true_cond and item[1] != true_cond for item in results])

    confusion_matrix = {"TP": tp, "FN": fn, "FP": fp, "TN": tn}
    return confusion_matrix if not perc else {k: v / len(results) for k, v in confusion_matrix.items()}

def accuracy(results):
    """
    :param results: a list of tuples of the form (real_label, predicted_label)
    :return: accuracy of the results, expressed as the percentage of labels correctly predicted.
    """
    return sum([item[0] == item[1] for item in results]) / len(results)
    
model_names = ["GB", "US", "AU"]
language_models = {}
for model_name in model_names:
    language_models[model_name] = np.load("language_model_freq_" + model_name + ".npy")


test_filename = "data/test.txt"

results = []
with open(test_filename, encoding="utf8") as f:
    lines = f.readlines()
for l in lines:
    [label, sentence] = l.split("\t", 1)
    sentence = preprocess_string(sentence)
    perp_res = {k: perplexity(sentence, language_model) for k, language_model in language_models.items()}
    results.append((label, min(perp_res.keys(), key=(lambda key: perp_res[key]))))
    print(sentence[:6], "-- REAL LABEL:", label, "-- PERP:", perp_res)
print(results)


print("\n======== GB =========\n\n", analyze_results(results, "GB", False))
print("\n======== US =========\n", analyze_results(results, "US", False))
print("\n======== AU =========\n", analyze_results(results, "AU", False))

print("\n===== ACCURACY ======\n", accuracy(results))

#import scipy.io as sio
# def save_to_matlab(filename, object_name):
#     occ_matrix = np.load(filename + ".npy")
#     occ_matrix -= 1
#     sio.savemat(filename + ".mat", {object_name: occ_matrix})
#
# occ_matrix_gb = np.load("language_model_occ_GB.npy")
# occ_matrix_gb -= 1
# sio.savemat("language_model_occurrencies_GB.mat", {"occ_matrix_gb": occ_matrix_gb})
#
# save_to_matlab("language_model_occ_GB_small", "occ_matrix_gb_small")
# save_to_matlab("language_model_occ_US_small", "occ_matrix_us_small")
# save_to_matlab("language_model_occ_AU_small", "occ_matrix_au_small")
