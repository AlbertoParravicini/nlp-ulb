import numpy as np
import re
import timeit
import string
import bigrams_kneser as bk

def perplexity(string, language_model, log=True, vocabulary=list(string.ascii_lowercase[:26] + "_")):
    v_dict = {char: num for num, char in enumerate(vocabulary)}

    perp = 0
    for i in range(len(string) - 1):
        perp += np.log2(language_model[v_dict[string [i-1]], v_dict[string[i]]])
    perp *= -(1/len(string))

    return perp if log==True else 2**perp

def analyze_results(results, true_cond, perc=True):
    tp = sum([item[0] == true_cond and item[1] == true_cond for item in results])
    fn = sum([item[0] == true_cond and item[1] != true_cond for item in results])
    fp = sum([item[0] != true_cond and item[1] == true_cond for item in results])
    tn = sum([item[0] != true_cond and item[1] != true_cond for item in results])

    confusion_matrix = {"TP": tp, "FN": fn, "FP": fp, "TN": tn}
    return confusion_matrix if not perc else {k: v / len(results) for k, v in confusion_matrix.items()}

def accuracy(results):
    return sum([item[0] == item[1] for item in results]) / len(results)


def train_wrapper(delta):
    print("delta: ", delta)
    versions = ["GB", "US", "AU"]
    language_models = {}
    for v in versions:
        occ_matrix = np.load("./language_model_occ_bigrams" + v + ".npy")
        language_models[v] = bk.train_kneser_ney(occ_matrix, delta)

    test_filename = "test_lines.txt"
    test_labels_filename = "test_labels.txt"

    results = []
    with open(test_filename, encoding="utf8") as f:
        test_lines = f.readlines()
    with open(test_labels_filename, encoding="utf8") as f:
        test_labels = f.readlines()
        test_labels = [x.strip() for x in test_labels]
    for sentence, label in zip(test_lines, test_labels):
        sentence = bk.preprocess_string(sentence)
        perp_res = {k: perplexity(sentence, language_model) for k, language_model in language_models.items()}
        results.append((label, min(perp_res.keys(), key=(lambda key: perp_res[key]))))
    return -accuracy(results)


# import scipy.optimize as optim
# res = -optim.minimize_scalar(train_wrapper, bounds=(0.01, 10), method='bounded')
# print(res)


test_filename = "data/test.txt"
versions = ["GB", "US", "AU"]
language_models = {}
for v in versions:
    occ_matrix = np.load("./language_model_occ_bigrams" + v + ".npy")
    language_models[v] = bk.train_kneser_ney(occ_matrix)
results = []
with open(test_filename, encoding="utf8") as f:
    lines = f.readlines()
for l in lines:
    [label, sentence] = l.split("\t", 1)
    sentence = bk.preprocess_string(sentence)
    perp_res = {k: perplexity(sentence, language_model) for k, language_model in language_models.items()}
    results.append((label, min(perp_res.keys(), key=(lambda key: perp_res[key]))))
    print(sentence[:6], "-- REAL LABEL:", label, "-- PERP:", perp_res)
print(results)


print("\n======== GB =========\n\n", analyze_results(results, "GB", False))
print("\n======== US =========\n", analyze_results(results, "US", False))
print("\n======== AU =========\n", analyze_results(results, "AU", False))

print("\n===== ACCURACY ======\n", accuracy(results))
