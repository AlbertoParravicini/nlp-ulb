import numpy as np
import re
import timeit
import string

# Used to check which packages are installed.
# Check http://stackoverflow.com/questions/1051254/check-if-python-package-is-installed
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
    # Join all the strings into one
    lines = "".join(lines)
    # Remove whitespaces at the start and end of each sentence.
    lines = lines.strip()
    # Substitute single and multiple whitespaces with a double underscore.
    lines = re.sub(r"[\s]+", "_", lines)
    # Also add a double underscore at the start and at the end of each sentence.
    lines = "_" + lines + "_"

    return lines


##########################################
# MODEL BUILDING #########################
##########################################

# Build the vocabulary, in our case a list of alphabetical character plus _
# Treating _ as character will allow to model the ending of words too!
vocabulary = list(string.ascii_lowercase[:26] + "_")

def count_occ(sentence, bigram):
    occurrences = 0

    for n in range(len(sentence) - 1):
      if sentence[n:n+2] == bigram:
          occurrences += 1
    return occurrences

def generate_n_grams(sentence, n):
    """
    Generate the set of n-grams for the given sentence.
    :param sentence: input text string
    :param n: size of the n-grams
    :return: the set of n-grams that appear in the sequence.
    """
    return set(sentence[i:i+n] for i in range(len(sentence)-n+1))


def build_occurrencies_matrix(vocabulary, lines):
    """
    Build a matrix in which position [i, j] corresponds
    to the number of occurrences of bigram "ij" in the given corpus
    :param vocabulary: the characters for which the occurrences are counted
    :param lines: a text string
    :return: a 2-D numpy tensor
    """
    start_time = timeit.default_timer()
    occurrencies_matrix = np.zeros(shape=(len(vocabulary), len(vocabulary)))
    v_dict = {char: num for num, char in enumerate(vocabulary)}

    # Generate all the trigrams that appear in the corpus
    bigrams = generate_n_grams(lines, 2)
    # For each trigram, count its occurrences
    for i_t, t in enumerate(bigrams):
        print(i_t / len(bigrams))
        occurrencies_matrix[v_dict[t[0]], v_dict[t[1]]] = count_occ(lines, t)
    end_time = timeit.default_timer()
    print("! -> EXECUTION TIME OF OCCURRENCIES COUNTING:", (end_time - start_time), "\n")

    return occurrencies_matrix

def train_kneser_ney(occ_matrix, delta=0.75, vocabulary=list(string.ascii_lowercase[:26] + "_")):
    frequency_matrix = np.zeros(shape=(len(vocabulary), len(vocabulary)))

    for i in range(len(vocabulary)):
        for j in range(len(vocabulary)):
            frequency_matrix[i, j] = (max(occ_matrix[i, j] - delta, 0) / sum(occ_matrix[i, :])) \
                                     + (delta / sum(occ_matrix[i, :])) * sum([x > 0 for x in occ_matrix[i, :]]) \
                                     * sum([x > 0 for x in occ_matrix[:, j]]) / (occ_matrix > 0).sum()

    return frequency_matrix

##########################################
# MAIN  TRAINING #########################
##########################################

def train_occ(lines, version, training_split=0.8):
    start_time = timeit.default_timer()

    lines = lines[:int(len(lines) * training_split)]
    lines = preprocess_string(lines)

    end_time = timeit.default_timer()
    print("! -> EXECUTION TIME OF TEXT PREPROCESSING:", (end_time - start_time), "\n")

    print(lines[:20])

    occurrencies_matrix = build_occurrencies_matrix(vocabulary, lines)
    np.save("./language_model_occ_bigrams" + version, occurrencies_matrix)

#
versions = ["GB", "US", "AU"]
training_split = 0.8
#
# test_lines = []
# test_labels = []
#
# for v in versions:
#    filename = "./data/training." + v + ".txt"
#    with open(filename, encoding="utf8") as f:
#        lines = f.readlines()
#    split = int(len(lines) * training_split)
#    train_lines = lines[:split]
#    test_lines += lines[split:]
#    test_labels += [v] * len(lines[split:])
#
#    train_occ(train_lines, v)
#
# with open("test_lines.txt", "w", encoding="utf8") as text_file:
#    for item in test_lines:
#        text_file.write("%s" % item)
# with open("test_labels.txt", "w", encoding="utf8") as text_file:
#    for item in test_labels:
#        text_file.write("%s\n" % item)
#
for v in versions:
    occ_matrix = np.load("./language_model_occ_bigrams" + v + ".npy")
    freq_matrix = train_kneser_ney(occ_matrix)
    np.save("./language_model_kn" + v, freq_matrix)

#
#
#


