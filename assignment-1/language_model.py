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
    lines = re.sub(r"[\s]+", "__", lines)
    # Also add a double underscore at the start and at the end of each sentence.
    lines = "__" + lines + "__"

    return lines


##########################################
# COUNT TRIGRAMS  ########################
##########################################

# Build the vocabulary, in our case a list of alphabetical character plus _
# Treating _ as character will allow to model the ending of words too!
vocabulary = list(string.ascii_lowercase[:26] + "_")

def count_occ(sentence, trigram):
    occurrences = 0

    for n in range(len(sentence) - 2):
      if sentence[n:n+3] == trigram:
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


def build_occurrencies_matrix(vocabulary, lines, smoothing="laplace"):
    """
    Build a matrix in which position [i, j, k] corresponds
    to the number of occurrences of trigram "ijk" in the given corpus
    :param vocabulary: the characters for which the occurrences are counted
    :param lines: a text string
    :param smoothing: the type of smoothing to be applied
    :return: a 3-D numpy tensor
    """
    start_time = timeit.default_timer()
    occurrencies_matrix = np.zeros(shape=(len(vocabulary), len(vocabulary), len(vocabulary)))
    v_dict = {char: num for num, char in enumerate(vocabulary)}

    # Generate all the trigrams that appear in the corpus
    trigrams = generate_n_grams(lines, 3)
    # For each trigram, count its occurrences
    for i_t, t in enumerate(trigrams):
        print(i_t / len(trigrams))
        occurrencies_matrix[v_dict[t[0]], v_dict[t[1]], v_dict[t[2]]] = count_occ(lines, t)
    end_time = timeit.default_timer()
    print("! -> EXECUTION TIME OF OCCURRENCIES COUNTING:", (end_time - start_time), "\n")

    if smoothing=="laplace":
        # Apply laplacian smoothing
        occurrencies_matrix += 1

    return occurrencies_matrix


##########################################
# ESTIMATE PROBABILITIES #################
##########################################

def build_freq_matrix(vocabulary, occurrencies_matrix):
    start_time = timeit.default_timer()

    # Estimate probabilities of encountering "k" after "ij":
    # prob(k | i, j) = count("ijk") / count("ij")
    frequency_matrix = np.zeros(shape=(len(vocabulary), len(vocabulary), len(vocabulary)))
    for i in range(len(vocabulary)):
        for j in range(len(vocabulary)):
            for k in range(len(vocabulary)):
                frequency_matrix[i, j, k] = occurrencies_matrix[i, j, k] / (sum(occurrencies_matrix[i, j, :]))
    end_time = timeit.default_timer()
    print("! -> EXECUTION TIME OF PROBABILITIES:", (end_time - start_time), "\n")

    return frequency_matrix

##########################################
# MAIN  TRAINING #########################
##########################################

def train(version):
    filename = "./data/training." + version + ".txt"

    start_time = timeit.default_timer()
    with open(filename, encoding="utf8") as f:
        lines = f.readlines()

    lines = lines[:int(len(lines)/200)]
    lines = preprocess_string(lines)

    end_time = timeit.default_timer()
    print("! -> EXECUTION TIME OF TEXT PREPROCESSING:", (end_time - start_time), "\n")

    print(lines[:20])

    occurrencies_matrix = build_occurrencies_matrix(vocabulary, lines)
    np.save("./language_model_occ_" + version, occurrencies_matrix)
    frequency_matrix = build_freq_matrix(vocabulary, occurrencies_matrix)
    np.save("./language_model_freq_" + version, frequency_matrix)

versions = ["GB", "US", "AU"]
for v in versions:
    train(v)




