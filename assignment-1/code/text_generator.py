import numpy as np
import string


def generate_text(language_model, min_length=100, vocabulary=list(string.ascii_lowercase[:26] + "_")):
    """
    Generate text from the given 3-Grams language model.
    :param language_model: a 3-D tensor in which [i, j, k] is the probability of encountering "k" after "ij".
    :param min_length: minimum length of the generated text. After that, the generation will stop after encountering "__"
    :param vocabulary: the set of characters to be used on the generation.
    :return: a string of generated text.
    """
    length = 0

    generated_text = "__"
    mem_vec = [26, 26]

    while True:
        prob_vec = language_model[mem_vec[0], mem_vec[1], :]
        #print("MEMORY:", [vocabulary[c] for c in mem_vec], list(zip(vocabulary, prob_vec)))
        new_char_index = np.random.choice(np.arange(len(vocabulary)), p=prob_vec)
        generated_text += vocabulary[new_char_index]
        mem_vec = mem_vec[1:] + [new_char_index]
        length += 1
        if length >= min_length and generated_text[-2:] == "__":
            break
    return generated_text

language_model = np.load("language_model_freq_US.npy")
generated_text = generate_text(language_model)
print(generated_text)
#print(re.sub(r"(_)", "\_", generated_text))


vocabulary=list(string.ascii_lowercase[:26] + "_")
v_dict = {char: num for num, char in enumerate(vocabulary)}

for k in vocabulary:
    print("\\textcolor{MidnightBlue}{", "iz", k, "} & ", "{0:.3f}".format(language_model[v_dict["i"], v_dict["z"], v_dict[k]]), " \\\\", sep="")
