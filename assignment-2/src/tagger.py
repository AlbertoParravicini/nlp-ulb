# Use NLTK for automatic POS tagging.
# Just a check to compare NLTK results with the hand-made ones.

import nltk
from nltk.stem import WordNetLemmatizer


if __name__ == "__main__":
    filename = "../data/raw_sentences.txt"
    with open(filename, encoding="utf8") as f:
        lines = f.readlines()
        
        
    # POS tagging
    tagged = []
    for l in lines:
        text = nltk.word_tokenize(l)
        pos_tagged = nltk.pos_tag(text)
        simplified_tags = [(word, nltk.tag.map_tag('en-ptb', 'universal', tag)) for word, tag in pos_tagged]
        tagged += [simplified_tags]
        
    
    output_filename = "../data/sentences_tagged_nltk.txt"
    with open(output_filename, "w+", encoding="utf8",) as f:
        for t in tagged:
            print(t, file=f)
            
    # Lemmatization
    wordnet_lemmatizer = WordNetLemmatizer()
    output_filename = "../data/sentences_lemmas_nltk.txt"
    with open(output_filename, "w+", encoding="utf8",) as f:
        for l in lines:
            text = nltk.word_tokenize(l)
            for t in text:    
                print(t, wordnet_lemmatizer.lemmatize(t), file=f)
            print("\n", file=f)
    
        