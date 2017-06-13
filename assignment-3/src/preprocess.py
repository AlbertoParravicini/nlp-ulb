#%% IMPORT STUFF

import pandas as pd
import numpy as np
import gzip
import matplotlib.pyplot as plt
plt.rcParams.update(plt.rcParamsDefault)
plt.style.use('seaborn-pastel')

import nltk

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.decomposition import TruncatedSVD

from nltk.sentiment.vader import SentimentIntensityAnalyzer
import spacy
nlp = spacy.load('en')

#%% READ THE REVIEWS
# (code from http://jmcauley.ucsd.edu/data/amazon/)
def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield eval(l)

def read_reviews(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

data_full = read_reviews('../data/reviews_Digital_Music_5.json.gz')

#%% BASIC PREPROCESSING
data = data_full[["summary", "reviewText", "overall"]]
data.rename(columns={'overall': 'score', 'reviewText': 'original_text'}, inplace=True)
data = data[['summary', 'original_text', 'score']]

# Concatenate summary and text, then remove summary.
data["original_text"] = data["summary"] + ". " + data["original_text"]
data.drop('summary', axis=1, inplace=True)

# Build a smaller dataset.
sample_size = 40000

df_sample = data.sample(sample_size)
df_sample = df_sample.reset_index(drop=True)

#%% STORE THE DATA AS CSV
data.to_csv("../data/review_basic.csv", index=False)
df_sample.to_csv("../data/review_basic_short.csv", index=False)

#%% PLOT SCORE DISTRIBUTION
plt.bar(df_sample["score"].value_counts().keys(),  df_sample["score"].value_counts(), align='center', alpha=0.8)
plt.ylabel('Count')
plt.title('Review Scores Distribution')
 
plt.show()

# In the original dataset, by predicting the most common class (5) we get an accuracy of 0.54

#%% PREPROCESS DATA

# Put every sentence to lowercase.
df_sample["original_text"] = df_sample["original_text"].str.lower()

def preprocess_string(string):
    # Process text
    text = nlp(string)

    # Tokenized version of the processed text, can be modified more easily.
    string_tokens = [t.lemma_ for t in text]

    # Inspect the dependecy tree: if a negation is found,
    # put "NOT_" in front of every word dependent on the head of the negation
    # (except for the negation).
    for t in text:
        if t.dep_ == "neg":
            string_tokens[t.head.i] = t.head.lemma_ + "_NOT"
            for child in t.head.subtree:
                if child.dep_ != "neg":
                    string_tokens[child.i] = child.lemma_ + "_NOT"
    
    # Remove stopwords and punctuation.
    text = [t for t in text if (not t.is_stop and not t.is_punct and not t.text == " ")]
    
    # Rebuild the tokenized string.
    string_tokens = [string_tokens[t.i] for t in text]
    
    return [string_tokens, text]     

res = df_sample["original_text"].apply(lambda t: preprocess_string(t))
df_sample["text"] = [res[i][0] for i in range(len(res))]

#%%
# Create a dataset where we store the sentiment values of each sentence.
sid = SentimentIntensityAnalyzer()

df_sent = pd.DataFrame({"text": df_sample["text"], "score": df_sample["score"]})
df_sent["compound"] = 0.0
df_sent["neg"] = 0.0
df_sent["neu"] = 0.0
df_sent["pos"] = 0.0
for i in range(len(res)):
    sent = sid.polarity_scores(" ".join(res[i][0]))
    df_sent.iat[i, df_sent.columns.get_loc("compound")] = sent["compound"] 
    df_sent.iat[i, df_sent.columns.get_loc("neg")] = sent["neg"] 
    df_sent.iat[i, df_sent.columns.get_loc("neu")] = sent["neu"] 
    df_sent.iat[i, df_sent.columns.get_loc("pos")] = sent["pos"] 

    
#%% Do the same, but using the unprocessed text.
sid = SentimentIntensityAnalyzer()

df_sent_orig = pd.DataFrame({"original_text": df_sample["original_text"]})
df_sent_orig["compound_o"] = 0.0
df_sent_orig["neg_o"] = 0.0
df_sent_orig["neu_o"] = 0.0
df_sent_orig["pos_o"] = 0.0
for i, t in enumerate(df_sample["original_text"]):
    sent = sid.polarity_scores(t)
    df_sent_orig.iat[i, df_sent_orig.columns.get_loc("compound_o")] = sent["compound"] 
    df_sent_orig.iat[i, df_sent_orig.columns.get_loc("neg_o")] = sent["neg"] 
    df_sent_orig.iat[i, df_sent_orig.columns.get_loc("neu_o")] = sent["neu"] 
    df_sent_orig.iat[i, df_sent_orig.columns.get_loc("pos_o")] = sent["pos"] 
    
#%% We can use both!

df_sent_tot = df_sent.join(df_sent_orig)
df_sent_tot.to_hdf("../data/df_sent_large.h5", "df_sent", mode="w")


#%% Count the occurrences of each word.
word_list = pd.Series([t for l in df_sample["text"] for t in l])

word_counts = word_list.value_counts()

# We can delete uncommon words. 
# The threshold is arbitrary, we can try different values and see if it makes any difference.
threshold = 20
word_counts = word_counts[word_counts >= threshold]

#%% Remove words that are uncommon.
df_sample["text"] = df_sample["text"].apply(lambda t: [c for c in t if c in word_counts.keys()])

#%% Create a new dataset by binarizing the bag-of-words model.
bin_df = pd.DataFrame(0, index = df_sample.index, columns=word_counts.keys(), dtype=np.int16)

for t_i, t in enumerate(df_sample["text"]):
    for w in t:
        bin_df.iat[t_i, bin_df.columns.get_loc(w)] = 1

#%% Attach the score.      
bin_df["SCORE"] = df_sample["score"]

#%% Save the dataframe
bin_df.to_hdf("../data/bin_df_large.h5", "bin_df", mode="w")


#%% Try to apply svd to the dataset

# Compute Tfidf 
vect = TfidfVectorizer(lowercase=False, analyzer=lambda d: d.split(' '))
res = vect.fit_transform([" ".join(t) for t in df_sample["text"]])

# LSA on the Tfidf matrix
svd = TruncatedSVD(n_components=500)
compressed = svd.fit_transform(res)

compressed = pd.DataFrame(compressed)
compressed["SCORE"] = df_sample["score"]

#%% Save the dataframe
compressed.to_hdf("../data/compressed_large.h5", "compressed", mode="w")
