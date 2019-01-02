import pandas
import numpy as np
import re
import pickle

# GLOBAL CONSTANTS
FILEPATH = r"../data/winemag-data-130k-v2.csv"
EMBEDDING_FILE = r"../data/glove.6B.100d.txt"
STOP_WORDS =  ["", "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves" ]
VALID_WORDS = []

def text_to_vector(text):
    letters_only = re.sub("[^a-zA-Z@]", " ", text)
    words = letters_only.lower().split(" ")
    result = []
    for word in words:
        if word not in STOP_WORDS and word in VALID_WORDS:
            result.append(word)
    return result

def word_to_embedding(word ,embedding_dict):
    return embedding_dict[word]

def words_to_embedding(words, embedding_dict):
    embeddings = []
    for word in words:
        embeddings.append(word_to_embedding(word, embedding_dict))
    return embeddings

def load_embedding(filepath):
    # Create a dictionary/map to store the word embeddings
    embeddings_index = {}

    # Load pre-computed word embeddings
    # These can be dowloaded from https://nlp.stanford.edu/projects/glove/
    # e.g., wget http://nlp.stanford.edu/data/glove.6B.zip

    f = open(filepath, encoding="utf-8")

    # Process file and load into structure
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    return embeddings_index


if __name__ == '__main__':
    df = pandas.read_csv(FILEPATH)
    embedding_dict = load_embedding(EMBEDDING_FILE)
    VALID_WORDS = set([word for word in embedding_dict.keys()])

    df["clean_text"] = df["description"].apply(lambda x : text_to_vector(x))
    print("Clean text df: ", df.head())


    df["text_embedding"] = df["clean_text"].apply(lambda x: words_to_embedding(x, embedding_dict))
    print("Embedding df: ", df.head())

    # Group by wine type (variety)
    wine_type_dict = {}
    wine_type_price_dict = {}
    wine_type_score_dict = {}
    for index, row in df.iterrows():
        if row["variety"] not in wine_type_dict.keys():
            wine_type_dict[row["variety"]] = []
            wine_type_price_dict[row["variety"]] = []
            wine_type_score_dict[row["variety"]] = []
        wine_type_dict[row["variety"]].append(row["text_embedding"])
        wine_type_price_dict[row["variety"]].append(np.array(row["price"]))
        wine_type_score_dict[row["variety"]].append(np.array(row["points"]))

    # Combine docs to one doc for each wine type
    for key in wine_type_dict.keys():
        result = []
        docs = wine_type_dict[key]
        for doc in docs:
            for word in doc:
                result.append(word)
        wine_type_dict[key] = result

    # Make a final df with attributes for wine_type(variety), documents, median_price and average score.
    wine_types = []
    docs = []
    median_price = []
    average_score = []
    for key in wine_type_dict.keys():
        wine_types.append(key)
        docs.append(wine_type_dict[key])
        median_price.append(np.nanmedian(wine_type_price_dict[key]))
        average_score.append(np.mean(wine_type_score_dict[key]))
    wine_type_df = pandas.DataFrame({'variety': wine_types, 'docs': docs, "median_price": median_price,
                                     "average_score": average_score})

    wine_type_df.to_pickle("../output/wine_reviews")


