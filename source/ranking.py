import sys
import pandas
import numpy as np
import pickle
from source.pre_processing import load_embedding


EMBEDDING_FILE = r"../data/glove.6B.100d.txt"
DATA_FILE = r"../output/wine_reviews"
ALPHA = 0.5 # How much weight is given to the search terms vs rating

def _get_args():
    """
    The arguments should be given in the order and format specified here:
    (String)(word1;word2;...;wordN) search words <space> (Float) lower price bound <space> (Float) higher price bound.


    :return: A list with the search words and price bounds
    """
    try:
        words = sys.argv[1].split(";")
        words = [word.lower() for word in words]
    except:
        raise Exception("The first argument has to be a string formatted as word1;word2;...;wordN")

    try:
        lower_bound = float(sys.argv[2])
        higher_bound = float(sys.argv[3])
        if higher_bound < 0:
            higher_bound = 1000000.0
    except:
        raise Exception("The 2. and 3. arguments must be floats. If no higher bound is given set it to -1.")

    return [words, lower_bound, higher_bound]


def distance(review_embeddings, embedding_vector):
    total_dist = 0
    for re_vec in review_embeddings:
        total_dist += np.sqrt(np.sum((re_vec-embedding_vector)**2))
    avr_dist = total_dist/len(review_embeddings)
    return avr_dist


def multiple_distance(review_embeddings, embedding_vectors):
    return np.mean([distance(review_embeddings, embedding_vector) for embedding_vector in embedding_vectors])


def max_min_normalize(dictionary, maximum=None, minimum=None):
    values = [ituple[1] for ituple in dictionary.items()]
    if maximum is None:
        maximum = np.max(values)
    if minimum is None:
        minimum = np.min(values)
    for key in dictionary.keys():
        val = dictionary[key]
        new_val = (val-minimum)/(maximum-minimum)
        dictionary[key] = new_val
    return dictionary


if __name__ == '__main__':
    words, lower_bound, higher_bound = _get_args()
    embedding_dict = load_embedding(EMBEDDING_FILE)
    word_embeddings = [embedding_dict[word] for word in words if word in embedding_dict.keys()]
    df_file = open(DATA_FILE, "rb")
    df = pickle.load(df_file)
    distance_dict = {}
    score_dict = {}
    for row in df.iterrows():
        # Filter for price
        median_price = row[1]["median_price"]
        if median_price < lower_bound or median_price > higher_bound or pandas.isna(median_price):
            continue
        else:
            distance_dict[row[1]["variety"]] = multiple_distance(row[1]["docs"], word_embeddings)
            score_dict[row[1]["variety"]] = row[1]["average_score"]

    # Max min normalize distances and scores
    distance_dict = max_min_normalize(distance_dict)
    score_dict = max_min_normalize(score_dict, maximum=100, minimum=80)

    # Make low a value in the dict correspond to a high rated wine type
    for key in score_dict.keys():
        score_dict[key] = 1 - score_dict[key]

    result_dict = {}
    for key in distance_dict.keys():
        result_dict[key] = (ALPHA*distance_dict[key] + (1-ALPHA)*score_dict[key])/2
    sorted_d = sorted(result_dict.items(), key=lambda x: x[1])

    # Print to frontend
    for i in range(3):
        line = str(sorted_d[i][0]) + ";"
        selected_row = None
        for index, row in df.iterrows():
            if row["variety"] == sorted_d[i][0]:
                selected_row = row
                break

        line += str(selected_row["median_price"]) + ";"
        line += str(selected_row["average_score"])
        print(line)
    


















