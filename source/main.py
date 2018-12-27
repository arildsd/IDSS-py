import pandas
import numpy as np
import re

# GLOBAL CONSTANTS
FILEPATH = r"../data/winemag-data-130k-v2.csv"



def text_to_vector(text):
    letters_only = re.sub("[^a-zA-Z@]", "", text)
    return letters_only.lower().split(" ")

if __name__ == '__main__':
    df = pandas.read_csv(FILEPATH)
    df["clean_text"] = df["description"].apply(lambda x : text_to_vector(x))


