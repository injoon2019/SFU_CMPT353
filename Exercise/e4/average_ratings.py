import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from difflib import get_close_matches


def close_match(data):
    new_data = get_close_matches(data, movie_list['title'])
    if len(new_data) == 0:
        return
    else:
        return new_data[0]


file1 = sys.argv[1] #movie_list.txt
#file1 = "movie_list.txt"
file2 = sys.argv[2] #movie_ratings.csv
#file2 = "movie_ratings.csv"
#file3 = "output.csv"
file3 = sys.argv[3] #output_csv

movie_list = pd.read_csv(file1, sep="\n", names = ['title', 'rating'])
movie_rating = pd.read_csv(file2, sep=",")
movie_rating.rating = pd.to_numeric(movie_rating.rating)
#print(movie_rating.dtypes)
match_func = np.vectorize(close_match)

#match movie rating title and movie list title
movie_rating['title'] = movie_rating['title'].apply(match_func)

#drop NA
movie_rating = movie_rating.dropna()
movie_rating = movie_rating.groupby('title').mean().round(2)

movie_rating.to_csv(file3)
