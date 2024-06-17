import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Loading the movie data from the CSV file into a Pandas DataFrame
movie_data = pd.read_csv('/content/movies.csv')

# Displaying the first 5 rows of the dataframe
movie_data.head()

# Getting the number of rows and columns in the dataframe
movie_data.shape

# Selecting the relevant features for movie recommendation
selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']
print("Selected Features:", selected_features)

# Replacing the null values with empty strings for the selected features
for feature in selected_features:
    movie_data[feature] = movie_data[feature].fillna('')

# Combining all the selected features into one string
combined_features = movie_data['genres'] + ' ' + movie_data['keywords'] + ' ' + movie_data['tagline'] + ' ' + movie_data['cast'] + ' ' + movie_data['director']

# Converting the combined text data into feature vectors
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)

# Calculating similarity scores using cosine similarity
similarity_scores = cosine_similarity(feature_vectors)
print("Similarity Matrix Shape:", similarity_scores.shape)

# Getting the user's favorite movie name
user_movie_name = input('Enter the name of your favorite movie: ')

# Creating a list with all the movie titles from the dataset
all_movie_titles = movie_data['title'].tolist()

# Finding the closest match for the movie name given by the user
closest_matches = difflib.get_close_matches(user_movie_name, all_movie_titles)
closest_match = closest_matches[0]
print("Closest Match:", closest_match)

# Finding the index of the movie with the closest match title
movie_index = movie_data[movie_data.title == closest_match].index[0]
print("Index of Closest Match:", movie_index)

# Getting a list of similarity scores for the closest match movie
similarity_scores_list = list(enumerate(similarity_scores[movie_index]))

# Sorting the movies based on their similarity scores
sorted_similar_movies = sorted(similarity_scores_list, key=lambda x: x[1], reverse=True)

# Printing the top recommended movies based on similarity
print('Recommended Movies:\n')

count = 1
for movie in sorted_similar_movies:
    index = movie[0]
    title = movie_data.iloc[index]['title']
    if count < 30:
        print(f"{count}. {title}")
        count += 1

# Asking the user for another movie recommendation
user_movie_name = input('Enter another favorite movie name: ')

# Finding the closest match for the new movie name given by the user
closest_matches = difflib.get_close_matches(user_movie_name, all_movie_titles)
closest_match = closest_matches[0]
print("Closest Match:", closest_match)

# Finding the index of the movie with the closest match title
movie_index = movie_data[movie_data.title == closest_match].index[0]
print("Index of Closest Match:", movie_index)

# Getting a list of similarity scores for the closest match movie
similarity_scores_list = list(enumerate(similarity_scores[movie_index]))

# Sorting the movies based on their similarity scores
sorted_similar_movies = sorted(similarity_scores_list, key=lambda x: x[1], reverse=True)

# Printing the top recommended movies based on similarity
print('Recommended Movies:\n')

count = 1
for movie in sorted_similar_movies:
    index = movie[0]
    title = movie_data.iloc[index]['title']
    if count < 30:
        print(f"{count}. {title}")
        count += 1
