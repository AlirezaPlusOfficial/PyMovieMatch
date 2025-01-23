import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process
import numpy as np

# Dataset
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')

# Merge Dataset
data = pd.merge(ratings, movies, on='movieId')

user_movie_matrix = data.pivot_table(index='userId', columns='title', values='rating')

movie_similarity = cosine_similarity(user_movie_matrix.T.fillna(0))

movie_sim_df = pd.DataFrame(movie_similarity, index=user_movie_matrix.columns, columns=user_movie_matrix.columns)

def find_closest_movie(input_title):
    all_titles = movie_sim_df.index.tolist()
    closest_match, score = process.extractOne(input_title, all_titles)
    if score > 60:  # Increase this if u want get more accurate recommendation...
        return closest_match
    else:
        return None


def personalized_recommendations(favorite_movies, num_recommendations=5):
    similar_scores = pd.Series(dtype=float)
    for movie in favorite_movies:
        if movie in movie_sim_df:
            similar_scores = similar_scores.add(movie_sim_df[movie], fill_value=0)
        else:
            print(f"'{movie}' not found in dataset.")

    # Sort recommendations for better view
    recommendations = similar_scores.sort_values(ascending=False).drop(favorite_movies, errors='ignore')
    return recommendations.head(num_recommendations)

# UInput
print("PyMovieMatch By https://github.com/AlirezaPlusOfficial")
print("Enter a list of your favorite movies separated by commas (e.g., Toy Story, Jurassic Park, etc...): ")
user_input = input("Your favorite movies: ")

input_titles = [title.strip() for title in user_input.split(",")]

favorite_movies = []
for title in input_titles:
    closest_movie = find_closest_movie(title)
    if closest_movie:
        print(f"Detected: '{closest_movie}' for input '{title}'")
        favorite_movies.append(closest_movie)
    else:
        print(f"'{title}' could not be matched to any movie in the dataset!")

# Generate
print("\nYour Personalized Recommendations:\n")
recommendations = personalized_recommendations(favorite_movies)

if recommendations.empty:
    print("No recommendations found. Please check your input movies or expand dataset!")
else:
    for movie, score in recommendations.items():
        print(f"{movie} (Similarity Score: {score:.2f})")
