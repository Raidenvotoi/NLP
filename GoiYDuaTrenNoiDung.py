from __future__ import print_function
import streamlit as st
# st.set_page_config(page_title="Hệ thống gợi ý phim dựa trên nội dung", page_icon=":thumbsup:", layout="wide")

import kagglehub
import csv 
import pandas as pd
import numpy as np
# Download latest version
if "path" not in st.session_state:
    
    # kagglehub.dataset_download("tmdb/tmdb-movie-metadata", force=True)
    st.session_state.path = kagglehub.dataset_download("tmdb/tmdb-movie-metadata")

# print("Path to dataset files:", path)
import pandas as pd 
import numpy as np 
path=st.session_state.path
import streamlit_antd_components as sac

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
import os
st.title("Hệ thống gợi ý phim dựa trên nội dung")
df1=pd.read_csv(path+'/tmdb_5000_credits.csv')
movies = pd.read_csv(os.path.join(path, "tmdb_5000_movies.csv"))[['genres','title']]

df1.columns = ['id','tittle','cast','crew']
# df2= df2.merge(df1,on='id')
# st.dataframe(movies.head(5))

# def weighted_rating(x, m=, C=C):
#     v = x['vote_count']
#     R = x['vote_average']
#     # Calculation based on the IMDB formula
#     return (v/(v+m) * R) + (m/(m+v) * C)







import pandas as pd
import ast
from sklearn.preprocessing import MultiLabelBinarizer

# Assuming you have a DataFrame named 'movies'

# 1. Convert string representations to lists:
movies['genres'] = movies['genres'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])
# movies['keywords'] = movies['keywords'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])


# 2. Extract genre names from dictionaries:
movies['genres'] = movies['genres'].apply(lambda x: [d['name'] for d in x] if isinstance(x, list) else [])

# 3. One-hot encode 'genres':
mlb_genres = MultiLabelBinarizer()
genres_encoded = mlb_genres.fit_transform(movies['genres'])
genres_df = pd.DataFrame(genres_encoded, columns=mlb_genres.classes_, index=movies.index)


# 4. Concatenate encoded features with original DataFrame:
movies = pd.concat([movies, genres_df], axis=1)

# 5. Drop original 'genres' and 'keywords' columns:
movies.drop(['genres'], axis=1, inplace=True)

# prompt: Compute the cosine similarity matrix

from sklearn.metrics.pairwise import cosine_similarity

# Assuming 'movies' DataFrame has been prepared as in the previous code.

# Select only the one-hot encoded genre columns for cosine similarity calculation.
genre_columns = [col for col in movies.columns if col not in ['title']]
genre_matrix = movies[genre_columns]

# Compute the cosine similarity matrix.
if "cosine_sim_matrix" not in st.session_state:
    st.session_state.cosine_sim_matrix = cosine_similarity(genre_matrix, genre_matrix)
cosine_sim_matrix = st.session_state.cosine_sim_matrix

import pandas as pd
import numpy as np

def recommend_movies(movie_title, cosine_sim_matrix, movies_df, top_n=6):
    """
    Recommends movies based on cosine similarity of genres.

    Args:
        movie_title (str): The title of the input movie.
        cosine_sim_matrix (np.ndarray): The cosine similarity matrix.
        movies_df (pd.DataFrame): The DataFrame containing movie information.
        top_n (int): The number of recommendations to return (default is 5).

    Returns:
        pd.DataFrame: A DataFrame of recommended movies with their titles and similarity scores.
                           Returns an empty DataFrame if the movie is not found or if an error occurs.
    """
    try:
        # Find the index of the input movie in the DataFrame.
        idx = movies_df.index[movies_df['title'] == movie_title].tolist()[0]

        # Get the pairwise similarity scores of all movies with that movie.
        sim_scores = list(enumerate(cosine_sim_matrix[idx]))

        # Sort the movies based on the similarity scores.
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get the scores of the top-n most similar movies. Ignore the first movie.
        sim_scores = sim_scores[1:top_n+1]

        # Get the movie indices.
        movie_indices = [i[0] for i in sim_scores]

        # Return the top-n most similar movies.
        recommendations_df = pd.DataFrame({
            'title': movies_df['title'].iloc[movie_indices].values,
            'similarity_score': [i[1] for i in sim_scores]  # Include similarity scores
        })
        return recommendations_df
    except IndexError:
        print(f"Movie '{movie_title}' not found in the dataset.")
        return pd.DataFrame()  # Return an empty DataFrame if the movie is not found
    except Exception as e:
        print(f"An error occurred: {e}")
        return pd.DataFrame() # Return empty DataFrame for other errors

# Example usage (assuming you have 'movies', 'cosine_sim_matrix' from the previous code):
# Replace 'your_movie_title' with the title of the movie you want recommendations for.
list_of_movie_names = movies['title'].tolist()
selected_movie_name = st.selectbox("Chọn tên phim", list_of_movie_names)
movie_recommendations = recommend_movies(selected_movie_name, cosine_sim_matrix, movies)
list_recommendations = movie_recommendations['title'].tolist()
st.write('Gợi ý cho film ',selected_movie_name,' là:')
for i in range(1,len(list_recommendations)):
    st.write(f"{i}. {list_recommendations[i]}")
st.markdown("### Hệ thống gợi ý phim dựa trên thể loại")
theloaibanthich=st.multiselect('Chọn thể loại phim mà bạn yêu thích:',list(mlb_genres.classes_,),default=list(mlb_genres.classes_)[0:3])


import pandas as pd

def add_new_film(movies, title, genres):
    """
    Adds a new film with its genres to the movie DataFrame.
    If a film with the same title already exists, it updates the genres.

    Args:
        movies: The DataFrame containing movie information.
        title: The title of the new film.
        genres: A list of genres for the new film.

    Returns:
        pandas.DataFrame: The updated DataFrame with the new film included.
    """

    # Check if the title exists
    if title in movies['title'].values:
        # If it exists, update the genres for that title
        movie_index = movies[movies['title'] == title].index[0]
        for genre in movies.columns:
            if genre not in ['title']:
                movies.loc[movie_index, genre] = int(genre in genres)

    else:
        # If it doesn't exist, create a new row for the new film
        new_movie = pd.DataFrame({'title': [title]})

        # Ensure genre columns exist
        for genre in set(genres):
            if genre not in movies.columns:
                movies[genre] = 0
        # Set the genre columns in new_movie
        for genre in movies.columns:
            if genre not in ['title']:
                new_movie[genre] = int(genre in genres)

        # Append the new movie to the DataFrame
        movies = pd.concat([movies, new_movie], ignore_index=True)

    return movies


new_genres = theloaibanthich
movies = add_new_film(movies, 'My New Film', new_genres)

if movies is not None:
    #Update the genre matrix, cosine similarity matrix etc.
    genre_columns = [col for col in movies.columns if col not in ['title']]
    genre_matrix = movies[genre_columns]

    # Compute the cosine similarity matrix.
    cosine_sim_matrix = cosine_similarity(genre_matrix, genre_matrix)

new_movie_recommendations = recommend_movies("My New Film", cosine_sim_matrix, movies)

new_list_recommendations = new_movie_recommendations['title'].tolist()
st.write('Gợi ý cho bạn là:')
for i in range(1,len(new_list_recommendations)):
    st.write(f"{i}. {new_list_recommendations[i]}")



# # Define a new feature 'score' and calculate its value with `weighted_rating()`
# q_movies['score'] = q_movies.apply(weighted_rating, axis=1)
# #Sort movies based on score calculated above
# q_movies = q_movies.sort_values('score', ascending=False)

# #Print the top 15 movies
# st.dataframe(q_movies[['title', 'vote_count', 'vote_average', 'score']].head(10))
# pop= df2.sort_values('popularity', ascending=False)
# import matplotlib.pyplot as plt
# plt.figure(figsize=(12,4))

# plt.barh(pop['title'].head(6),pop['popularity'].head(6), align='center',
#         color='skyblue')
# plt.gca().invert_yaxis()
# plt.xlabel("Popularity")
# plt.title("Popular Movies")
# st.pyplot(plt)
# #Import TfIdfVectorizer from scikit-learn
# from sklearn.feature_extraction.text import TfidfVectorizer

# #Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
# tfidf = TfidfVectorizer(stop_words='english')

# #Replace NaN with an empty string
# df2['overview'] = df2['overview'].fillna('')

# #Construct the required TF-IDF matrix by fitting and transforming the data
# tfidf_matrix = tfidf.fit_transform(df2['overview'])

# #Output the shape of tfidf_matrix
# tfidf_matrix.shape
# # Import linear_kernel
# from sklearn.metrics.pairwise import linear_kernel

# # Compute the cosine similarity matrix
# cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
# #Construct a reverse map of indices and movie titles
# indices = pd.Series(df2.index, index=df2['title']).drop_duplicates()
# # Function that takes in movie title as input and outputs most similar movies
# def get_recommendations(title, cosine_sim=cosine_sim):
#     # Get the index of the movie that matches the title
#     idx = indices[title]

#     # Get the pairwsie similarity scores of all movies with that movie
#     sim_scores = list(enumerate(cosine_sim[idx]))

#     # Sort the movies based on the similarity scores
#     sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

#     # Get the scores of the 10 most similar movies
#     sim_scores = sim_scores[1:11]

#     # Get the movie indices
#     movie_indices = [i[0] for i in sim_scores]

#     # Return the top 10 most similar movies
#     return df2['title'].iloc[movie_indices]
# list_of_movie_names = df2['title'].tolist()
# selected_movie_name = st.selectbox("Chọn tên phim", list_of_movie_names)

# st.write('Tên phim bạn đã chọn:',selected_movie_name)
# st.write('Nội dung phim:',df2.loc[df2['title'] == selected_movie_name, 'overview'].values[0])
# st.write('Gợi ý phim tương tự:')
# recommendations = get_recommendations(selected_movie_name)
# for i, movie in enumerate(recommendations, 1):
#     st.write(f"{i}. {movie}")
# newfilm=st.text_input('Nhập tên phim mới')
# newfilm_detail=st.text_input('Nhập nội dung phim')

# st.multiselect('Chọn thể loại phim', ['Hành động', 'Kinh dị', 'Tình cảm', 'Hài hước'])