#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

# Load the movie ratings dataset
ratings = pd.read_csv('ratings.csv')
movies = pd.read_csv('movies.csv')

# Create a user-item matrix
user_item_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)

# Convert the user-item matrix to a sparse matrix
csr_user_item_matrix = csr_matrix(user_item_matrix.values)

def recommend_movies(user_id, num_recs=10):
    # Get the target user's ratings
    target_user_ratings = user_item_matrix.loc[user_id]

    # Create a KNN model
    knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=10, n_jobs=-1)

    # Fit the KNN model to the user-item matrix
    knn.fit(csr_user_item_matrix)

    # Get the distances and indices of the nearest neighbors
    distances, indices = knn.kneighbors(csr_user_item_matrix[user_id], n_neighbors=num_recs+1)

    # Get the movie IDs of the nearest neighbors
    movie_ids = indices.squeeze().tolist()

    # Get the ratings of the nearest neighbors
    ratings = distances.squeeze().tolist()

    # Create a list to store the recommended movies
    recommended_movies = []

    # Loop through the nearest neighbors and get the recommended movies
    for i in range(1, num_recs+1):
        movie_id = movie_ids[i]
        rating = ratings[i]
        recommended_movies.append({'Title': movies.loc[movie_id, 'title'], 'Rating': rating})

    # Return the recommended movies
    return recommended_movies

# Test the recommendation function
user_id = 1
recommended_movies = recommend_movies(user_id, num_recs=10)
print("Recommended movies for user", user_id, ":")
for movie in recommended_movies:
    print(movie['Title'], "with rating", movie['Rating'])


# In[ ]:




