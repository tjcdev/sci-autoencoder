from keras.models import load_model
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
np.random.seed(0)
from time import gmtime, strftime
from evaluate import evaluate
from keras.models import Model
from scipy.sparse import vstack
import load_data
import sys

emedding_size = 10 # int(sys.argv[1])

model = load_model('autoencoder_movies_' + str(emedding_size) + '.h5')

# Load the proejct data
train_users, train_x, test_users, test_x = load_data.load_movies()
train_x_users = np.array(train_users, dtype=np.int32).reshape(len(train_users), 1)
test_x_users = np.array(test_users, dtype=np.int32).reshape(len(test_users), 1)

# Create a model that we will use to extract the embedding layer output
embed_model = Model(inputs=model.input, outputs=model.get_layer('embedding_layer').output)
x = np.vstack((train_x, test_x))
x_users = train_users + test_users

embedding_size = model.get_layer('embedding_layer').output_shape[2]

# TODO: remove some movies from the users "watched list"

embeddings = embed_model.predict(x=[x, np.array(x_users, dtype=np.int32).reshape(len(x_users), 1)])
embeddings = embeddings.reshape(len(x_users), embedding_size)

# Calculate our Cosine Similarity Matrix
similarity_matrix = pd.DataFrame(cosine_similarity(
    X=embeddings))

# Get the index of movies the user has already rated
user_index = 102
user_movies_idices = np.nonzero(x[user_index])

# Get the 5 most similar users
k = 5
similar_users_indices = similarity_matrix.iloc[user_index].nlargest(k).index

similar_movies = np.array([np.nonzero(x[user_index]) for user_index in similar_users_indices])
similar_movies = set([thing for sublist in similar_movies for item in sublist for thing in item])

projects_scores = dict.fromkeys(similar_movies, 0)

for s_movie in similar_movies:
    for user in similar_users_indices:
        first = similarity_matrix.loc[user]
        second = x[user][s_movie]
        mult = np.sum(first * second)
        projects_scores[s_movie] += mult

projects_scores = sorted(projects_scores.items(), key=lambda x: x[1], reverse=True)  # sort
known_user_projects = list(user_movies_idices[0]) # get_user_projects(user_index)
print(known_user_projects)

recommended_projects = [i[0] for i in projects_scores]
print(recommended_projects)

recommended_projects = list(set(recommended_projects) - set(known_user_projects))[:k]
print(recommended_projects)

# TODO: compare these projects with the users known watched list (that we help out at the start)