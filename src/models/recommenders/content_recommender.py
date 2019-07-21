import numpy as np
np.random.seed(0)
from time import gmtime, strftime

import os
import sys

# Import our python libraries
from keras.models import Model
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import math
from scipy.sparse import vstack

class ContentRecommender:
    def __init__(self):
        print("Recommender")

    def similarity(self, embeddings):
        # Calculate our Cosine Similarity Matrix
        similarity_matrix = pd.DataFrame(cosine_similarity(
            X=embeddings))

        return similarity_matrix

    def top_projects(self, train_projects, test_projects, similarity_matrix, k):
        # Calculate the similarity between user and projects
        num_projects = train_projects.count_nonzero()
        if num_projects == 0:
            num_projects = 1
        user_projects_sim = np.sum(train_projects * similarity_matrix.values, axis=0) / num_projects

        # TODO: Remove items from the list that are already done
        # We can do this through masking
        projects_to_not_suggest_again = train_projects.nonzero()[1]
        user_projects_sim[projects_to_not_suggest_again] = -1

        # Order the projects by similarity
        similar_items = pd.DataFrame(user_projects_sim)
        similar_items.columns = ['similarity_score']
        similar_items['project_id'] = similarity_matrix.columns
        similar_items = similar_items.sort_values('similarity_score', ascending=False)

        # Pick the Top-K new items
        similar_items = similar_items.head(k)

        return similar_items

    def generate_y(self, test_x, similar_items):
        y_true = test_x.T.todense()

        y_true = np.squeeze(np.asarray(y_true))

        predicted_projects = np.array(similar_items.index, dtype=int)
        y_pred = np.zeros(y_true.shape)
        y_pred[predicted_projects] = 1

        return y_true, y_pred