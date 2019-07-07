import numpy as np
np.random.seed(0)
from time import gmtime, strftime

import os
import sys
module_path = os.path.abspath(os.path.join('../../../'))
if module_path not in sys.path:
    sys.path.append(module_path)


import CDAE
import load_data
import metrics

'''
from src.models.cdea import CDAE
from src.models.cdea import load_data
from src.models.cdea import metrics 
'''

from keras.models import Model
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import math

from scipy.sparse import vstack

class Recommender:
    def __init__(self, projects, users_projects_matrix):
        self.project_ids = list(projects['project_id'])
        self.users_projects_matrix = users_projects_matrix
        self.projects = projects

    def similarity(self, users_embeddings):
        # Calculate our Cosine Similarity Matrix
        similarity_matrix= pd.DataFrame(cosine_similarity(
            X=users_embeddings),
            index=self.project_ids)

        return similarity_matrix

    def top_projects(self, user_projects_list, similarity_matrix):
        # Get user id
        user_id = user_projects_list['profile']

        # Get list of projects
        projects_list = [val for val in user_projects_list['projects'] if not math.isnan(val)]

        # Get the adjacency vector for this user
        adj_matrix = self.users_projects_matrix[self.users_projects_matrix['profile'] == user_id]
        adj_matrix = adj_matrix.drop(columns=['profile'])

        # Cut out projects that occured after the 80% time step
        cutoff_idx = int(np.ceil(len(projects_list)*0.8))

        # Project Ids of projects before the cutoff
        before_cutoff = list(set(projects_list[:cutoff_idx]))

        # Project Ids of projects after the cutoff
        after_cutoff = list(set(projects_list[cutoff_idx:]))

        # Figure out which projects to cut out of the 
        projects_to_cut = np.setdiff1d(after_cutoff, before_cutoff)

        # Set the project to 0
        adj_matrix[projects_to_cut] = 0
        adj_matrix = np.array(adj_matrix.T)

        # Calculate the similarity between user and projects
        # TODO: assert here because there shouldn't be a case where num_projects is 0
        num_projects = np.count_nonzero(adj_matrix)
        user_projects_sim = np.sum(adj_matrix * similarity_matrix.values, axis=0) / num_projects

        # Order the projects by similarity
        similar_items = pd.DataFrame(user_projects_sim)
        similar_items.columns = ['similarity_score']
        similar_items['project_id'] = self.projects['project_id']
        similar_items = similar_items.sort_values('similarity_score', ascending=False)

        # Pick the Top-N item
        N = 5 # TODO: change this number to be the correct number of projects
        similar_items = similar_items.head(N)

        return after_cutoff, similar_items

    def predictions(self, after_cutoff, similar_items):
        
        after_cutoff = np.array(after_cutoff, dtype=int)
        after_cutoff_idx = self.projects.index[self.projects['project_id'].isin(after_cutoff)].tolist()

        y_true = np.zeros(self.projects.shape[0])
        y_true[after_cutoff_idx] = 1

        predicted_projects = np.array(similar_items.index, dtype=int)
        y_pred = np.zeros(self.projects.shape[0])
        y_pred[predicted_projects] = 1

        return y_true, y_pred