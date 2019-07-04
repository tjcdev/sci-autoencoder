import numpy as np
np.random.seed(0)
from time import gmtime, strftime

import CDAE
import load_data
import metrics
from keras.models import Model
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

from scipy.sparse import vstack

from sklearn.metrics import precision_recall_fscore_support

class Recommender:
    def __init__(self, user_projects, similarity_matrix, project_ids):
        self.user_projects = user_projects
        self.similarity_matrix = similarity_matrix
        self.project_ids = project_ids

    def evaluate(self):
        indices_of_done_projects, top_N = self.top_projects() # get_top_N(np.array(self.user_projects, copy=True), similarity_matrix, project_ids)
        top_N_project_ids = list(top_N['project_id'])

        # Filter out
        self.user_projects[indices_of_done_projects] = 0
        y_true = self.user_projects

        project_ids = list(self.similarity_matrix.index)

        y_pred = [project_id in top_N_project_ids for project_id in project_ids]*1
        
        # Check the predicted projects and the true projects
        ones_idx = np.nonzero(y_true)
        true_projects = np.array(project_ids)[np.array(ones_idx).flatten()]
        true_ones = len(np.array(ones_idx).flatten())

        ones_idx = np.nonzero(y_pred)
        pred_projects = np.array(project_ids)[np.array(ones_idx).flatten()]
        pred_ones = len(np.array(ones_idx).flatten())

        precision, recall, fscore, support = precision_recall_fscore_support(y_true, y_pred, average='binary', pos_label=1)

        return precision, recall

    def top_projects(self):
        # Cut out some of the projects that are done to test our model
        perc_projects = 0.2

        user_projects = np.array(self.user_projects, copy=True)

        ones_idx = np.nonzero(user_projects)
        num_ones = len(np.array(ones_idx).flatten())
        to_cut = np.random.choice(num_ones, int(np.ceil(perc_projects * num_ones)), replace=False)
        ones_indices_to_cut = np.array(ones_idx).flatten()[to_cut]
        user_projects[ones_indices_to_cut] = 0

        # Calculate the similarity between user and projects
        num_projects = np.count_nonzero(user_projects)
        user_projects_sim = np.sum(user_projects * self.similarity_matrix.values, axis=1) / num_projects

        similar_items = pd.DataFrame(user_projects_sim)
        similar_items.columns = ['similarity_score']
        similar_items['project_id'] = self.project_ids

        # Filter out projects already done (only if we are not doing the precision and recall)
        indices_of_done_projects = list(np.nonzero(user_projects))[0]
        done_projects = similar_items.iloc[indices_of_done_projects]
        # TODO: check that this definitely removes the projects that have been done already
        similar_items = similar_items[~similar_items['project_id'].isin(list(done_projects['project_id']))]

        # Pick the Top-N item
        N = max([20, to_cut.shape[0]*4])
        similar_items = similar_items.sort_values('similarity_score', ascending=False)
        similar_items = similar_items.head(N)
        return indices_of_done_projects, similar_items[['project_id', 'similarity_score']]