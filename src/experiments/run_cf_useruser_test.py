import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import math
from sklearn.metrics import precision_recall_fscore_support
from sklearn.neighbors import NearestNeighbors
from time import gmtime, strftime

import os
import sys

# Load the user project_matrix
data = pd.read_pickle('data/processed/active_profile_projects')
data_items = data.drop(columns=['profile'])

# Load the project_data and profile_projects
projects = pd.read_pickle('data/processed/project_data')
profile_projects = pd.read_pickle('data/processed/profile_projects_time_consistent')

def get_user_projects(user_id):
    known_user_likes = data_items.loc[user_id]
    known_user_likes = known_user_likes[known_user_likes > 0].index.values
    return known_user_likes

def find_k_similar_users(data_items, user_id, metric='cosine', k=50):
    model_knn = NearestNeighbors(k, 1.0, 'brute', 30, metric)
    
    # Edit the data_items entry for the user_id so they don't contain the projects after cutoff
    data_items.iloc[user_id, :] 

    model_knn.fit(data_items)
    distances, indices = model_knn.kneighbors(
        data_items.iloc[user_id, :].values.reshape(1, -1), n_neighbors=k + 1)
    similarities = 1 - distances.flatten()
    return pd.Series(similarities, indices[0])

k = int(sys.argv[1])

precisions = []
recalls = []
refined_precisions = []
y_preds = []


for user_index in range(0, data.shape[0]):

    user_id = data.loc[user_index]['profile']

    # User projects
    user_projects_list = profile_projects[profile_projects['profile'] == user_id]

    # Get list of projects
    projects_list = [val for val in user_projects_list['projects'].iloc[0] if not math.isnan(val)]

    # Cut out projects that occured after the 80% time step
    cutoff_idx = int(np.ceil(len(projects_list)*0.6))

    # Project Ids of projects before the cutoff
    before_cutoff = list(set(projects_list[:cutoff_idx]))

    # Project Ids of projects after the cutoff
    after_cutoff = list(set(projects_list[cutoff_idx:]))

    # Figure out which projects to cut out of the 
    projects_to_cut = np.setdiff1d(after_cutoff, before_cutoff)

    # Edit data items so that projects to cut out are 0
    data_items.iloc[user_index][projects_to_cut] = 0

    # Get the similar users
    similar_users = find_k_similar_users(data_items, user_index).drop(user_index, 0)

    # Get the recommender projects
    similar_projects = [get_user_projects(user) for user in similar_users.index]
    similar_projects = list(set([item for sublist in similar_projects for item in sublist]))
    projects_scores = dict.fromkeys(similar_projects, 0)
    for s_project in similar_projects:
        for user in similar_users.index:
            projects_scores[s_project] += similar_users.loc[user] * data_items.loc[user][s_project]
    projects_scores = sorted(projects_scores.items(), key=lambda x: x[1], reverse=True)  # sort
    recommended_projects = [i[0] for i in projects_scores]
    known_user_projects = get_user_projects(user_index)
    recommended_projects = list(set(recommended_projects) - set(known_user_projects))[:k]

    # Create the y_true
    true_idx = projects[projects['project_id'].isin(projects_to_cut)].index
    print(true_idx)
    y_true = np.zeros(len(projects))
    y_true[true_idx] = 1

    # Create the y_pred
    pred_idx = projects[projects['project_id'].isin(recommended_projects)].index
    print(pred_idx)
    y_pred = np.zeros(len(projects))
    y_pred[pred_idx] = 1

    y_preds = y_preds + [y_pred]
    '''
        Precision and Recall
    '''
    precision, recall, fscore, support = precision_recall_fscore_support(y_true, y_pred, average='binary', pos_label=1)

    '''
        Refined Precisions
    '''
    # Calculate similarity
    data_sparse = sparse.csr_matrix(data_items)
    similarities = cosine_similarity(data_sparse.transpose())
    sim_matrix = pd.DataFrame(similarities, data_items.columns, data_items.columns)

    pred_sim_matrix = sim_matrix[recommended_projects]

    # Get the indices of all the projects that were actually participated with after cut_off time
    true_idx = np.nonzero(y_true)

    # This should now mean we have a 2D matrix which has 
    masked_pred_sim_matrix = pred_sim_matrix.iloc[true_idx]

    # Calculate the refined precision
    refined_precision = np.mean(masked_pred_sim_matrix.max(axis=0)) + precision

    precisions = precisions + [precision]
    recalls = recalls + [recall]
    refined_precisions = refined_precisions + [refined_precision]

precision_string = ', '.join([str(prec) for prec in precisions])
recall_string = ', '.join([str(recall) for recall in recalls])
refined_precision_string = ', '.join([str(ref_prec) for ref_prec in refined_precisions])

y_preds = np.array(y_preds)
np.save('data/raw-experiment-results/cfuseruser-sci-results-' + str(k) + '.npy', y_preds)

fileName = 'data/raw-experiment-results/cfuseruser-sci-results-' + str(k) + '.json'

f = open(fileName,"w+")
f.write('{"precision": [' + precision_string + '],')
f.write('"recall": [' + recall_string + '],')
f.write('"refined_precision": [' + refined_precision_string + ']}')
f.close()

print("Finished")