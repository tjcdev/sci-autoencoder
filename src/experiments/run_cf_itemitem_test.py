import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import math
from sklearn.metrics import precision_recall_fscore_support
from time import gmtime, strftime

import os
import sys

# Load the user project_matrix
data = pd.read_pickle('data/processed/active_profile_projects')
data.head()

# Drop the profile column so it's basically just an adjacency matrix
data_items = data.drop(columns=['profile'])
data_items.head()

# Load the project_data and profile_projects
projects = pd.read_pickle('data/processed/project_data')
profile_projects = pd.read_pickle('data/processed/profile_projects_time_consistent')

k_size = int(sys.argv[1])

def calculate_similarity(data_items):
    data_sparse = sparse.csr_matrix(data_items)
    similarities = cosine_similarity(data_sparse.transpose())
    sim = pd.DataFrame(similarities, data_items.columns, data_items.columns)
    return sim

def get_user_projects(user_id):
    known_user_likes = data_items.loc[user_id]
    known_user_likes = known_user_likes[known_user_likes > 0].index.values
    return known_user_likes

precisions = []
recalls = []
refined_precisions = []

data_matrix = calculate_similarity(data_items)

# Open file to write to
fileName = 'data/raw-experiment-results/cf-item-' + str(k_size) + '.txt'
f = open(fileName,"w+")
f.write(' {k: %s, results: [' % str(k_size))

# Loop over all the users
for user_index in range(0, len(data['profile'])):
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

    # Get the recommendations
    k = k_size
    # The actual recommender code to test with
    known_user_projects = data_items.loc[user_index]
    known_user_projects = known_user_projects[known_user_projects > 0].index

    # Cutout the projects that occur after the cutoff
    known_user_projects = np.array([x for x in known_user_projects if x not in projects_to_cut])

    user_projects = data_matrix[known_user_projects]  # without ratings!!
    neighbourhood_size = 10
    data_neighbours = pd.DataFrame(0, user_projects.columns, range(1, neighbourhood_size + 1))

    for i in range(0, len(user_projects.columns)):
        data_neighbours.iloc[i, :neighbourhood_size] = user_projects.iloc[0:, i].sort_values(0, False)[
                                                        :neighbourhood_size].index

    most_similar_to_likes = data_neighbours.loc[known_user_projects]

    similar_list = most_similar_to_likes.values.tolist()
    similar_list = list(set([item for sublist in similar_list for item in sublist]))
    similar_list = list(set(similar_list) - set(known_user_projects))

    neighbourhood = data_matrix[similar_list].loc[similar_list]

    user_vector = data_items.loc[user_index].loc[similar_list]

    score = neighbourhood.dot(user_vector).div(neighbourhood.sum(1))

    recommended_projects = score.nlargest(k).index.tolist()

    # Evaluate the recommendations
    # Create the y_true
    true_idx = projects[projects['project_id'].isin(projects_to_cut)].index
    y_true = np.zeros(max(projects.index)+1)
    y_true[true_idx] = 1

    # Create the y_pred
    pred_idx = projects[projects['project_id'].isin(recommended_projects)].index
    y_pred = np.zeros(max(projects.index)+1)
    y_pred[pred_idx] = 1

    # Get precision and recall
    precision, recall, fscore, support = precision_recall_fscore_support(y_true, y_pred, average='binary', pos_label=1)

    # Calculate the refined precision
    data_matrix = calculate_similarity(data_items)
    pred_sim_matrix = data_matrix[recommended_projects]
    
    # Get the indices of all the projects that were actually participated with after cut_off time
    true_idx = np.nonzero(y_true)

    # This should now mean we have a 2D matrix which has 
    masked_pred_sim_matrix = pred_sim_matrix.iloc[true_idx]

    # Calculate the refined precision
    refined_precision = np.mean(masked_pred_sim_matrix.max(axis=0)) + precision

    if not math.isnan(refined_precision):
        precisions = precisions + [precision]
        recalls = recalls + [recall]
        refined_precisions = refined_precisions + [refined_precision]
        f.write('{ epoch: %s, precision: %s, recall: %s, refined precision: %s },' % (str(user_index), str(precision), str(recall), str(refined_precision)))
        print('{ epoch: %s, precision: %s, recall: %s, refined precision: %s },' % (str(user_index), str(precision), str(recall), str(refined_precision)))

f.write('{ precision: %s, recall: %s, refined precisions: %s } ]},' % (str(np.mean(precisions)), str(np.mean(recalls)), str(np.mean(refined_precisions))))
f.close()

print('--------------------------------')
print('Model Results')
print('Precision: ' + str(np.mean(precisions)))
print('Recall: ' + str(np.mean(recalls)))
print('Refined Precision: ' + str(np.mean(refined_precisions)))
print('Max Precision: ' + str(np.max(precisions)))
print('Max Recall: ' + str(np.max(recalls)))
print('Max Refined Precision: ' + str(np.max(refined_precisions)))
print('--------------------------------')

'''
fileName = 'cf-item-' + str(k) + '__' + strftime("%Y-%m-%d-%H-%M-%S", gmtime()) + '.txt'
f = open(fileName,"w+")
f.write(' {k: %s, results: [' % str(k))
for i in range(0, len(precisions)):
    f.write('{ epoch: %s, precision: %s, recall: %s, refined precision: %s },' % (str(i), str(precisions[i]), str(recalls[i]), str(refined_precisions[i])))
f.write('{ precision: %s, recall: %s, refined precisions: %s } ]},' % (str(np.mean(precisions)), str(np.mean(recalls)), str(np.mean(refined_precisions))))
f.close()
'''

