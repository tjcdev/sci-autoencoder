from keras.models import load_model
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
np.random.seed(0)
from time import gmtime, strftime
from keras.models import Model
from scipy.sparse import vstack
import math

import sys

from evaluate import evaluate
from timerecommender import Recommender
import load_data as load_data

embedding_size = int(sys.argv[1])

# Load all the data we need
users_projects_list = pd.read_pickle('data/processed/profile_projects_time_consistent')
projects = pd.read_pickle('data/processed/project_data')
users_projects_matrix =  pd.read_pickle('data/processed/active_profile_projects')
similarity_matrix = pd.read_pickle('data/processed/similarity_matrix_'+str(embedding_size))

max_sim = np.max(np.max(similarity_matrix))

# Normalise the similarity
similarity_matrix = (similarity_matrix + 1) / 2

max_sim = np.max(np.max(similarity_matrix))

# Setup our recommender
rec = Recommender(projects, users_projects_matrix)

precisions = []
recalls = []
refined_precisions = []


num_uniques = lambda x: len(set([i for i in x if not math.isnan(i)]))
users_projects_list['num_projects'] = users_projects_list['projects'].apply(num_uniques)

# Loop over a certain number of users
for index, user_projects_list in users_projects_list[users_projects_list['num_projects'] > 1].iloc[:100].iterrows():
    # Get the top projects
    after_cutoff, similar_items = rec.top_projects(user_projects_list, similarity_matrix)

    # Generate our y_true and y_pred
    y_true, y_pred = rec.predictions(after_cutoff, similar_items)

    # Evaluate our model
    precision, recall, refined_precision = evaluate(y_true, y_pred, similar_items, similarity_matrix)

    if (isinstance(precision, float) 
            and isinstance(recall, float) 
            and isinstance(refined_precision, float)
            and not math.isnan(refined_precision)):
        precisions = precisions + [precision]
        recalls = recalls + [recall]
        refined_precisions = refined_precisions + [refined_precision]

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
fileName = 'cdea-results' + strftime("%Y-%m-%d-%H-%M-%S", gmtime()) + '.txt'
f = open(fileName,"w+")
for i in range(0, len(precisions)):
    f.write('{ epoch: %s, precision: %s, recall: %s },' % (str(i), str(precisions[i]), str(recalls[i])))
f.write('{ precision: %s, recall: %s },' % (str(np.mean(precisions)), str(np.mean(recalls))))
f.close()
'''