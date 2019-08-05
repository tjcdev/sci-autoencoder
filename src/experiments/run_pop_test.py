import os
import sys
from time import gmtime, strftime

from keras.models import load_model
from sklearn.metrics import precision_recall_fscore_support, mean_squared_error, average_precision_score
from math import sqrt
import numpy as np
import pandas as pd
import math
import keras.backend as K

dir_path = os.path.dirname(os.path.realpath(__file__))[:-15]
sys.path.append(dir_path + 'data')
sys.path.append(dir_path + 'src/models')
from recommenders.pop_recommender import PopRecommender
from data_models.cf_data import load_users_projects, load_new_users_projects, load_movies

k = int(sys.argv[1])
autoencoder_model = 'pop' # str(sys.argv[2]) # 'train_autoencoder_32_cdae_users_projects'
dataSource = 'new_users_projects' # str(sys.argv[3]) # 'movies' 

# Load out time consistent collaborative filtering data
if dataSource == 'users_projects':
    train_labels, train_x, val_labels, val_x, test_labels, test_x = load_users_projects()

if dataSource == 'new_users_projects':
    train_labels, train_x, val_labels, val_x, test_labels, test_x = load_new_users_projects()

if dataSource == 'movies':
    train_labels, train_x, test_labels, test_x = load_movies()
    val_x = None
    val_labels = None

recommender = PopRecommender(k)

fileName = 'data/raw-experiment-results/' + autoencoder_model + '_' + str(k) + '.json'
f = open(fileName,"w+")
# Clear the current contents of the file
f.truncate(0)
f.write('[')

for profile_idx in range(0, train_x.shape[1]):
        profile_col = np.squeeze(np.asarray(train_x.getcol(profile_idx).todense())).reshape(1,-1)
        labels = np.asarray(train_labels.index)

        # Get the Top-K Recommendataions
        if k == 1:
                recommendations = np.array([0])

        if k == 5:
                recommendations = np.array([7, 11,  3,  1,  0])

        if k == 10:
                recommendations = np.array([12,  9, 14, 16, 13,  7, 11,  3,  1,  0])

        # Generate the y_pred and y_true for evaluation
        if val_x != None:
                y_true, y_pred = recommender.generate_y(recommendations, train_labels, test_x.getcol(profile_idx), val_x=val_x.getcol(profile_idx))
        else:
                y_true, y_pred = recommender.generate_y(recommendations, train_labels, test_x.getcol(profile_idx))

        # Get precision and recall
        precision, recall, fscore, support = precision_recall_fscore_support(y_true, y_pred, average='binary', pos_label=1)

        # Write the results to a JSON file
        things1 = np.nonzero(y_pred)[0].astype('str')
        things2 = np.nonzero(y_true)[0].astype('str')
        y_pred_string = '[' + ', '.join(things1) + ']'
        y_true_string = '[' + ', '.join(things2) + ']'
        f.write('{ "user_index": %s, "precision": %s, "recall": %s, "y_pred": %s, "y_true": %s },' % (str(profile_idx), str(precision), str(recall), y_pred_string, y_true_string))

# Delete the last trailing comma
f.seek(f.tell() - 1, os.SEEK_SET)
f.write('')

# Close the results file
f.write(']')
f.close()

print("-------TEST COMPLETE--------")