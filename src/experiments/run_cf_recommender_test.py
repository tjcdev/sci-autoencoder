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

def get_weighted_loss():
    def weighted_loss(y_true, y_pred):
        return K.mean((0.01*(1-y_true))*(0.99*(y_true))*K.binary_crossentropy(y_true, y_pred), axis=-1)
    return weighted_loss

dir_path = os.path.dirname(os.path.realpath(__file__))[:-15]
sys.path.append(dir_path + 'data')
sys.path.append(dir_path + 'src/models')
from recommenders.cf_recommender import CFRecommender
from data_models.cf_data import load_users_projects, load_new_users_projects, load_movies

k = 10 # int(sys.argv[1])
autoencoder_model = 'deep3-custom_loss-1-example' #str(sys.argv[2]) # 'train_autoencoder_32_cdae_users_projects'
dataSource = 'new_users_projects'# str(sys.argv[3]) # 'movies' 

# Load the autoencoder to use
model = load_model('data/autoencoders/' + autoencoder_model + '.h5')

# Load out time consistent collaborative filtering data
if dataSource == 'users_projects':
    train_labels, train_x, val_labels, val_x, test_labels, test_x = load_users_projects()

if dataSource == 'new_users_projects':
    train_labels, train_x, val_labels, val_x, test_labels, test_x = load_new_users_projects()

if dataSource == 'movies':
    train_labels, train_x, test_labels, test_x = load_movies()
    val_x = None
    val_labels = None

recommender = CFRecommender(k)

fileName = 'data/raw-experiment-results/' + autoencoder_model + '_' + str(k) + '.json'
f = open(fileName,"w+")
# Clear the current contents of the file
f.truncate(0)
f.write('[')

for profile_idx in range(0, train_x.shape[1]):
    profile_col = np.squeeze(np.asarray(train_x.getcol(profile_idx).todense())).reshape(1,-1)
    labels = np.asarray(train_labels.index)

    # Make a prediction for 
    predictions = model.predict([profile_col, labels])

    # Get the Top-K Recommendataions
    recommendations = recommender.top_projects(profile_col, predictions, train_labels)

    things = recommendations

    # Generate the y_pred and y_true for evaluation
    if val_x != None:
        y_true, y_pred = recommender.generate_y(recommendations, train_labels, test_x.getcol(profile_idx), val_x=val_x.getcol(profile_idx))
    else:
        y_true, y_pred = recommender.generate_y(recommendations, train_labels, test_x.getcol(profile_idx))

    # Get precision and recall
    precision, recall, fscore, support = precision_recall_fscore_support(y_true, y_pred, average='binary', pos_label=1)
    avg_precision = average_precision_score(y_true, predictions.reshape(y_true.shape), average='weighted', pos_label=1)
    rmse = sqrt(mean_squared_error(y_true, predictions.reshape(y_true.shape)))

    if math.isnan(avg_precision):
        avg_precision = 0
    if math.isnan(rmse):
        rmse = 0

    # Write the results to a JSON file
    things1 = np.nonzero(y_pred)[0].astype('str')
    things2 = np.nonzero(y_true)[0].astype('str')
    y_pred_string = '[' + ', '.join(things1) + ']'
    y_true_string = '[' + ', '.join(things2) + ']'
    f.write('{ "user_index": %s, "precision": %s, "recall": %s, "y_pred": %s, "y_true": %s, "avg_precision": %s, "rmse": %s },' % (str(profile_idx), str(precision), str(recall), y_pred_string, y_true_string, str(avg_precision), str(rmse)))

# Delete the last trailing comma
f.seek(f.tell() - 1, os.SEEK_SET)
f.write('')

# Close the results file
f.write(']')
f.close()

print("-------TEST COMPLETE--------")