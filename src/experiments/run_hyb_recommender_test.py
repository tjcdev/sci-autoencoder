import os
import sys
from time import gmtime, strftime

from keras.models import load_model
from sklearn.metrics import precision_recall_fscore_support, mean_squared_error, average_precision_score
import numpy as np
import pandas as pd
import math
import keras.backend as K
from scipy import sparse
from scipy.sparse import vstack
from gensim.models import doc2vec
from collections import namedtuple
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.models import Model

dir_path = os.path.dirname(os.path.realpath(__file__))[:-15]
sys.path.append(dir_path + 'data')
sys.path.append(dir_path + 'src/models')
from recommenders.cf_recommender import CFRecommender
from data_models.cf_data import load_users_projects, load_new_users_projects, load_movies
from data_models.content_data import load_projects_tfidf
from autoencoders import hyb2, hyb3
from recommenders.content_recommender import ContentRecommender

k = int(sys.argv[1])
autoencoder_model = str(sys.argv[2]) # 'train_autoencoder_32_cdae_users_projects'
dataSource = str(sys.argv[3]) # 'movies' 
field = str(sys.argv[4])

# Load the CF data
if dataSource == 'new_users_projects':
    train_labels, train_x, val_labels, val_x, test_labels, test_x = load_new_users_projects()

# Get the content data
project_train_labels, project_train_x, project_val_labels, project_val_x, project_test_labels, project_test_x = load_projects_tfidf(field)
x = vstack([project_train_x, project_val_x, project_test_x]).tocsr()
x_projects = project_train_labels + project_val_labels + project_test_labels

# Construct users TF-IDF
users_tf_idf = np.load('data/processed/user-project-similarity.npy')
'''
for user_index in range(0, train_x.shape[1]):
    user_project_idx = np.nonzero(train_x[:, user_index])[0]
    user_tf_idf = np.squeeze(np.asarray(x[user_project_idx].sum(axis=0)))
    users_tf_idf = vstack([users_tf_idf, user_tf_idf])
users_tf_idf = sparse.csr_matrix(users_tf_idf)
'''

# Load the autoencoder to use
model = load_model('data/autoencoders/' + autoencoder_model + '.h5')


fileName = 'data/raw-experiment-results/' + autoencoder_model + '_' + str(k) + '.json'
f = open(fileName,"w+")
# Clear the current contents of the file
f.truncate(0)
f.write('[')

recommender = CFRecommender(k)

for profile_idx in range(0, train_x.shape[1]):
    profile_col = np.squeeze(np.asarray(train_x.getcol(profile_idx).todense())).reshape(1,-1)
    labels = np.asarray(train_labels.index)
    this_users_tf_idf = users_tf_idf[profile_idx].reshape(1,-1)

    # Make a prediction for 
    predictions = model.predict([profile_col, this_users_tf_idf, labels])

    # Get the Top-K Recommendataions
    recommendations = recommender.top_projects(profile_col, predictions, train_labels)

    # Generate y_pred and y_true
    y_true, y_pred = recommender.generate_y(recommendations, train_labels, test_x.getcol(profile_idx), val_x=val_x.getcol(profile_idx))

    # Get precision and recall
    precision, recall, fscore, support = precision_recall_fscore_support(y_true, y_pred, average='binary', pos_label=1)
    avg_precision = average_precision_score(y_true, predictions.reshape(y_true.shape), average='weighted', pos_label=1)
    rmse = math.sqrt(mean_squared_error(y_true, predictions.reshape(y_true.shape)))

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