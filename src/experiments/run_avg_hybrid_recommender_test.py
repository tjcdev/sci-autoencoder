import os
import sys
from time import gmtime, strftime

from keras.models import load_model
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import pandas as pd
import math
import keras.backend as K
from scipy import sparse
from scipy.sparse import vstack

dir_path = os.path.dirname(os.path.realpath(__file__))[:-15]
sys.path.append(dir_path + 'data/data_models')
sys.path.append(dir_path + 'src/models/recommenders')
from cf_recommender import CFRecommender

# Load the autoencoder to use
autoencoder_model = 'train_autoencoder_0_deep2_users_projects' #str(sys.argv[2])
dataSource = 'users_projects' # str(sys.argv[3])
model = load_model('data/autoencoders/' + autoencoder_model + '.h5')
projects = pd.read_pickle('data/processed/cf_projects_data')

from content_recommender import ContentRecommender
from gensim.models import doc2vec
from collections import namedtuple
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.models import Model
from cf_data import load_users_projects
from content_data import load_projects_tfidf, load_projects_doc2vec, load_cf_projects_tfidf

k = int(sys.argv[1])

# Load the autoencoder to use
autoencoder_model = 'train_autoencoder_32_cdae_tfidf_description'
autoencoder = load_model('data/autoencoders/' + autoencoder_model + '.h5')

field='description'
project_train_labels, project_train_x, project_val_labels, project_val_x, project_test_labels, project_test_x = load_cf_projects_tfidf(field)

# Generate the embeddings
x = vstack([project_train_x, project_val_x, project_test_x]).tocsr()
x_projects = project_train_labels + project_val_labels + project_test_labels

train = sparse.load_npz("data/processed/train.npz")
test = sparse.load_npz("data/processed/test.npz")

# Build our recommender and similarity matrix
recommender = ContentRecommender()
similarity_matrix = recommender.similarity(x)

train_labels, train_x, val_labels, val_x, test_labels, test_x = load_users_projects()

precisions = []
recalls = []
refined_precisions = []

# Pick the user
for profile_idx in range(0, len(train_labels)):
    # Collaborative Filtering Predictions
    profile_col = np.squeeze(np.asarray(train_x.getcol(profile_idx).todense())).reshape(1,-1)
    labels = np.asarray(train_labels.index)

    # Make a prediction for 
    predictions = model.predict([profile_col, labels])
    
    # Calculate the similarity between user and projects
    user_projects_sim = np.sum(np.asarray(train_x.getcol(profile_idx).todense()) * similarity_matrix.values, axis=0) / 1021

    # We can do this through masking
    projects_to_not_suggest_again = train_x.nonzero()[0]

    # Order the projects by similarity
    similar_items = pd.DataFrame(user_projects_sim)
    similar_items.columns = ['similarity_score']
    similar_items['project_id'] = similarity_matrix.columns
    similar_items = similar_items.sort_values('similarity_score', ascending=False)
    
    cb_preds = similar_items['similarity_score']
    cf_preds = pd.Series(predictions[0], index=train_labels.values.flatten())
    
    cb_sum = sum(cb_preds)
    cf_sum = sum(cf_preds)
    
    norm_cb_preds = cb_preds / cb_sum
    norm_cf_preds = cf_preds / cf_sum
    
    sum_preds = pd.Series(np.zeros(len(train_labels.values.flatten())), index=train_labels.values.flatten())
    for i in range(0, len(norm_cb_preds)):
        index = norm_cb_preds.index[i]
        pred = norm_cb_preds.iloc[i]
        result = pred + norm_cf_preds.iloc[i]
        sum_preds.loc[i] = result
        
    top_5_idx = sum_preds.sort_values(ascending=False).head(5).index
    
    top_5_project_ids = train_labels.iloc[top_5_idx].values.flatten()
    
    y_true = np.squeeze(np.asarray(test_x.getcol(profile_idx).todense())).reshape(1,-1)

    y_true = np.squeeze(np.asarray(y_true))

    predicted_projects = top_5_project_ids
    y_pred = np.zeros(y_true.shape)
    y_pred[predicted_projects] = 1
    
    # Get precision and recall
    precision, recall, fscore, support = precision_recall_fscore_support(y_true, y_pred, average='binary', pos_label=1)

    # Get the similarity matrix entries for the most similar items to our 
    pred_sim_matrix = similarity_matrix[top_5_idx]

    # Get the indices of all the projects that were actually participated with after cut_off time
    true_idx = np.nonzero(y_true)

    # This should now mean we have a 2D matrix which has 
    # len(similar_items) columns 
    # len(true_idx) rows
    masked_pred_sim_matrix = pred_sim_matrix.iloc[true_idx]

    refined_precision = np.mean(masked_pred_sim_matrix.max(axis=0)) + precision
    
    precisions = precisions + [precision]
    recalls = recalls + [recall]
    refined_precisions = refined_precisions + [refined_precision]

np.save('data/raw-experiment-results/precisions_' + str(k), np.array(precisions))
np.save('data/raw-experiment-results/recalls_' + str(k), np.array(recalls))
np.save('data/raw-experiment-results/refined_precisions_' + str(k), np.array(refined_precisions))