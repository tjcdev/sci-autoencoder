import os
import sys
from time import gmtime, strftime

from keras.models import load_model
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import pandas as pd
import math
import keras.backend as K

sys.path.append('/Users/thomascartwright/Documents/Development/sci-autoencoder/src/models')
sys.path.append('/Users/thomascartwright/Documents/Development/sci-autoencoder/data')
from recommenders.cf_recommender import CFRecommender
from data_models.cf_data import load_users_projects

k = 5 #int(sys.argv[1])
autoencoder_model = 'autoencoder_32_cdae_users_projects' # str(sys.argv[2]) # 'autoencoder_32_cdae_tfidf_desc'
dataSource = 'tfidf_desc' #str(sys.argv[3]) # 'tfidf_desc' 

# Load the autoencoder to use
model = load_model('data/autoencoders/' + autoencoder_model + '.h5')

# Load out time consistent collaborative filtering data
train_labels, train_x, val_labels, val_x, test_labels, test_x = load_users_projects()

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

    # Generate the y_pred and y_true for evaluation
    y_true, y_pred = recommender.generate_y(recommendations, train_labels, val_x.getcol(profile_idx), test_x.getcol(profile_idx))

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