import os
import sys
from time import gmtime, strftime

from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import pandas as pd
import math
import keras.backend as K

from recommenders import CFRecommender
import CDAE as CDAE
import load_data as load_data
import metrics as metrics

k = 5 #int(sys.argv[1])
autoencoder_model = 'autoencoder_32_cdae_tfidf_desc' # str(sys.argv[2]) # 'autoencoder_32_cdae_tfidf_desc'
dataSource = 'tfidf_desc' #str(sys.argv[3]) # 'tfidf_desc' 

# Load the autoencoder to use
model = load_model('data/autoencoders/' + autoencoder_model + '.h5')

# Load out time consistent collaborative filtering data
train_labels, train_x, val_labels, val_x, test_labels, test_x = load_users_projects()

recommender = CFRecommender(k)

for profile_idx in range(0, train_x.shape[1]):
    profile_col = train_x.getcol(profile_idx)

    # Make a prediction for 
    predictions = model.predict([profile_col, train_labels.index])

    # Get the Top-K Recommendataions
    recommendations = recommender.top_projects(profile_col, predictions, train_labels)

    # Generate the y_pred and y_true for evaluation
    y_true, y_pred = recommender.generate_y(recommendations, train_labels, val_x.getcol(profile_idx), test_x.getcol(profile_idx))

    precisions, recalls, fscore, support = precision_recall_fscore_support(y_true, y_pred)

    # TODO: Write the recommendations, experiment results, profiles etc. to JSON file

'''
precision_string = ', '.join([str(prec) for prec in precisions])
recall_string = ', '.join([str(recall) for recall in recalls])

np.save('autoencoder_cf_sci_y_pred_' + str(embedding_size) + '_' + str(recommendations) + '.npy', y_pred)

# Save the model
model_name = 'autoencoder_cf_sci_' + str(embedding_size) + '_' + str(recommendations) + '.h5'
model.save(model_name)

fileName = 'cf-sci-results-' + str(embedding_size) + '_' + str(recommendations) +  '_' + strftime("%Y-%m-%d-%H-%M-%S", gmtime()) + '.txt'
f = open(fileName,"w+")
f.write('{"precision": [' + precision_string + '],')
f.write('"recall": [' + recall_string + ']}')
f.close()
'''