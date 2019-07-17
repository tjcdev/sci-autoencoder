import os
import sys
from time import gmtime, strftime

import CDAE as CDAE
import load_data as load_data
import metrics as metrics

from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import pandas as pd
import math
import keras.backend as K

batch_size = int(sys.argv[1])
epochs = int(sys.argv[2])
embedding_size = int(sys.argv[3])
recommendations = int(sys.argv[4])

# Load the proejct data
users_projects = pd.read_pickle('data/processed/active_profile_projects')

def train_test_split(users_projects_matrix):
    # Get the adjacency vector for this user
    users_projects_matrix = users_projects_matrix.drop(columns=['profile'])
        
    split_percentage = 0.2
    split_column_index = int(split_percentage * len(users_projects_matrix.columns))
    
    train = users_projects_matrix.copy()
    test = users_projects_matrix.copy()
    
    # Set a certain amount of the projects to 0
    train.iloc[:, :split_column_index] = 0
    
    return train, test, split_column_index

train, test, split_column_index = train_test_split(users_projects)

# Create our model
model = CDAE.create(I=train.shape[1], U=train.shape[0]+1, K=embedding_size,
                    hidden_activation='relu', output_activation='sigmoid', q=0.50, l=0.01)
model.compile(loss='mean_absolute_error', optimizer='adam')
model.summary()

# Train our Autoencoder
history = model.fit(x=[train, np.arange(0,train.shape[0]).reshape(train.shape[0],1)], y=train,
                    batch_size=batch_size, nb_epoch=epochs, verbose=1)

full_pred = model.predict([train, np.arange(0,train.shape[0]).reshape(train.shape[0],1)])

y_pred_floats = full_pred[:, :split_column_index]
y_pred_indices = y_pred_floats.argsort()[-recommendations:][::-1]

y_pred = np.zeros(y_pred_floats.shape)
for i in range(0, y_pred_indices.shape[1]):
    y_pred[y_pred_indices[:, i], i] = 1

y_true = test.iloc[:, :split_column_index]

# Get precision and recall
precisions, recalls, fscore, support = precision_recall_fscore_support(y_true, y_pred)

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