# Import some standard python libraries
import numpy as np
import os
import sys
import json

# Importing all our autoencoder models
import CDAE

# Importing our data models
sys.path.append('/Users/thomascartwright/Documents/Development/sci-autoencoder/src/models/data_models')
from content_data import load_projects_doc2vec, load_projects_tfidf 
from cf_data import load_users_projects, load_movies, load_profile_labels

# Input Parameters for training our autoencoder
batch_size = 128 # int(sys.argv[1])
epochs = 2 # int(sys.argv[2])
embedding_size = 32 # int(sys.argv[3])
autoencoder_type = 'cdae' # str(sys.argv[4])
dataSource = 'users_projects' # str(sys.argv[5])

# Load the data
loadData = None
if dataSource == 'tfidf_desc':
    train_labels, train_x, val_labels, val_x, test_labels, test_x = load_projects_tfidf()
    
    U = train_labels.shape[0] + val_labels.shape[0] + test_labels.shape[0]

if dataSource == 'doc2vec_desc':
    train_labels, train_x, val_labels, val_x, test_labels, test_x = load_projects_doc2vec()

if dataSource == 'users_projects':
    train_labels, train_x, val_labels, val_x, test_labels, test_x = load_users_projects()
    
    U = train_x.shape[1]
    I = train_x.shape[0]
    labels = load_profile_labels()

if dataSource == 'movies':
    loadData = load_movies

autoencoder = None
if autoencoder_type == 'cdae':
    autoencoder = CDAE

# Create our autoencoder model
model = autoencoder.create(I=I, U=U, K=embedding_size,
                    hidden_activation='relu', output_activation='sigmoid', q=0.50, l=0.01)
model.compile(loss='mean_absolute_error', optimizer='adam')
model.summary()

# Train our autoencoder
train_x = train_x.T
val_x = val_x.T
history = model.fit(x=[train_x, labels.index], y=train_x,
                    batch_size=batch_size, nb_epoch=epochs, verbose=1,
                    validation_data=[[val_x, labels.index], val_x])

# Save history and model
name = "train_autoencoder_%s_%s_%s" % (str(embedding_size), autoencoder_type, dataSource)

with open('data/raw-experiment-results/' + name + '.json', 'w') as f:
    json.dump(history.history, f)

model_name = name + '.h5'
model.save('data/raw-experiment-results/' + model_name)
