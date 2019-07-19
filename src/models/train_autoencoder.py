# Importing all our autoencoder models
from cdea import CDAE

# Importing our data models
from data import load_projects_doc2vec, load_projects_tfidf, load_users_projects, load_movies

# Import some standard python libraries
import numpy as np
import os
import sys

# Input Parameters for training our autoencoder
batch_size = 128 # int(sys.argv[1])
epochs = 100 # int(sys.argv[2])
embedding_size = 32 # int(sys.argv[3])
autoencoder_type = 'cdae' # str(sys.argv[4])
dataSource = 'tfidf-desc' # str(sys.argv[5])

# Setup the methods for loading the data
loadData = None
if dataSource == 'tfidf-desc':
    loadData = load_projects_tfidf
if dataSource == 'doc2vec-desc':
    loadData = load_projects_doc2vec
if dataSource == 'users-projects':
    loadData = load_users_projects
if dataSource == 'movies':
    loadData = load_movies

autoencoder = None
if autoencoder_type == 'cdae':
    autoencoder = CDAE

# Load the proejct data
train_projects, train_x, test_projects, test_x, train_project_ids, test_project_ids = loadData()
train_x_projects = np.array(train_projects, dtype=np.int32).reshape(len(train_projects), 1)
test_x_projects = np.array(test_projects, dtype=np.int32).reshape(len(test_projects), 1)

# Create our autoencoder model
model = autoencoder.create(I=train_x.shape[1], U=len(train_projects)+1, K=embedding_size,
                    hidden_activation='relu', output_activation='sigmoid', q=0.50, l=0.01)
model.compile(loss='mean_absolute_error', optimizer='adam')
model.summary()

# Train our autoencoder
history = model.fit(x=[train_x, train_x_projects], y=train_x,
                    batch_size=batch_size, nb_epoch=epochs, verbose=1,
#                    validation_split=0.2)
                    validation_data=[[test_x, test_x_projects], test_x])

# Save the model
model_name = 'autoencoder_' + str(embedding_size) + '.h5'
model.save(model_name)
