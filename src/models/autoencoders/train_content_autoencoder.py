# Import some standard python libraries
import numpy as np
import os
import sys
import json

# Importing all our autoencoder models
import CDAE

# Importing our data models
sys.path.append('/Users/thomascartwright/Documents/Development/sci-autoencoder/data/data_models')
from content_data import load_projects_doc2vec, load_projects_tfidf 
from cf_data import load_users_projects, load_movies, load_profile_labels

# Input Parameters for training our autoencoder
batch_size = int(sys.argv[1])
epochs = int(sys.argv[2])
embedding_size = int(sys.argv[3])
autoencoder_type = str(sys.argv[4])
dataSource = str(sys.argv[5])
field = str(sys.argv[6])

# Load the data
loadData = None
if dataSource == 'tfidf':
    train_labels, train_x, val_labels, val_x, test_labels, test_x = load_projects_tfidf(field)

if dataSource == 'doc2vec':
    train_labels, train_x, val_labels, val_x, test_labels, test_x = load_projects_doc2vec(field)


autoencoder = None
if autoencoder_type == 'cdae':
    autoencoder = CDAE

# Create our autoencoder model
model = autoencoder.create(I=train_x.shape[1], U=train_labels.shape[0] + val_labels.shape[0] + test_labels.shape[0], K=embedding_size,
                    hidden_activation='relu', output_activation='sigmoid', q=0.50, l=0.01)
model.compile(loss='mean_absolute_error', optimizer='adam')
model.summary()

# Train our autoencoder
history = model.fit(x=[train_x, train_labels.index], y=train_x,
                    batch_size=batch_size, nb_epoch=epochs, verbose=1,
                    validation_data=[[val_x, val_labels.index], val_x])

# Save history and model
name = "train_autoencoder_%s_%s_%s_%s" % (str(embedding_size), autoencoder_type, dataSource, field)

with open('data/raw-experiment-results/' + name + '.json', 'w') as f:
    json.dump(history.history, f)

model_name = name + '.h5'
model.save('data/raw-experiment-results/' + model_name)
