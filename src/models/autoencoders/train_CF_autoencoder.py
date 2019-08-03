# Import some standard python libraries
import numpy as np
import os
import sys
import json
import scipy.sparse as sparse
import pandas as pd


# Importing all our autoencoder models
import CDAE
import deep_1
import deep_2
import deep_3
import deep_4

# Importing our data models
dir_path = os.path.dirname(os.path.realpath(__file__))[:-23]
sys.path.append(dir_path + 'data/data_models')
from content_data import load_projects_doc2vec, load_projects_tfidf 
from cf_data import load_users_projects, load_new_users_projects, load_movies, load_profile_labels, load_new_profile_labels

# Input Parameters for training our autoencoder
batch_size = int(sys.argv[1])
epochs = int(sys.argv[2])
embedding_size = int(sys.argv[3])
autoencoder_type = str(sys.argv[4])
dataSource = str(sys.argv[5])
q = float(sys.argv[6])

# Load the data
loadData = None

if dataSource == 'users_projects':
    train_labels, train_x, val_labels, val_x, test_labels, test_x = load_users_projects()
    
    U = train_x.shape[1]
    I = train_x.shape[0]
    labels = load_profile_labels()
    labels_index = labels.index

if dataSource == 'new_users_projects':
    train_labels, train_x, val_labels, val_x, test_labels, test_x = load_new_users_projects()
    
    U = train_x.shape[1]
    I = train_x.shape[0]
    labels = load_new_profile_labels()
    labels_index = labels.index

if dataSource == 'movies':
    train_labels, train_x, test_labels, test_x = load_movies()
        
    val_x = test_x

    U = train_x.shape[1]
    I = train_x.shape[0]

    # Create a place holder labels dataframe
    train_labels = pd.DataFrame(np.arange(0, train_x.shape[1]), index=np.arange(0, train_x.shape[1]))

    labels_index = train_labels.index

autoencoder = None
if autoencoder_type == 'cdae':
    autoencoder = CDAE

if autoencoder_type == 'deep1':
    autoencoder = deep_1

if autoencoder_type == 'deep2':
    autoencoder = deep_2

if autoencoder_type == 'deep3':
    autoencoder = deep_3

if autoencoder_type == 'deep4':
    autoencoder = deep_4

# Create our autoencoder model
model = autoencoder.create(I=I, U=U, K=embedding_size,
                    hidden_activation='relu', output_activation='sigmoid', q=q, l=0.01)
model.compile(loss='mean_absolute_error', optimizer='adam')
model.summary()

# Train our autoencoder
train_x = train_x.T
val_x = val_x.T

history = model.fit(x=[train_x, labels_index], y=train_x,
                    batch_size=batch_size, nb_epoch=epochs, verbose=1,
                    validation_data=[[val_x, labels_index], val_x])

# Save history and model
name = "train_autoencoder_%s_%s_%s_%s" % (str(embedding_size), autoencoder_type, dataSource, str(q))

with open('data/raw-experiment-results/' + name + '.json', 'w') as f:
    json.dump(history.history, f)

model_name = name + '.h5'
model.save('data/raw-experiment-results/' + model_name)
