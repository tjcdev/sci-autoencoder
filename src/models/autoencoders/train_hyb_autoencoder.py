import os
import sys
from time import gmtime, strftime
import json

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

# Importing our data models
dir_path = os.path.dirname(os.path.realpath(__file__))[:-23]
sys.path.append(dir_path + 'data/data_models')
sys.path.append(dir_path + 'src/models')

from content_data import load_projects_doc2vec, load_projects_tfidf 
from cf_data import load_users_projects, load_new_users_projects, load_movies, load_profile_labels, load_new_profile_labels
from recommenders.cf_recommender import CFRecommender
from recommenders.content_recommender import ContentRecommender
from autoencoders import hyb2, hyb3

# 32 100 0 hyb3 new_users_projects 0.8 description

# Input Parameters for training our autoencoder
batch_size = 32 #int(sys.argv[1]) # 32
epochs = 40 #int(sys.argv[2]) # 100
embedding_size = 1024 #int(sys.argv[3]) # 32
autoencoder_type = 'hyb3' #str(sys.argv[4]) # 'hyb1'
dataSource = 'new_users_projects' #str(sys.argv[5]) # 'new_users_projects'
q = 0.8 #float(sys.argv[6]) # 0.8
field = 'description' # str(sys.argv[7]) # 'description'

# Load out time consistent collaborative filtering data
train_labels, train_x, val_labels, val_x, test_labels, test_x = load_new_users_projects()
U = train_x.shape[1]
I = train_x.shape[0]

# Load content data
project_train_labels, project_train_x, project_val_labels, project_val_x, project_test_labels, project_test_x = load_projects_tfidf(field)

# Generate the embeddings
x = vstack([project_train_x, project_val_x, project_test_x]).tocsr()
x_projects = project_train_labels + project_val_labels + project_test_labels


autoencoder = None
if autoencoder_type == 'hyb2':
    autoencoder = hyb2

if autoencoder_type == 'hyb3':
    autoencoder = hyb3

# Create a TF_IDF matrix for all users
users_tf_idf = np.load('data/processed/user-project-similarity.npy')
'''
for user_index in range(0, train_x.shape[1]):
    user_project_idx = np.nonzero(train_x[:, user_index])[0]
    user_tf_idf = np.squeeze(np.asarray(x[user_project_idx].sum(axis=0)))
    users_tf_idf = vstack([users_tf_idf, user_tf_idf])
users_tf_idf = sparse.csr_matrix(users_tf_idf)
'''

# Create the autoencoder
model = autoencoder.create(I=I, U=U, K=embedding_size,
                    hidden_activation='relu', output_activation='sigmoid', q=q, l=0.001)
model.compile(loss='mean_absolute_error', optimizer='adam')
model.summary()

# Prepare the data for training our autoencoder
train_x_t = train_x.T
val_x_t = val_x.T
test_x_t = test_x.T

train_val_x = train_x_t + val_x_t
train_test_x = train_x_t + test_x_t

users = np.arange(0, train_x_t.shape[0])

# Train the autoencoder
history = model.fit(x=[train_x_t, users_tf_idf, users], y=train_x_t,
                    batch_size=batch_size, nb_epoch=epochs, verbose=1,
                    validation_data=[[train_x_t, users_tf_idf, users], train_val_x])

# Save history and model
name = "train_autoencoder_%s_%s_%s_%s" % (str(embedding_size), autoencoder_type, dataSource, str(q))

with open('data/raw-experiment-results/' + name + '.json', 'w') as f:
    json.dump(history.history, f)

model_name = name + '.h5'
model.save('data/raw-experiment-results/' + model_name)