from keras.models import load_model
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
np.random.seed(0)
from time import gmtime, strftime
from keras.models import Model
from scipy.sparse import vstack
import math
from scipy import sparse

import sys
import os 

from evaluate import evaluate
from content_recommender import Recommender
from data_models.cf_data import load_users_projects
from data_models.content_data import load_projects_tfidf

k = 5 # int(sys.argv[1])
# TODO: pass in autoencoder to use
# TODO: pass in the embedding type (tfidf or doc2vec) to use

# Load the autoencoder to use
autoencoder = load_model('data/autoencoders/autoencoder_32_cdae_tfidf_desc.h5')

# Load the project data
project_train_labels, project_train_x, project_val_labels, project_val_x, project_test_labels, project_test_x = load_projects_tfidf()

# Generate the embeddings
embed_model = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('embedding_layer').output)
x = vstack([project_train_x, project_val_x, project_test_x])
x_projects = project_train_labels + project_val_labels + project_test_labels
embedding_size = autoencoder.get_layer('embedding_layer').output_shape[2]
embeddings = embed_model.predict(x=[x, np.array(x_projects.index, dtype=np.int32).reshape(len(x_projects), 1)])
embeddings = embeddings.reshape(len(x_projects), embedding_size)

# Load the users projects
# train, test = load_users_projects()

# sparse.save_npz("train.npz", train)
# sparse.save_npz("test.npz", test)

train = sparse.load_npz("train.npz")
test = sparse.load_npz("test.npz")

# Build our recommender and similarity matrix
recommender = Recommender()
similarity_matrix = recommender.similarity(embeddings)

fileName = 'results-' + str(embedding_size) + '_' + '.json'
f = open(fileName,"w+")
# Clear the current contents of the file
f.truncate(0)
f.write('{ "results": [')
for i in range(1, 10): #train.shape[0]):
    # Get the top projects that were predicted
    top_projects = recommender.top_projects(train.getrow(i), test.getrow(i), similarity_matrix, k)

    # Generate our predictions and true values
    y_true, y_pred = recommender.generate_y(test.getrow(i), top_projects)

    # Evaluate our model
    precision, recall, refined_precision = evaluate(y_true, y_pred, top_projects, similarity_matrix)

    y_pred_string = '[' + ', '.join(y_pred.astype('str')) + ']'
    y_true_string = '[' + ', '.join(y_true.astype('str')) + ']'
    # Write the results to a JSON file
    f.write('{ "user_index": %s, "precision": %s, "recall": %s, "refined_precision": %s, "y_pred": %s, "y_true": %s },' % (str(i), str(precision), str(recall), str(refined_precision), y_pred_string, y_true_string))

# Delete the last trailing comma
f.seek(f.tell() - 1, os.SEEK_SET)
f.write('')

# Close the results file
f.write(']}')
f.close()
