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
dir_path = os.path.dirname(os.path.realpath(__file__))[:-15]
sys.path.append(dir_path + 'data/data_models')
sys.path.append(dir_path + 'src/models/recommenders')

from content_recommender import ContentRecommender as Recommender
from cf_data import load_users_projects
from content_data import load_projects_tfidf, load_projects_doc2vec

k = int(sys.argv[1])
autoencoder_model = str(sys.argv[2]) # 'autoencoder_32_cdae_tfidf_desc'
dataSource = str(sys.argv[3]) # 'tfidf' 
field = str(sys.argv[4])

# Load the autoencoder to use
autoencoder = load_model('data/autoencoders/' + autoencoder_model + '.h5')

# Load the project data
loadData = None
if dataSource == 'tfidf':
    loadData = load_projects_tfidf
if dataSource == 'doc2vec':
    loadData = load_projects_doc2vec

project_train_labels, project_train_x, project_val_labels, project_val_x, project_test_labels, project_test_x = loadData(field)

# Generate the embeddings
embed_model = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('embedding_layer').output)
x = vstack([project_train_x, project_val_x, project_test_x]).tocsr()
x_projects = project_train_labels + project_val_labels + project_test_labels
embedding_size = autoencoder.get_layer('embedding_layer').output_shape[2]
other_x = np.array(x_projects.index, dtype=np.int32).reshape(len(x_projects), 1).flatten()
embeddings = embed_model.predict(x=[x, other_x])
embeddings = embeddings.reshape(len(x_projects), embedding_size)

train = sparse.load_npz("data/processed/train.npz")
test = sparse.load_npz("data/processed/test.npz")

# Build our recommender and similarity matrix
recommender = Recommender()
similarity_matrix = recommender.similarity(embeddings)

fileName = 'data/raw-experiment-results/results-' + autoencoder_model + '_' + str(k) + '.json'
f = open(fileName,"w+")
# Clear the current contents of the file
f.truncate(0)
f.write('[')
for i in range(1, train.shape[0]):
    # Get the top projects that were predicted
    top_projects = recommender.top_projects(train.getrow(i), test.getrow(i), similarity_matrix, k)

    # Generate our predictions and true values
    y_true, y_pred = recommender.generate_y(test.getrow(i), top_projects)

    # Evaluate our model
    precision, recall, refined_precision = evaluate(y_true, y_pred, top_projects, similarity_matrix)

    things1 = np.nonzero(y_pred)[0].astype('str')
    things2 = np.nonzero(y_true)[0].astype('str')
    y_pred_string = '[' + ', '.join(things1) + ']'
    y_true_string = '[' + ', '.join(things2) + ']'

    if math.isnan(refined_precision):
        refined_precision = -1

    # Write the results to a JSON file
    f.write('{ "user_index": %s, "precision": %s, "recall": %s, "refined_precision": %s, "y_pred": %s, "y_true": %s },' % (str(i), str(precision), str(recall), str(refined_precision), y_pred_string, y_true_string))

# Delete the last trailing comma
f.seek(f.tell() - 1, os.SEEK_SET)
f.write('')

# Close the results file
f.write(']')
f.close()

print("-------TEST COMPLETE--------")
