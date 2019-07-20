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

for i in range(1, train.shape[0]):
    top_projects = recommender.top_projects(train.getrow(i), test.getrow(i), similarity_matrix, k)

    y_true, y_pred = recommender.generate_y(test.getrow(i), top_projects)

    # Evaluate our model
    precision, recall, refined_precision = evaluate(y_true, y_pred, top_projects, similarity_matrix)

    print(precision)
    print(recall)
    print(refined_precision)

'''
# Load all the data we need
users_projects_list = pd.read_pickle('data/processed/profile_projects_time_consistent')
projects = pd.read_pickle('data/processed/project_data')
users_projects_matrix =  pd.read_pickle('data/processed/active_profile_projects')
similarity_matrix = pd.read_pickle('data/processed/similarity_matrix_'+str(embedding_size))

max_sim = np.max(np.max(similarity_matrix))

# Normalise the similarity
similarity_matrix = (similarity_matrix + 1) / 2

max_sim = np.max(np.max(similarity_matrix))

# Setup our recommender
rec = Recommender(projects, users_projects_matrix)

precisions = []
recalls = []
refined_precisions = []


num_uniques = lambda x: len(set([i for i in x if not math.isnan(i)]))
users_projects_list['num_projects'] = users_projects_list['projects'].apply(num_uniques)

# Loop over a certain number of users
for index, user_projects_list in users_projects_list[users_projects_list['num_projects'] > 1].iloc[:1000].iterrows():
    # Get the top projects
    after_cutoff, similar_items = rec.top_projects(user_projects_list, similarity_matrix, k)

    # Generate our y_true and y_pred
    y_true, y_pred = rec.predictions(after_cutoff, similar_items)

    # Evaluate our model
    precision, recall, refined_precision = evaluate(y_true, y_pred, similar_items, similarity_matrix)

    if (isinstance(precision, float) 
            and isinstance(recall, float) 
            and isinstance(refined_precision, float)
            and not math.isnan(refined_precision)):
        precisions = precisions + [precision]
        recalls = recalls + [recall]
        refined_precisions = refined_precisions + [refined_precision]

print('--------------------------------')
print('Model Results')
print('Precision: ' + str(np.mean(precisions)))
print('Recall: ' + str(np.mean(recalls)))
print('Refined Precision: ' + str(np.mean(refined_precisions)))
print('Max Precision: ' + str(np.max(precisions)))
print('Max Recall: ' + str(np.max(recalls)))
print('Max Refined Precision: ' + str(np.max(refined_precisions)))
print('--------------------------------')


fileName = 'cdea-results-' + str(embedding_size) + '_' + strftime("%Y-%m-%d-%H-%M-%S", gmtime()) + '.txt'
f = open(fileName,"w+")
for i in range(0, len(precisions)):
    f.write('{ epoch: %s, precision: %s, recall: %s },' % (str(i), str(precisions[i]), str(recalls[i])))
f.write('{ precision: %s, recall: %s },' % (str(np.mean(precisions)), str(np.mean(recalls))))
f.close()
'''