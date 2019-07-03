import numpy as np
np.random.seed(0)
from time import gmtime, strftime

import CDAE
import sci_projects_user
import metrics
from keras.models import Model
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

import sys

from scipy.sparse import vstack

from sklearn.metrics import precision_recall_fscore_support

batch_size = int(sys.argv[1])
epochs = int(sys.argv[2])
embedding_size = int(sys.argv[3])
num_users = int(sys.argv[4])

def evaluate_user_threshold(user_projects, similarity_matrix, project_ids):
    # Percentage of done projects to set to 0
    perc_projects = 0.1

    # Cut out some of the projects that are done to test our model
    ones_idx = np.nonzero(user_projects)
    indices_to_cut = np.random.choice(len(ones_idx), int(np.ceil(perc_projects * len(ones_idx))), replace=False)
    user_projects[indices_to_cut] = 0

    # Calculate the similarity between user and projects
    num_projects = np.count_nonzero(user_projects)
    if num_projects == 0:
        num_projects = 1
    user_projects_sim = np.sum(user_projects * similarity_matrix.values, axis=1) / num_projects

    similar_items = pd.DataFrame(user_projects_sim)
    similar_items.columns = ['similarity_score']
    similar_items['project_id'] = project_ids

    # Measure the Precision and Recall
    y_true = user_projects
    predictions = np.array(similar_items['similarity_score'])
    y_pred = (predictions > 0.2)*1

    precision, recall, fscore, support = precision_recall_fscore_support(y_true, y_pred, average='macro')

    return precision, recall

def evaluate_user_top_N(user_projects, similarity_matrix, project_ids):
    indices_of_done_projects, top_N = get_top_N(np.array(user_projects, copy=True), similarity_matrix, project_ids)
    top_N_project_ids = list(top_N['project_id'])

    # Filter out
    user_projects[indices_of_done_projects] = 0
    y_true = user_projects

    project_ids = list(similarity_matrix.index)

    y_pred = [project_id in top_N_project_ids for project_id in project_ids]*1
    
    # Check the predicted projects and the true projects
    ones_idx = np.nonzero(y_true)
    true_projects = np.array(project_ids)[np.array(ones_idx).flatten()]
    true_ones = len(np.array(ones_idx).flatten())

    ones_idx = np.nonzero(y_pred)
    pred_projects = np.array(project_ids)[np.array(ones_idx).flatten()]
    pred_ones = len(np.array(ones_idx).flatten())

    precision, recall, fscore, support = precision_recall_fscore_support(y_true, y_pred, average='binary', pos_label=1)

    return precision, recall


def get_top_N(user_projects, similarity_matrix, project_ids):
    # Cut out some of the projects that are done to test our model
    perc_projects = 0.2
    ones_idx = np.nonzero(user_projects)
    num_ones = len(np.array(ones_idx).flatten())
    to_cut = np.random.choice(num_ones, int(np.ceil(perc_projects * num_ones)), replace=False)
    ones_indices_to_cut = np.array(ones_idx).flatten()[to_cut]
    user_projects[ones_indices_to_cut] = 0

    # Calculate the similarity between user and projects
    num_projects = np.count_nonzero(user_projects)
    user_projects_sim = np.sum(user_projects * similarity_matrix.values, axis=1) / num_projects

    similar_items = pd.DataFrame(user_projects_sim)
    similar_items.columns = ['similarity_score']
    similar_items['project_id'] = project_ids

    # Filter out projects already done (only if we are not doing the precision and recall)
    indices_of_done_projects = list(np.nonzero(user_projects))[0]
    done_projects = similar_items.iloc[indices_of_done_projects]
    # TODO: check that this definitely removes the projects that have been done already
    similar_items = similar_items[~similar_items['project_id'].isin(list(done_projects['project_id']))]

    # Pick the Top-N item
    N = max([20, to_cut.shape[0]*4])
    similar_items = similar_items.sort_values('similarity_score', ascending=False)
    similar_items = similar_items.head(N)
    return indices_of_done_projects, similar_items[['project_id', 'similarity_score']]

# Load the proejct data
train_projects, train_x, test_projects, test_x, train_project_ids, test_project_ids = sci_projects_user.load_projects()
train_x_projects = np.array(train_projects, dtype=np.int32).reshape(len(train_projects), 1)
test_x_projects = np.array(test_projects, dtype=np.int32).reshape(len(test_projects), 1)

# Load the users projects
users_projects = sci_projects_user.load_user()

# Create our model
model = CDAE.create(I=train_x.shape[1], U=len(train_projects)+1, K=embedding_size,
                    hidden_activation='relu', output_activation='sigmoid', q=0.50, l=0.01)
model.compile(loss='mean_absolute_error', optimizer='adam')
model.summary()

# Train our Autoencoder
history = model.fit(x=[train_x, train_x_projects], y=train_x,
                    batch_size=batch_size, nb_epoch=epochs, verbose=1,
#                    validation_split=0.2)
                    validation_data=[[test_x, test_x_projects], test_x])

# pred = model.predict(x=[train_x, np.array(train_projects, dtype=np.int32).reshape(len(train_projects), 1)])

# Create a model that we will use to extract the embedding layer output
embed_model = Model(inputs=model.input, outputs=model.get_layer('embedding_layer').output)
x = vstack([train_x, test_x])
x_projects = train_projects + test_projects
x_project_ids = train_project_ids + test_project_ids

embeddings = embed_model.predict(x=[x, np.array(x_projects, dtype=np.int32).reshape(len(x_projects), 1)])
embeddings = embeddings.reshape(len(x_projects), embedding_size)

# Calculate our Cosine Similarity Matrix
similarity_matrix= pd.DataFrame(cosine_similarity(
    X=embeddings),
    index=x_project_ids)
'''
    Now we have trained an autoencoder and it's embeddings
    We want to now test our model for recommendations
'''

# Loop over all users and get the average precision and recall of the model
precisions = []
recalls = []
cnt = 0
for index, user_projects in users_projects.iloc[:num_users].iterrows():
    cnt += 1
    print(str(cnt) + '/' + str(len(users_projects)))
    user_projects = np.array(user_projects.values[1:].tolist())
    # train_user_projects = user_projects[:len(train_project_ids)]
    # test_user_projects = user_projects[len(train_project_ids):]

    # precision, recall = evaluate_user_threshold(test_user_projects, similarity_matrix, test_project_ids)
    precision, recall = evaluate_user_top_N(user_projects, similarity_matrix, x_project_ids)

    if (isinstance(precision, float) and isinstance(recall, float)):
        precisions = precisions + [precision]
        recalls = recalls + [recall]

fileName = 'cdea-results' + strftime("%Y-%m-%d-%H-%M-%S", gmtime()) + '.txt'
f = open(fileName,"w+")
for i in range(0, len(precisions)):
    f.write('{ epoch: %s, precision: %s, recall: %s },' % (str(i), str(precisions[i]), str(recalls[i])))
f.write('{ precision: %s, recall: %s },' % (str(np.mean(precisions)), str(np.mean(recalls))))
f.close()

print('--------------------------------')
print('Model Results')
print('Precision: ' + str(np.mean(precisions)))
print('Recall: ' + str(np.mean(recalls)))
print('Max Precision: ' + str(np.max(precisions)))
print('Max Recall: ' + str(np.max(recalls)))
print('--------------------------------')

'''
similar_items = get_top_N(train_user_projects, similarity_matrix)
print('--------------------------------')
print(similar_items)
print('--------------------------------')
'''
