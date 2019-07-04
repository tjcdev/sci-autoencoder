import numpy as np
np.random.seed(0)
from time import gmtime, strftime

import CDAE
import load_data
import metrics
from keras.models import Model
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

from recommender import Recommender

import sys

from scipy.sparse import vstack

from sklearn.metrics import precision_recall_fscore_support

batch_size = 4 #int(sys.argv[1])
epochs = 1 #int(sys.argv[2])
embedding_size = 10 #int(sys.argv[3])
num_users = 10 #int(sys.argv[4])

# Load the proejct data
train_projects, train_x, test_projects, test_x, train_project_ids, test_project_ids = load_data.load_projects()
train_x_projects = np.array(train_projects, dtype=np.int32).reshape(len(train_projects), 1)
test_x_projects = np.array(test_projects, dtype=np.int32).reshape(len(test_projects), 1)

# Load the users projects
users_projects = load_data.load_user()

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
    rec = Recommender(user_projects, similarity_matrix, x_project_ids) 
    precision, recall = rec.evaluate()

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
