from keras.models import load_model
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
np.random.seed(0)
from time import gmtime, strftime
from evaluate import evaluate
from recommender import Recommender
from keras.models import Model
from scipy.sparse import vstack
import load_data

num_users = 10

model = load_model('autoencoder.h5')

# Load the proejct data
train_projects, train_x, test_projects, test_x, train_project_ids, test_project_ids = load_data.load_projects()
train_x_projects = np.array(train_projects, dtype=np.int32).reshape(len(train_projects), 1)
test_x_projects = np.array(test_projects, dtype=np.int32).reshape(len(test_projects), 1)

# Load the users projects
users_projects = load_data.load_user()

# Create a model that we will use to extract the embedding layer output
embed_model = Model(inputs=model.input, outputs=model.get_layer('embedding_layer').output)
x = vstack([train_x, test_x])
x_projects = train_projects + test_projects
x_project_ids = train_project_ids + test_project_ids

embedding_size = model.get_layer('embedding_layer').output_shape[2]

embeddings = embed_model.predict(x=[x, np.array(x_projects, dtype=np.int32).reshape(len(x_projects), 1)])
embeddings = embeddings.reshape(len(x_projects), embedding_size)


'''
    Now we have trained an autoencoder and it's embeddings
    We want to now test our model for recommendations
'''
precisions = []
recalls = []

rec = Recommender(x_project_ids)
# Calculate our Cosine Similarity Matrix
similarity_matrix = rec.similarity(embeddings)

# Loop over all users and get the average precision and recall of the model
for index, user_projects in users_projects.iloc[:num_users].iterrows():
 
    done_projects, top_projects = rec.top_projects(similarity_matrix, user_projects.values)

    y_true, y_pred = rec.predictions(user_projects.values, done_projects, top_projects) 

    precision, recall = evaluate(y_true, y_pred)

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