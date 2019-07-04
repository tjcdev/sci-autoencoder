import numpy as np
np.random.seed(0)
from time import gmtime, strftime

import CDAE
import load_data
import metrics
from keras.models import Model
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Load the data
train_projects, train_x, test_projects, test_x, train_project_ids, test_project_ids = sci_projects.load_data()
train_x_projects = np.array(train_projects, dtype=np.int32).reshape(len(train_projects), 1)
test_x_projects = np.array(test_projects, dtype=np.int32).reshape(len(test_projects), 1)

embedding_size = 50

# Create our model
model = CDAE.create(I=train_x.shape[1], U=len(train_projects)+1, K=embedding_size,
                    hidden_activation='relu', output_activation='sigmoid', q=0.50, l=0.01)
model.compile(loss='mean_absolute_error', optimizer='adam')
model.summary()

# Train our Autoencoder
history = model.fit(x=[train_x, train_x_projects], y=train_x,
                    batch_size=128, nb_epoch=1, verbose=1,
#                    validation_split=0.2)
                    validation_data=[[test_x, test_x_projects], test_x])

pred = model.predict(x=[train_x, np.array(train_projects, dtype=np.int32).reshape(len(train_projects), 1)])

# Create a model that we will use to extract the embedding layer output
embed_model = Model(inputs=model.input, outputs=model.get_layer('embedding_layer').output)
embeddings = embed_model.predict(x=[train_x, np.array(train_projects, dtype=np.int32).reshape(len(train_projects), 1)])
embeddings = embeddings.reshape(len(train_projects), embedding_size)

# Pick the Top-N Projects and Return the Project Ids, using Cosine Similarity
similarity_matrix= pd.DataFrame(cosine_similarity(
    X=embeddings),
    index=train_project_ids)

N = 20

similar_items = pd.DataFrame(similarity_matrix.loc[25])
similar_items.columns = ["similarity_score"]
similar_items['project_id'] = train_project_ids

similar_items = similar_items.sort_values('similarity_score', ascending=False)
similar_items = similar_items.head(N)
print(similar_items[['project_id', 'similarity_score']])