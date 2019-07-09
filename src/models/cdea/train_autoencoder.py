import CDAE
import load_data
import metrics

import sys

from sklearn.metrics import precision_recall_fscore_support
import numpy as np

batch_size = int(sys.argv[1])
epochs = int(sys.argv[2])
embedding_size = int(sys.argv[3])

# Load the proejct data
train_projects, train_x, test_projects, test_x, train_project_ids, test_project_ids = load_data.load_projects()
train_x_projects = np.array(train_projects, dtype=np.int32).reshape(len(train_projects), 1)
test_x_projects = np.array(test_projects, dtype=np.int32).reshape(len(test_projects), 1)

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

# Save the model
model_name = 'autoencoder_' + str(embedding_size) + '.h5'
model.save(model_name)
