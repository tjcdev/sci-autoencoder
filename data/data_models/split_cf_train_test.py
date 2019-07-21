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

from cf_data import load_users_projects

# Load the users projects
train, test = load_users_projects()

sparse.save_npz("train.npz", train)
sparse.save_npz("test.npz", test)