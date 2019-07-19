# Import our python libraries libraries
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import doc2vec
from collections import namedtuple
import pandas as pd
from tqdm import tqdm
from scipy import spatial
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse

# Movie Lens Imports
from keras.utils.data_utils import get_file
from keras.utils.np_utils import to_categorical
from zipfile import ZipFile

def load_users_projects():
    # Load the adjacency matrix for users and projects
    # Note: these are only for users who have interacted with multiple projects
    users_projects =  sparse.load_npz('data/processed/active_profile_projects_matrix.npz')

    return users_projects
    

def load_movies():
    '''
    load data from MovieLens 100K Dataset
    http://grouplens.org/datasets/movielens/

    Note that this method uses ua.base and ua.test in the dataset.

    :return: train_users, train_x, test_users, test_x
    :rtype: list of int, numpy.array, list of int, numpy.array
    '''
    path = get_file('ml-100k.zip', origin='http://files.grouplens.org/datasets/movielens/ml-100k.zip')
    with ZipFile(path, 'r') as ml_zip:
        max_item_id  = -1
        train_history = {}
        with ml_zip.open('ml-100k/ua.base', 'r') as file:
            for line in file:
                user_id, item_id, rating, timestamp = line.decode('utf-8').rstrip().split('\t')
                if int(user_id) not in train_history:
                    train_history[int(user_id)] = [int(item_id)]
                else:
                    train_history[int(user_id)].append(int(item_id))

                if max_item_id < int(item_id):
                    max_item_id = int(item_id)

        test_history = {}
        with ml_zip.open('ml-100k/ua.test', 'r') as file:
            for line in file:
                user_id, item_id, rating, timestamp = line.decode('utf-8').rstrip().split('\t')
                if int(user_id) not in test_history:
                    test_history[int(user_id)] = [int(item_id)]
                else:
                    test_history[int(user_id)].append(int(item_id))

    max_item_id += 1 # item_id starts from 1
    train_users = list(train_history.keys())
    train_x = np.zeros((len(train_users), max_item_id), dtype=np.int32)
    for i, hist in enumerate(train_history.values()):
#        if i==1:
#            print hist
#            print "\n"
        mat = to_categorical(hist, max_item_id)
        train_x[i] = np.sum(mat, axis=0)
#        if i==1: print numpy.nonzero(train_x[i])

    test_users = list(test_history.keys())
    test_x = np.zeros((len(test_users), max_item_id), dtype=np.int32)
    for i, hist in enumerate(test_history.values()):
        mat = to_categorical(hist, max_item_id)
        test_x[i] = np.sum(mat, axis=0)

    return train_users, train_x, test_users, test_x