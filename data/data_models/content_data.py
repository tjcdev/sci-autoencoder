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

def load_projects_tfidf(field):
    # Load the full project data from the pickle file
    projects = pd.read_pickle("data/processed/project_data")

    # Get the TF-IDF for the description fields
    v = TfidfVectorizer()
    desc_idf = v.fit_transform(projects[field])

    # Train/Val/Test Split
    test_split_idx = int(np.floor(desc_idf.shape[0] * 0.8))
    val_split_idx = int(test_split_idx * 0.9)

    train_x = desc_idf[:val_split_idx]
    val_x = desc_idf[val_split_idx:test_split_idx]
    test_x = desc_idf[test_split_idx:]

    train_labels_idx = np.arange(0, val_split_idx)
    val_labels_idx = np.arange(val_split_idx, test_split_idx)
    test_labels_idx = np.arange(test_split_idx, desc_idf.shape[0])

    train_labels = pd.DataFrame(projects['project_id'].iloc[:val_split_idx], index=train_labels_idx)
    val_labels = pd.DataFrame(projects['project_id'].iloc[val_split_idx:test_split_idx], index=val_labels_idx)
    test_labels = pd.DataFrame(projects['project_id'].iloc[test_split_idx:], index=test_labels_idx)

    return train_labels, train_x, val_labels, val_x, test_labels, test_x

def load_projects_doc2vec(field):
    # Load the full project data from the pickle file
    projects = pd.read_pickle("data/processed/project_data")

    # Get the Doc2Vec for the description fields
    # Transform data (you can add more data preprocessing steps) 
    docs = []
    analyzedDocument = namedtuple('AnalyzedDocument', 'words tags')
    for idx, project in projects.iterrows():
        words = project[field].lower().split()
        tags = [project['project_id']]
        docs.append(analyzedDocument(words, tags))
    
    model = doc2vec.Doc2Vec(docs, min_count = 3, workers = 4)

    desc_idf = []
    for idx, project in projects.iterrows():
        vector = model.infer_vector(project[field])
        desc_idf = desc_idf + [vector]

    desc_idf = np.array(desc_idf)

    # Train/Val/Test Split
    test_split_idx = int(np.floor(desc_idf.shape[0] * 0.8))
    val_split_idx = int(test_split_idx * 0.9)

    train_x = desc_idf[:val_split_idx]
    val_x = desc_idf[val_split_idx:test_split_idx]
    test_x = desc_idf[test_split_idx:]

    train_labels_idx = np.arange(0, val_split_idx)
    val_labels_idx = np.arange(val_split_idx, test_split_idx)
    test_labels_idx = np.arange(test_split_idx, desc_idf.shape[0])

    train_labels = pd.DataFrame(projects['project_id'].iloc[:val_split_idx], index=train_labels_idx)
    val_labels = pd.DataFrame(projects['project_id'].iloc[val_split_idx:test_split_idx], index=val_labels_idx)
    test_labels = pd.DataFrame(projects['project_id'].iloc[test_split_idx:], index=test_labels_idx)

    return train_labels, train_x, val_labels, val_x, test_labels, test_x