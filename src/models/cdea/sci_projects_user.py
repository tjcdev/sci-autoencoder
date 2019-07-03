import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

'''
    This class will load in the projects data, convert it to TF-IDF vectors
    and then autoencoder these vectors
'''
def load_projects():
    # Load the full project data from the pickle file
    projects = pd.read_pickle("scistarterdata/project_data")
    
    # Get the TF-IDF for the description fields
    v = TfidfVectorizer()
    desc_idf = v.fit_transform(projects['title'])

    # Split into train and test set
    split_idx = int(np.floor((desc_idf).shape[0] * 0.8))

    train_x = desc_idf[:split_idx]
    # train_users = list(projects['project_id'].iloc[:split_idx])
    train_users = list(np.arange(0, split_idx))
   
    test_x = desc_idf[split_idx:]
    # test_users = list(projects['project_id'].iloc[split_idx:])
    test_users = list(np.arange(0, (desc_idf).shape[0]-split_idx))

    project_ids = list(projects['project_id'])

    return train_users, train_x, test_users, test_x, project_ids[:split_idx], project_ids[split_idx:]

def load_user():
    # Load the adjacency matrix for users and projects
    # Note: these are only for users who have interacted with multiple projects
    users_projects =  pd.read_pickle('scistarterdata/multiple_profile_all_projects')

    return users_projects
    '''
    # Pick out the user we want to work with
    user_projects = users_projects[users_projects['profile'] == user_id]

    if len(user_projects) == 0:
        return None

    return user_projects.drop(columns=['profile'])
    '''
    