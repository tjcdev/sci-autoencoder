from UserProfileHelper import UserProfileHelper 
import pandas as pd
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def generateProfile(projects, userProjects):
    # Get the ids of projects that the user has interacted with
    project_ids = list(userProjects[userProjects == 1].index)

    project_titles = []

    for project in project_ids:       
        project_titles.extend(projects[projects['project_id'] == int(project)]['title'].tolist())

    # Make a 'user profile' out of these projects
    projects_fields_combined = ' '.join(project_titles)

    # Set this part of the profile to the user profile
    return projects_fields_combined

# Setup our TF-IDF model
tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words='english')

ignored_project_id = 169

projectsFile = '../Data/project_data'
all_projects = pd.read_pickle(projectsFile)

filtered_projects = all_projects[all_projects['project_id'] != ignored_project_id]

profilesProjects = pd.read_pickle('../Data/useful_profile_project_adj')

start = time.time()

# Generate the profile for this 
profilesProjects['profile_titles'] = profilesProjects.apply(lambda x: generateProfile(filtered_projects, x), axis=1)

end = time.time()
print(end - start)

fields = profilesProjects['profile_titles'].tolist()

new_project_title = str(all_projects[all_projects['project_id'] == ignored_project_id].iloc[0]['title'])

fields = fields + [new_project_title]

tfidf_matrix = tf.fit_transform(fields)

cosine_similarities = cosine_similarity(tfidf_matrix, Y=None, dense_output=True)

projects_similarities_to_users = cosine_similarities[-1][:-1]

top_10_idx = projects_similarities_to_users.argsort()[-10:][::-1]

project_pred = (projects_similarities_to_users > 0.1)*1

pred_idx = np.argwhere(project_pred).flatten()

# Bring up the original matrix for the project
print(profilesProjects.iloc[pred_idx])

print(pred_idx)