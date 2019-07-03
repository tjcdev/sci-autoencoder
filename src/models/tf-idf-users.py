import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import traceback
from scipy import sparse

user_id = '0a82d141917f90acee9a7162b3432107'

field = 'description'

# Returns a row matching the id along with project field. 
def item(id):
    return ds.loc[ds['project_id'] == int(id)].iloc[0][field]

# Load the projects into a pandas dataframe
projects = pd.read_pickle("../../data/raw/project_data")
tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words='english')

# Adjency matrix
profiles_projects = pd.read_pickle("../../data/processed/profiles_projects")

# Get all projects the profile has participated with
profile_projects = list(profiles_projects[profiles_projects['profile'] == user_id]['project'])

# Combine all fields in all projects
projects_fields_combined = ' '.join([str(projects[projects['project_id'] == int(project)][field]) for project in profile_projects])

print("Projects already done")
print(projects_fields_combined)

# Remove the projects that have already been done from the series
project_fields = projects[~projects['project_id'].isin(profile_projects)][field]
fields = project_fields.append(pd.Series([projects_fields_combined]))

# Calculate our TF-IDF
tfidf_matrix = tf.fit_transform(fields)
print(tfidf_matrix.shape)
cosine_similarities = cosine_similarity(tfidf_matrix, Y=None, dense_output=True)
print(cosine_similarities.shape)

# The below code 'similar_indice' stores similar ids based on cosine similarity. 
#   - Sorts them in ascending order. 
#   - [:-5:-1] is then used so that the indices with most similarity are got. 
#   - 0 means no similarity and 1 means perfect similarity
similar_indices = cosine_similarities[len(cosine_similarities) - 1].argsort()[:-7:-1] 

similar_indices = similar_indices[1:]

# Stores 5 most similar books, you can change it as per your needs
similar_items = [(cosine_similarities[len(cosine_similarities) - 1][i], projects.iloc[i]['project_id']) for i in similar_indices]

results = {}
for similar_item in similar_items:
    print('--------------------')
    print('Similarity: ' + str(similar_item[0]))
    print('ID: ' + str(projects[projects['project_id'] == similar_item[1]].iloc[0]['project_id']))
    print('Title: ' + str(projects[projects['project_id'] == similar_item[1]].iloc[0]['title']))

