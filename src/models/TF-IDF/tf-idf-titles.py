import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import traceback

# Load the projects into a pandas dataframe
ds = pd.read_pickle("../../../data/raw/project_data")
tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words='english')

# Calculate our TF-IDF
tfidf_matrix = tf.fit_transform(ds['title'])
cosine_similarities = cosine_similarity(tfidf_matrix, Y=None, dense_output=True)

results = {} # (ID : (Score,item_id))
for row_index in range(0, len(ds)):  
    idx = row_index
    row = ds.iloc[row_index]
    if idx >= 1781:
        print(row)      
    # The below code 'similar_indice' stores similar ids based on cosine similarity. 
    #   - Sorts them in ascending order. 
    #   - [:-5:-1] is then used so that the indices with most similarity are got. 
    #   - 0 means no similarity and 1 means perfect similarity
    similar_indices = cosine_similarities[idx].argsort()[:-5:-1] 

    # Stores 5 most similar books, you can change it as per your needs
    similar_items = [(cosine_similarities[idx][i], ds.iloc[i]['project_id']) for i in similar_indices]
    results[row['project_id']] = similar_items[1:]
    
# Returns a row matching the id along with project 'title'. 
def item(id):
    return ds.loc[ds['project_id'] == int(id)].iloc[0]['title']
    
# Takes the project id and number of recommendations and returns the recommendations    
def recommend(id, num):
    if (num == 0):
        print("Unable to recommend any book as you have not chosen the number of book to be recommended")
    elif (num==1):
        print("Recommending " + str(num) + " book similar to " + item(id))
        
    else :
        print("Recommending " + str(num) + " books similar to " + item(id))
        
    print("----------------------------------------------------------")
    recs = results[id][:num]
    for rec in recs:
        print("You may also like to read: " + item(rec[1]) + " (score:" + str(rec[0]) + ")")

# Pass the id of the project, second argument is the number of projects you want to be recommended
recommend(400,2)