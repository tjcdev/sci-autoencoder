from sklearn.metrics import precision_recall_fscore_support
import numpy as np
'''
    Evaluate the performance of our recommender systems
'''
def evaluate(y_true, y_pred, similar_items, similarity_matrix):
    # Get precision and recall
    precision, recall, fscore, support = precision_recall_fscore_support(y_true, y_pred, average='binary', pos_label=1)

    # Get the similarity matrix entries for the most similar items to our 
    pred_sim_matrix = similarity_matrix[similar_items.index]

    # Get the indices of all the projects that were actually participated with after cut_off time
    true_idx = np.nonzero(y_true)

    # This should now mean we have a 2D matrix which has 
    # len(similar_items) columns 
    # len(true_idx) rows
    masked_pred_sim_matrix = pred_sim_matrix.iloc[true_idx]

    refined_precision = np.mean(masked_pred_sim_matrix.max(axis=0)) + precision

    return precision, recall, refined_precision

