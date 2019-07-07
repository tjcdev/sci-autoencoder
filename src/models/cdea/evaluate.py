from sklearn.metrics import precision_recall_fscore_support

'''
    Evaluate the performance of our recommender systems
'''
def evaluate(y_true, y_pred, similarity_matrix = None):
    precision, recall, fscore, support = precision_recall_fscore_support(y_true, y_pred, average='binary', pos_label=1)

    refined_precision = None
    # Calculate the refined precision
    if similarity_matrix is not None:
        pred_sim_matrix = similarity_matrix[similar_items.index]
        true_idx = np.nonzero(y_true)

        masked_pred_sim_matrix = pred_sim_matrix.iloc[true_idx]

        refined_precision = np.mean(masked_pred_sim_matrix.max(axis=0)) + precision

    return precision, recall, refined_precision

