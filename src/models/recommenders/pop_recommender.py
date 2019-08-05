import scipy.sparse as sparse
import numpy as np

class PopRecommender:
    def __init__(self, k):
        self.k = k
  
    '''
        Generate y_true and y_pred for evaluation purposes
    '''
    def generate_y(self, top_projects, all_projects, test_x, val_x=None):
        # Generate the y_true
        predicted_idx = np.asarray(top_projects)
        y_pred = np.zeros(len(all_projects))
        y_pred[predicted_idx] = 1

        # Geneate y_true
        x = test_x
        if val_x != None:
            x = val_x + test_x
            
        true_idx = x.nonzero()[0]
        y_true = np.zeros(len(all_projects))
        y_true[true_idx] = 1

        return y_true, y_pred