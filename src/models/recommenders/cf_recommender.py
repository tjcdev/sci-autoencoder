import scipy.sparse as sparse

class CFRecommender:
    def __init__(self, k):
        self.k = k

    '''
        Find the project_ids of the highest predicted projects that haven't been done before
    '''
    def top_projects(self, done_vector, predicted_vector, projects):
        # Get the indices of all projects that have already been done by this 
        done_idx = done_vector.nonzero()

        # Set all the done projects to 0 in the predictions (so we don't pick them again)
        predicted_vector[done_idx] = 0

        # Get the indices of the top projects
        y_pred_indices = y_pred_floats.argsort()[-k:][::-1]

        # return the project_ids of the top recommendations
        return projects.iloc[y_pred_indices]
        
    '''
        Generate y_true and y_pred for evaluation purposes
    '''
    def generate_y(self, top_projects, all_projects, val_x, test_x):
        # Generate the y_true
        true_idx = all_projects.loc[top_projects].index
        y_true = np.zeros(len(all_projects))
        y_true[project_idx] = 1

        # Geneate y_pred
        x = val_x + test_x
        predicted_idx = x.nonzero()
        y_pred = np.zeros(len(all_projects))
        y_pred[predicted_idx] = 1

        return y_true, y_pred