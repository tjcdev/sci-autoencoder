import numpy
from keras.utils.data_utils import get_file
from keras.utils.np_utils import to_categorical
from zipfile import ZipFile

def load_data():
    '''
    This class is for loading the scistarter dataset

    :return: train_users, train_x, test_users, test_x
    :rtype: list of int, numpy.array, list of int, numpy.array
    '''
    # This max_item_id variable is used to figure out the dimensions of the matrix
    max_item_id  = -1
    train_history = {}
    with open('scistarterdata/train.txt', 'r') as file:
        for line in file:
            user_id, project, rating, timestamp = line.rstrip().split('\t')
            if str(user_id) not in train_history:
                train_history[str(user_id)] = [int(project)]
            else:
                train_history[str(user_id)].append(int(project))

            if max_item_id < int(project):
                max_item_id = int(project)

    test_history = {}
    with open('scistarterdata/test.txt', 'r') as file:
        for line in file:
            user_id, project, rating, timestamp = line.rstrip().split('\t')
            if str(user_id) not in test_history:
                test_history[str(user_id)] = [int(project)]
            else:
                test_history[str(user_id)].append(int(project))

    # Convert all the user_ids to integers (0-N)
    user_ids = list(train_history.keys())
    for i in range(0, len(user_ids)):
        train_history[i] = train_history.pop(user_ids[i])
        
        # Check if this user_id is also in the the test_history
        if user_ids[i] in test_history:
            test_history[i] = test_history.pop(user_ids[i])
    
    # Convert the remaining test_user_ids to integers (N-M)
    test_user_ids = [u for u in train_history.keys() if type(u) != int] 
    for i in range(0, len(test_user_ids)):
            new_id = i + len(user_ids)
            train_history[new_id] = train_history.pop(test_user_ids[i])

    max_item_id += 1 # item_id starts from 1
    train_users = list(train_history.keys())
    train_x = numpy.zeros((len(train_users), max_item_id), dtype=numpy.int32)
    for i, hist in enumerate(train_history.values()):
#        if i==1:
#            print hist
#            print "\n"
        mat = to_categorical(hist, max_item_id)
        train_x[i] = numpy.sum(mat, axis=0)
#        if i==1: print numpy.nonzero(train_x[i])

    test_users = list(test_history.keys())
    test_x = numpy.zeros((len(test_users), max_item_id), dtype=numpy.int32)
    for i, hist in enumerate(test_history.values()):
        mat = to_categorical(hist, max_item_id)
        test_x[i] = numpy.sum(mat, axis=0)

    return train_users, train_x, test_users, test_x
