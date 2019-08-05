from keras.layers import Input, Dense, Embedding, Flatten, Dropout, Add, Activation
from keras.models import Model
from keras.regularizers import l2

def create(I, U, K, hidden_activation, output_activation, q=0.5, l=0.01):
    '''
    :param I: number of items
    :param U: number of users
    :param K: number of units in hidden layer
    :param hidden_activation: activation function of hidden layer
    :param output_activation: activation function of output layer
    :param q: drop probability
    :param l: regularization parameter of L2 regularization
    :return: CDAE
    :rtype: keras.models.Model
    '''
    # Collaborative Filtering Data
    x_item = Input((I,), name='x_item')
    h_item = Dropout(q)(x_item)
    h_item = Dense(K, W_regularizer=l2(l), b_regularizer=l2(l))(h_item)

    # Content Based Data (text vectors)
    content_item = Input((1021,), name='content_item')
    h_content = Dense(K, W_regularizer=l2(l), b_regularizer=l2(l))(content_item)

    # Other input
    x_user = Input((1,), dtype='int32', name='x_user')
    h_user = Embedding(input_dim=U, output_dim=K, input_length=1, W_regularizer=l2(l), name='embedding_layer')(x_user)
    h_user = Flatten()(h_user)


    # I replaced this function as per
    # https://keras.io/layers/merge/#add
    h = Add()([h_item, h_user, h_content])
    if hidden_activation:
        h = Activation(hidden_activation)(h)
    h = Dropout(q)(h)
    y = Dense(I, activation=output_activation)(h)

    return Model(input=[x_item, content_item, x_user], output=y)