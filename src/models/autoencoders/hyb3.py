from keras.layers import Input, Dense, Embedding, Flatten, Dropout, Add, Activation, Multiply
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
    h_content_1024 = Dense(1024, W_regularizer=l2(l), b_regularizer=l2(l))(content_item)
    h_content_512 = Dense(512, W_regularizer=l2(l), b_regularizer=l2(l))(content_item)

    # Other input
    x_user = Input((1,), dtype='int32', name='x_user')
    h_user = Embedding(input_dim=U, output_dim=K, input_length=1, W_regularizer=l2(l), name='embedding_layer')(x_user)
    h_user = Flatten()(h_user)

    # I replaced this function as per
    # https://keras.io/layers/merge/#add
    h = Add()([h_item, h_user, h_content])
    if hidden_activation:
        h = Activation(hidden_activation)(h)
    
    # "encoded" is the encoded representation of the input
    encoded_1 = Dense(1024, activation='relu')(h)
    encoded_1 = Dropout(q)(encoded_1)
    encoded_1 = Add()([encoded_1, h_content_1024])
    encoded_2 = Dense(512, activation='relu')(encoded_1)
    encoded_2 = Dropout(q)(encoded_2) 
    encoded_2 = Add()([encoded_2, h_content_512])
    encoded_3 = Dense(512, activation='relu')(encoded_2)
    encoded_3 = Dropout(q)(encoded_3) 
    encoded_3 = Add()([encoded_3, h_content_512])

    # "decoded" is the lossy reconstruction of the input
    decoded_1 = Dense(512, activation='relu')(encoded_3)
    decoded_2 = Dense(512, activation='relu')(decoded_1)
    decoded_3 = Dense(1024, activation='relu')(decoded_2)

    decoded = Dense(I, activation=output_activation)(decoded_3)

    return Model(input=[x_item, content_item, x_user], output=decoded)