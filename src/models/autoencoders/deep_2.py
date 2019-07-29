from keras.layers import Input, Dense, Embedding, Flatten, Dropout, Add, Activation
from keras.models import Model
from keras.regularizers import l2

'''
    This structure came from:
    Deep Autoencoders for Collaborative Filtering (Medium article)
'''
def create(I, U, K, hidden_activation, output_activation, q=0.5, l=0.01):
    # Do nothing with x_item
    x_item = Input((I,), name='x_item')
    h_item = Dropout(q)(x_item)
    h_item = Dense(K, W_regularizer=l2(l), b_regularizer=l2(l))(h_item)

    # Pass in the vector representing the users 
    x_user = Input((1,), name='x_user')
    h_user = Embedding(input_dim=U, output_dim=K, input_length=1, W_regularizer=l2(l), name='embedding_layer')(x_user)
    h_user = Flatten()(h_user)
    
    # I replaced this function as per
    # https://keras.io/layers/merge/#add
    h = Add()([h_item, h_user])
    if hidden_activation:
        h = Activation(hidden_activation)(h)

    # "encoded" is the encoded representation of the input
    encoded = Dense(K, activation='relu')(h)   
    # Middle Layer
    middle = Dense(K, activation='relu')(encoded)
    # "decoded" is the lossy reconstruction of the input
    decoded = Dense(I, activation=output_activation)(middle)

    return Model(input=[x_item, x_user], output=decoded)