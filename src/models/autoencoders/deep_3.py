from keras.layers import Input, Dense, Embedding, Flatten, Dropout, Add, Activation
from keras.models import Model
from keras.regularizers import l2

'''
This structure came from:
Training Deep Autoencoders for Collaborative Filtering
'''
def create(I, U, K, hidden_activation, output_activation, q=0.5, l=0.01):
    # Do nothing with x_item
    x_item = Input((I,), name='x_item')
    h_item = Dropout(q)(x_item)
    h_item = Dense(1024, W_regularizer=l2(l), b_regularizer=l2(l))(h_item)

    # Pass in the vector representing the users 
    x_user = Input((1,), name='x_user')
    h_user = Embedding(input_dim=U, output_dim=1024, input_length=1, W_regularizer=l2(l), name='embedding_layer')(x_user)
    h_user = Flatten()(h_user)
    
    # I replaced this function as per
    # https://keras.io/layers/merge/#add
    h = Add()([h_item, h_user])
    if hidden_activation:
        h = Activation(hidden_activation)(h)

    # "encoded" is the encoded representation of the input
    encoded_1 = Dense(1024, activation='relu')(h)
    encoded_1 = Dropout(q)(encoded_1)
    encoded_2 = Dense(512, activation='relu')(encoded_1)
    encoded_2 = Dropout(q)(encoded_2) 
    encoded_3 = Dense(512, activation='relu')(encoded_2)
    encoded_3 = Dropout(q)(encoded_3) 

    # "decoded" is the lossy reconstruction of the input
    decoded_1 = Dense(512, activation='relu')(encoded_3)
    decoded_2 = Dense(512, activation='relu')(decoded_1)
    decoded_3 = Dense(1024, activation='relu')(decoded_2)

    decoded = Dense(I, activation=output_activation)(decoded_3)

    return Model(input=[x_item, x_user], output=decoded)