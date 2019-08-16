#!/bin/bash

# Train our TF-IDF autoencoders
screen -dm bash -c 'source activate cdea; python src/models/autoencoders/train_content_autoencoder.py 32 100 32 cdae tfidf description;'
screen -dm bash -c 'source activate cdea; python src/models/autoencoders/train_content_autoencoder.py 32 100 64 cdae tfidf description;'
screen -dm bash -c 'source activate cdea; python src/models/autoencoders/train_content_autoencoder.py 32 100 128 cdae tfidf description;'
screen -dm bash -c 'source activate cdea; python src/models/autoencoders/train_content_autoencoder.py 32 100 32 cdae tfidf title;'
screen -dm bash -c 'source activate cdea; python src/models/autoencoders/train_content_autoencoder.py 32 100 64 cdae tfidf title;'
screen -dm bash -c 'source activate cdea; python src/models/autoencoders/train_content_autoencoder.py 32 100 128 cdae tfidf title;'


# Train our Doc2Vec autoencoders
screen -dm bash -c 'source activate cdea; python src/models/autoencoders/train_content_autoencoder.py 32 100 32 cdae doc2vec description;'
screen -dm bash -c 'source activate cdea; python src/models/autoencoders/train_content_autoencoder.py 32 100 64 cdae doc2vec description;'
screen -dm bash -c 'source activate cdea; python src/models/autoencoders/train_content_autoencoder.py 32 100 128 cdae doc2vec description;'
screen -dm bash -c 'source activate cdea; python src/models/autoencoders/train_content_autoencoder.py 32 100 32 cdae doc2vec title;'
screen -dm bash -c 'source activate cdea; python src/models/autoencoders/train_content_autoencoder.py 32 100 64 cdae doc2vec title;'
screen -dm bash -c 'source activate cdea; python src/models/autoencoders/train_content_autoencoder.py 32 100 128 cdae doc2vec title;'
