#!/bin/bash

# Train our Movies Collaborative Filtering Autoencoders
screen -dm bash -c 'source activate cdea; python src/models/autoencoders/train_CF_autoencoder.py 128 100 32 cdae movies;'
screen -dm bash -c 'source activate cdea; python src/models/autoencoders/train_CF_autoencoder.py 128 100 64 cdae movies;'
screen -dm bash -c 'source activate cdea; python src/models/autoencoders/train_CF_autoencoder.py 128 100 128 cdae movies;'