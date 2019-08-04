#!/bin/bash

# Train our SciStarter Collaborative Filtering Autoencoders
screen -dm bash -c 'source activate cdea; python src/models/autoencoders/train_CF_autoencoder.py 32 100 128 cdae movies 0.8;'
screen -dm bash -c 'source activate cdea; python src/models/autoencoders/train_CF_autoencoder.py 32 100 0 deep3 movies 0.8;'
