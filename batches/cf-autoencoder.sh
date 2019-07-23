#!/bin/bash

# Train our SciStarter Collaborative Filtering Autoencoders
screen -dm bash -c 'source activate cdea; python src/models/autoencoders/train_CF_autoencoder.py 1024 1 32 cdae users_projects;'
# screen -dm bash -c 'source activate cdea; python src/models/autoencoders/train_CF_autoencoder.py 128 100 64 cdae users_projects;'
# screen -dm bash -c 'source activate cdea; python src/models/autoencoders/train_CF_autoencoder.py 128 100 128 cdae users_projects;'

