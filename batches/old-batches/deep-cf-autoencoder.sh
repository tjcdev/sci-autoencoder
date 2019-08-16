#!/bin/bash

# Train our SciStarter Collaborative Filtering Autoencoders
screen -dm bash -c 'source activate cdea; python src/models/autoencoders/train_CF_autoencoder.py 32 100 32 deep1 new_users_projects 0.8;'
screen -dm bash -c 'source activate cdea; python src/models/autoencoders/train_CF_autoencoder.py 32 100 64 deep1 new_users_projects 0.8;'
screen -dm bash -c 'source activate cdea; python src/models/autoencoders/train_CF_autoencoder.py 32 100 128 deep1 new_users_projects 0.8;'

screen -dm bash -c 'source activate cdea; python src/models/autoencoders/train_CF_autoencoder.py 32 100 0 deep2 new_users_projects 0.8;'
screen -dm bash -c 'source activate cdea; python src/models/autoencoders/train_CF_autoencoder.py 32 100 0 deep3 new_users_projects 0.8;'
screen -dm bash -c 'source activate cdea; python src/models/autoencoders/train_CF_autoencoder.py 32 100 0 deep4 new_users_projects 0.8;'