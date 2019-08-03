#!/bin/bash

# Train our New-SciStarter Collaborative Filtering Autoencoders
screen -dm bash -c 'source activate cdea; python src/models/autoencoders/train_CF_autoencoder.py 1024 100 32 cdae new_users_projects 0.5;'
screen -dm bash -c 'source activate cdea; python src/models/autoencoders/train_CF_autoencoder.py 1024 100 32 cdae new_users_projects 0.65;'
screen -dm bash -c 'source activate cdea; python src/models/autoencoders/train_CF_autoencoder.py 1024 100 32 cdae new_users_projects 0.8;'

screen -dm bash -c 'source activate cdea; python src/models/autoencoders/train_CF_autoencoder.py 1024 100 64 cdae new_users_projects 0.5;'
screen -dm bash -c 'source activate cdea; python src/models/autoencoders/train_CF_autoencoder.py 1024 100 64 cdae new_users_projects 0.65;'
screen -dm bash -c 'source activate cdea; python src/models/autoencoders/train_CF_autoencoder.py 1024 100 64 cdae new_users_projects 0.8;'

screen -dm bash -c 'source activate cdea; python src/models/autoencoders/train_CF_autoencoder.py 1024 100 128 cdae new_users_projects 0.5;'
screen -dm bash -c 'source activate cdea; python src/models/autoencoders/train_CF_autoencoder.py 1024 100 128 cdae new_users_projects 0.65;'
screen -dm bash -c 'source activate cdea; python src/models/autoencoders/train_CF_autoencoder.py 1024 100 128 cdae new_users_projects 0.8;'