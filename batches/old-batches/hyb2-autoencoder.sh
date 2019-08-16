#!/bin/bash

# Train our New-SciStarter Collaborative Filtering Autoencoders
screen -dm bash -c 'source activate cdea; python src/models/autoencoders/train_hyb_autoencoder.py 32 100 32 hyb2 new_users_projects 0.8 description;'
screen -dm bash -c 'source activate cdea; python src/models/autoencoders/train_hyb_autoencoder.py 32 100 64 hyb2 new_users_projects 0.8 description;'
screen -dm bash -c 'source activate cdea; python src/models/autoencoders/train_hyb_autoencoder.py 32 100 128 hyb2 new_users_projects 0.8 description;'