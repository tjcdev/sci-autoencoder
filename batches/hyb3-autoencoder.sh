#!/bin/bash

# Train our New-SciStarter Collaborative Filtering Autoencoders
screen -dm bash -c 'source activate cdea; python src/models/autoencoders/train_hyb_autoencoder.py 32 100 0 hyb3 new_users_projects 0.8 description;'