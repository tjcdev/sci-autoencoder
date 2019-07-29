#!/bin/bash

screen -dm bash -c 'source activate cdea; python src/models/autoencoders/train_CF_autoencoder.py 1024 100 0 deep2 users_projects;'

screen -dm bash -c 'source activate cdea; python src/models/autoencoders/train_CF_autoencoder.py 1024 100 0 deep2 movies;'

screen -dm bash -c 'source activate cdea; python src/models/autoencoders/train_CF_autoencoder.py 1024 100 0 deep2 new_users_projects;'

