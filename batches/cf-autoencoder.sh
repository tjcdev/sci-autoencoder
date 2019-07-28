#!/bin/bash

# Train our SciStarter Collaborative Filtering Autoencoders
screen -dm bash -c 'source activate cdea; python src/models/autoencoders/train_CF_autoencoder.py 1024 100 32 cdae users_projects;'