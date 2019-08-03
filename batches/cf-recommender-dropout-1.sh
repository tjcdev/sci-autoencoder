#!/bin/bash

# Train our New-SciStarter Collaborative Filtering Autoencoders
screen -dm bash -c 'source activate cdea; python src/experiments/run_cf_recommender_test.py 1 train_autoencoder_32_cdae_new_users_projects_0.5 new_users_projects;'
screen -dm bash -c 'source activate cdea; python src/experiments/run_cf_recommender_test.py 1 train_autoencoder_32_cdae_new_users_projects_0.65 new_users_projects;'
screen -dm bash -c 'source activate cdea; python src/experiments/run_cf_recommender_test.py 1 train_autoencoder_32_cdae_new_users_projects_0.8 new_users_projects;'

screen -dm bash -c 'source activate cdea; python src/experiments/run_cf_recommender_test.py 1 train_autoencoder_64_cdae_new_users_projects_0.5 new_users_projects;'
screen -dm bash -c 'source activate cdea; python src/experiments/run_cf_recommender_test.py 1 train_autoencoder_64_cdae_new_users_projects_0.65 new_users_projects;'
screen -dm bash -c 'source activate cdea; python src/experiments/run_cf_recommender_test.py 1 train_autoencoder_64_cdae_new_users_projects_0.8 new_users_projects;'

screen -dm bash -c 'source activate cdea; python src/experiments/run_cf_recommender_test.py 1 train_autoencoder_128_cdae_new_users_projects_0.5 new_users_projects;'
screen -dm bash -c 'source activate cdea; python src/experiments/run_cf_recommender_test.py 1 train_autoencoder_128_cdae_new_users_projects_0.65 new_users_projects;'
screen -dm bash -c 'source activate cdea; python src/experiments/run_cf_recommender_test.py 1 train_autoencoder_128_cdae_new_users_projects_0.8 new_users_projects;'