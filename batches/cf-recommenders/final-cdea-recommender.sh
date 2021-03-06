#!/bin/bash

# CDEA autoencoder with Embedding Size 32 on UserProjects
screen -dm bash -c 'source activate cdea; python src/experiments/run_cf_recommender_test.py 1 train_autoencoder_32_cdae_new_users_projects_0.8 new_users_projects;'
screen -dm bash -c 'source activate cdea; python src/experiments/run_cf_recommender_test.py 5 train_autoencoder_32_cdae_new_users_projects_0.8 new_users_projects;'
screen -dm bash -c 'source activate cdea; python src/experiments/run_cf_recommender_test.py 10 train_autoencoder_32_cdae_new_users_projects_0.8 new_users_projects;'

screen -dm bash -c 'source activate cdea; python src/experiments/run_cf_recommender_test.py 1 train_autoencoder_64_cdae_new_users_projects_0.8 new_users_projects;'
screen -dm bash -c 'source activate cdea; python src/experiments/run_cf_recommender_test.py 5 train_autoencoder_64_cdae_new_users_projects_0.8 new_users_projects;'
screen -dm bash -c 'source activate cdea; python src/experiments/run_cf_recommender_test.py 10 train_autoencoder_64_cdae_new_users_projects_0.8 new_users_projects;'

screen -dm bash -c 'source activate cdea; python src/experiments/run_cf_recommender_test.py 1 train_autoencoder_128_cdae_new_users_projects_0.8 new_users_projects;'
screen -dm bash -c 'source activate cdea; python src/experiments/run_cf_recommender_test.py 5 train_autoencoder_128_cdae_new_users_projects_0.8 new_users_projects;'
screen -dm bash -c 'source activate cdea; python src/experiments/run_cf_recommender_test.py 10 train_autoencoder_128_cdae_new_users_projects_0.8 new_users_projects;'

