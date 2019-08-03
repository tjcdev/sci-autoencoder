#!/bin/bash

screen -dm bash -c 'source activate cdea; python src/experiments/run_cf_recommender_test.py 1 train_autoencoder_128_deep1_new_users_projects_0.8 new_users_projects;'
screen -dm bash -c 'source activate cdea; python src/experiments/run_cf_recommender_test.py 5 train_autoencoder_128_deep1_new_users_projects_0.8 new_users_projects;'
screen -dm bash -c 'source activate cdea; python src/experiments/run_cf_recommender_test.py 10 train_autoencoder_128_deep1_new_users_projects_0.8 new_users_projects;'

screen -dm bash -c 'source activate cdea; python src/experiments/run_cf_recommender_test.py 1 train_autoencoder_0_deep2_new_users_projects_0.8 new_users_projects;'
screen -dm bash -c 'source activate cdea; python src/experiments/run_cf_recommender_test.py 5 train_autoencoder_0_deep2_new_users_projects_0.8 new_users_projects;'
screen -dm bash -c 'source activate cdea; python src/experiments/run_cf_recommender_test.py 10 train_autoencoder_0_deep2_new_users_projects_0.8 new_users_projects;'

screen -dm bash -c 'source activate cdea; python src/experiments/run_cf_recommender_test.py 1 train_autoencoder_0_deep3_new_users_projects_0.8 new_users_projects;'
screen -dm bash -c 'source activate cdea; python src/experiments/run_cf_recommender_test.py 5 train_autoencoder_0_deep3_new_users_projects_0.8 new_users_projects;'
screen -dm bash -c 'source activate cdea; python src/experiments/run_cf_recommender_test.py 10 train_autoencoder_0_deep3_new_users_projects_0.8 new_users_projects;'

screen -dm bash -c 'source activate cdea; python src/experiments/run_cf_recommender_test.py 1 train_autoencoder_0_deep4_new_users_projects_0.8 new_users_projects;'
screen -dm bash -c 'source activate cdea; python src/experiments/run_cf_recommender_test.py 5 train_autoencoder_0_deep4_new_users_projects_0.8 new_users_projects;'
screen -dm bash -c 'source activate cdea; python src/experiments/run_cf_recommender_test.py 10 train_autoencoder_0_deep4_new_users_projects_0.8 new_users_projects;'
