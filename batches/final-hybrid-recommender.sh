#!/bin/bash

screen -dm bash -c 'source activate cdea; python src/experiments/run_hyb_recommender_test.py 1 train_autoencoder_32_hyb2_new_users_projects_0.8 new_users_projects description;'
screen -dm bash -c 'source activate cdea; python src/experiments/run_hyb_recommender_test.py 1 train_autoencoder_64_hyb2_new_users_projects_0.8 new_users_projects description;'
screen -dm bash -c 'source activate cdea; python src/experiments/run_hyb_recommender_test.py 1 train_autoencoder_128_hyb2_new_users_projects_0.8 new_users_projects description;'

screen -dm bash -c 'source activate cdea; python src/experiments/run_hyb_recommender_test.py 5 train_autoencoder_32_hyb2_new_users_projects_0.8 new_users_projects description;'
screen -dm bash -c 'source activate cdea; python src/experiments/run_hyb_recommender_test.py 5 train_autoencoder_64_hyb2_new_users_projects_0.8 new_users_projects description;'
screen -dm bash -c 'source activate cdea; python src/experiments/run_hyb_recommender_test.py 5 train_autoencoder_128_hyb2_new_users_projects_0.8 new_users_projects description;'

screen -dm bash -c 'source activate cdea; python src/experiments/run_hyb_recommender_test.py 10 train_autoencoder_32_hyb2_new_users_projects_0.8 new_users_projects description;'
screen -dm bash -c 'source activate cdea; python src/experiments/run_hyb_recommender_test.py 10 train_autoencoder_64_hyb2_new_users_projects_0.8 new_users_projects description;'
screen -dm bash -c 'source activate cdea; python src/experiments/run_hyb_recommender_test.py 10 train_autoencoder_128_hyb2_new_users_projects_0.8 new_users_projects description;'

screen -dm bash -c 'source activate cdea; python src/experiments/run_hyb_recommender_test.py 1 train_autoencoder_1024_hyb3_new_users_projects_0.8 new_users_projects description;'
screen -dm bash -c 'source activate cdea; python src/experiments/run_hyb_recommender_test.py 5 train_autoencoder_1024_hyb3_new_users_projects_0.8 new_users_projects description;'
screen -dm bash -c 'source activate cdea; python src/experiments/run_hyb_recommender_test.py 10 train_autoencoder_1024_hyb3_new_users_projects_0.8 new_users_projects description;'