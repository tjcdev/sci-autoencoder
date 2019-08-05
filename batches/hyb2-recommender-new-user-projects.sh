#!/bin/bash

screen -dm bash -c 'source activate cdea; python src/experiments/run_cf_recommender_test.py 1 train_autoencoder_128_deep1_new_users_projects_0.8 new_users_projects;'