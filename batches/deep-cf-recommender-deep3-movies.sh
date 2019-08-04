#!/bin/bash
screen -dm bash -c 'source activate cdea; python src/experiments/run_cf_recommender_test.py 1 train_autoencoder_0_deep3_movies_0.8 movies;'
screen -dm bash -c 'source activate cdea; python src/experiments/run_cf_recommender_test.py 5 train_autoencoder_0_deep3_movies_0.8movies;'
screen -dm bash -c 'source activate cdea; python src/experiments/run_cf_recommender_test.py 10 train_autoencoder_0_deep3_movies_0.8 movies;'
