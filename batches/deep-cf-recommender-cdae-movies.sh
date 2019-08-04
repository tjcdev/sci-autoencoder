#!/bin/bash
screen -dm bash -c 'source activate cdea; python src/experiments/run_cf_recommender_test.py 1 train_autoencoder_128_cdae_movies_0.8 movies;'
screen -dm bash -c 'source activate cdea; python src/experiments/run_cf_recommender_test.py 5 train_autoencoder_128_cdae_movies_0.8 movies;'
screen -dm bash -c 'source activate cdea; python src/experiments/run_cf_recommender_test.py 10 train_autoencoder_128_cdae_movies_0.8 movies;'
