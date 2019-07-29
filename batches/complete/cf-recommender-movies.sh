#!/bin/bash

# CDEA autoencoder with Embedding Size 32 on Movies
screen -dm bash -c 'source activate cdea; python src/experiments/run_cf_recommender_test.py 1 train_autoencoder_32_cdae_movies movies;'
screen -dm bash -c 'source activate cdea; python src/experiments/run_cf_recommender_test.py 5 train_autoencoder_32_cdae_movies movies;'
screen -dm bash -c 'source activate cdea; python src/experiments/run_cf_recommender_test.py 10 train_autoencoder_32_cdae_movies movies;'

# CDEA autoencoder with Embedding Size 64 on Movies
screen -dm bash -c 'source activate cdea; python src/experiments/run_cf_recommender_test.py 1 train_autoencoder_64_cdae_movies movies;'
screen -dm bash -c 'source activate cdea; python src/experiments/run_cf_recommender_test.py 5 train_autoencoder_64_cdae_movies movies;'
screen -dm bash -c 'source activate cdea; python src/experiments/run_cf_recommender_test.py 10 train_autoencoder_64_cdae_movies movies;'

# CDEA autoencoder with Embedding Size 128 on Movies
screen -dm bash -c 'source activate cdea; python src/experiments/run_cf_recommender_test.py 1 train_autoencoder_128_cdae_movies movies;'
screen -dm bash -c 'source activate cdea; python src/experiments/run_cf_recommender_test.py 5 train_autoencoder_128_cdae_movies movies;'
screen -dm bash -c 'source activate cdea; python src/experiments/run_cf_recommender_test.py 10 train_autoencoder_128_cdae_movies movies;'
