#!/bin/bash

# CDEA autoencoder with Embedding 32
screen -dm bash -c 'source activate cdea; python src/experiments/run_content_recommender_test.py 1 train_autoencoder_32_cdae_tfidf_description tfidf description;'
screen -dm bash -c 'source activate cdea; python src/experiments/run_content_recommender_test.py 5 train_autoencoder_32_cdae_tfidf_description tfidf description;'
screen -dm bash -c 'source activate cdea; python src/experiments/run_content_recommender_test.py 10 train_autoencoder_32_cdae_tfidf_description tfidf description;'

screen -dm bash -c 'source activate cdea; python src/experiments/run_content_recommender_test.py 1 train_autoencoder_32_cdae_tfidf_title tfidf title;'
screen -dm bash -c 'source activate cdea; python src/experiments/run_content_recommender_test.py 5 train_autoencoder_32_cdae_tfidf_title tfidf title;'
screen -dm bash -c 'source activate cdea; python src/experiments/run_content_recommender_test.py 10 train_autoencoder_32_cdae_tfidf_title tfidf title;'

# CDEA autoencoder with Embedding 64
screen -dm bash -c 'source activate cdea; python src/experiments/run_content_recommender_test.py 1 train_autoencoder_64_cdae_tfidf_description tfidf description;'
screen -dm bash -c 'source activate cdea; python src/experiments/run_content_recommender_test.py 5 train_autoencoder_64_cdae_tfidf_description tfidf description;'
screen -dm bash -c 'source activate cdea; python src/experiments/run_content_recommender_test.py 10 train_autoencoder_64_cdae_tfidf_description tfidf description;'

screen -dm bash -c 'source activate cdea; python src/experiments/run_content_recommender_test.py 1 train_autoencoder_64_cdae_tfidf_title tfidf title;'
screen -dm bash -c 'source activate cdea; python src/experiments/run_content_recommender_test.py 5 train_autoencoder_64_cdae_tfidf_title tfidf title;'
screen -dm bash -c 'source activate cdea; python src/experiments/run_content_recommender_test.py 10 train_autoencoder_64_cdae_tfidf_title tfidf title;'

# CDEA autoencoder with Embedding 128
screen -dm bash -c 'source activate cdea; python src/experiments/run_content_recommender_test.py 1 train_autoencoder_128_cdae_tfidf_description tfidf description;'
screen -dm bash -c 'source activate cdea; python src/experiments/run_content_recommender_test.py 5 train_autoencoder_128_cdae_tfidf_description tfidf description;'
screen -dm bash -c 'source activate cdea; python src/experiments/run_content_recommender_test.py 10 train_autoencoder_128_cdae_tfidf_description tfidf description;'

screen -dm bash -c 'source activate cdea; python src/experiments/run_content_recommender_test.py 1 train_autoencoder_128_cdae_tfidf_title tfidf title;'
screen -dm bash -c 'source activate cdea; python src/experiments/run_content_recommender_test.py 5 train_autoencoder_128_cdae_tfidf_title tfidf title;'
screen -dm bash -c 'source activate cdea; python src/experiments/run_content_recommender_test.py 10 train_autoencoder_128_cdae_tfidf_title tfidf title;'