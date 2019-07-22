#!/bin/bash

# CDEA autoencoder with Embedding 32
screen -dm bash -c 'source activate cdea; python src/experiments/run_content_recommender_test.py 1 train_autoencoder_32_cdae_doc2vec_description doc2vec description;'
screen -dm bash -c 'source activate cdea; python src/experiments/run_content_recommender_test.py 5 train_autoencoder_32_cdae_doc2vec_description doc2vec description;'
screen -dm bash -c 'source activate cdea; python src/experiments/run_content_recommender_test.py 10 train_autoencoder_32_cdae_doc2vec_description doc2vec description;'

screen -dm bash -c 'source activate cdea; python src/experiments/run_content_recommender_test.py 1 train_autoencoder_32_cdae_doc2vec_title doc2vec title;'
screen -dm bash -c 'source activate cdea; python src/experiments/run_content_recommender_test.py 5 train_autoencoder_32_cdae_doc2vec_title doc2vec title;'
screen -dm bash -c 'source activate cdea; python src/experiments/run_content_recommender_test.py 10 train_autoencoder_32_cdae_doc2vec_title doc2vec title;'

# CDEA autoencoder with Embedding 64
screen -dm bash -c 'source activate cdea; python src/experiments/run_content_recommender_test.py 1 train_autoencoder_64_cdae_doc2vec_description doc2vec description;'
screen -dm bash -c 'source activate cdea; python src/experiments/run_content_recommender_test.py 5 train_autoencoder_64_cdae_doc2vec_description doc2vec description;'
screen -dm bash -c 'source activate cdea; python src/experiments/run_content_recommender_test.py 10 train_autoencoder_64_cdae_doc2vec_description doc2vec description;'

screen -dm bash -c 'source activate cdea; python src/experiments/run_content_recommender_test.py 1 train_autoencoder_64_cdae_doc2vec_title doc2vec title;'
screen -dm bash -c 'source activate cdea; python src/experiments/run_content_recommender_test.py 5 train_autoencoder_64_cdae_doc2vec_title doc2vec title;'
screen -dm bash -c 'source activate cdea; python src/experiments/run_content_recommender_test.py 10 train_autoencoder_64_cdae_doc2vec_title doc2vec title;'

# CDEA autoencoder with Embedding 128
screen -dm bash -c 'source activate cdea; python src/experiments/run_content_recommender_test.py 1 train_autoencoder_128_cdae_doc2vec_description doc2vec description;'
screen -dm bash -c 'source activate cdea; python src/experiments/run_content_recommender_test.py 5 train_autoencoder_128_cdae_doc2vec_description doc2vec description;'
screen -dm bash -c 'source activate cdea; python src/experiments/run_content_recommender_test.py 10 train_autoencoder_128_cdae_doc2vec_description doc2vec description;'

screen -dm bash -c 'source activate cdea; python src/experiments/run_content_recommender_test.py 1 train_autoencoder_128_cdae_doc2vec_title doc2vec title;'
screen -dm bash -c 'source activate cdea; python src/experiments/run_content_recommender_test.py 5 train_autoencoder_128_cdae_doc2vec_title doc2vec title;'
screen -dm bash -c 'source activate cdea; python src/experiments/run_content_recommender_test.py 10 train_autoencoder_128_cdae_doc2vec_title doc2vec title;'