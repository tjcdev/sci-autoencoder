#!/bin/bash

# CDEA autoencoder with Embedding 32
screen -dm bash -c 'source activate cdea; python src/models/run_content_recommender_test.py 1 autoencoder_32_cdae_tfidf_desc tfidf_desc;'
screen -dm bash -c 'source activate cdea; python src/models/run_content_recommender_test.py 5 autoencoder_32_cdae_tfidf_desc tfidf_desc;'
screen -dm bash -c 'source activate cdea; python src/models/run_content_recommender_test.py 10 autoencoder_32_cdae_tfidf_desc tfidf_desc;'