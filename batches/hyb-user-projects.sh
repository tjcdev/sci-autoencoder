#!/bin/bash

screen -dm bash -c 'source activate cdea; python src/experiments/run_hybrid_recommender_test.py 1;'
screen -dm bash -c 'source activate cdea; python src/experiments/run_hybrid_recommender_test.py 5;'
screen -dm bash -c 'source activate cdea; python src/experiments/run_hybrid_recommender_test.py 10;'