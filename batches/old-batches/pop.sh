#!/bin/bash

# Train our New-SciStarter Collaborative Filtering Autoencoders
screen -dm bash -c 'source activate cdea; python src/experiments/run_pop_test.py 1;'
screen -dm bash -c 'source activate cdea; python src/experiments/run_pop_test.py 5;'
screen -dm bash -c 'source activate cdea; python src/experiments/run_pop_test.py 10;'
