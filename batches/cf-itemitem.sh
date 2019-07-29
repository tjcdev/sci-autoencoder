#!/bin/bash

# Train our New-SciStarter Collaborative Filtering Autoencoders
screen -dm bash -c 'source activate cdea; python src/experiments/run_cf_useruser_test.py 1;'
screen -dm bash -c 'source activate cdea; python src/experiments/run_cf_useruser_test.py 5;'
screen -dm bash -c 'source activate cdea; python src/experiments/run_cf_useruser_test.py 10;'
