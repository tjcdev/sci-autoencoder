#!/bin/bash

screen -dm bash -c 'source activate cdea; python src/collectors/new-sci-participations.py;'
