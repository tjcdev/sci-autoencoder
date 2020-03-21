# Recommendations for the SciStarter Dataset

In this repo we have build a recommender system that uses the content of SciStarter projects in order to recommend users new projects that are most similar to projects they have already done.


## Setup

To run the notebooks and src files you will first need to set up your conda environment using the ```environment.yml``` file

## Datasets

There are two main aspects to these datasets.

- Projects: The content of projects were retrieved from the SciStarter API as JSON objects and then converted to the ```data/raw/project_data``` dataframe
- Participation: The records of users interacting with a project (storeed at ```data/raw/participation_data```) 

## Data Exploration and Model Exploration

You can see some of our data exploration and model exploration in our notebooks folder.

The ```data-processing``` folder is also where we create our train, validation and test splits.

## src

This contains our source files which includes,
- The different autoencoder architectures
- Files for training out autoencoders
- Recommender architectures
- Files for training our recommenders    

## batches

- This contains all the bash files for training and running our autoencoders and recommenders

## Author 

- **Thomas Cartwright**: (MSc) Artificial Intelligence, University of Edinburgh
