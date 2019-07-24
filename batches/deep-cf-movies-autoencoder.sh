# Train our Movies Collaborative Filtering Autoencoders
screen -dm bash -c 'source activate cdea; python src/models/autoencoders/train_CF_autoencoder.py 1024 100 32 deep1 movies;'
screen -dm bash -c 'source activate cdea; python src/models/autoencoders/train_CF_autoencoder.py 1024 100 64 deep1 movies;'
screen -dm bash -c 'source activate cdea; python src/models/autoencoders/train_CF_autoencoder.py 1024 100 128 deep1 movies;'