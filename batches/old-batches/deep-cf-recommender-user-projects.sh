# Deep autoencoder with Embedding 32 on Users Projects
screen -dm bash -c 'source activate cdea; python src/experiments/run_cf_recommender_test.py 1 train_autoencoder_32_deep1_users_projects users_projects;'
screen -dm bash -c 'source activate cdea; python src/experiments/run_cf_recommender_test.py 5 train_autoencoder_32_deep1_users_projects users_projects;'
screen -dm bash -c 'source activate cdea; python src/experiments/run_cf_recommender_test.py 10 train_autoencoder_32_deep1_users_projects users_projects;'

screen -dm bash -c 'source activate cdea; python src/experiments/run_cf_recommender_test.py 1 train_autoencoder_64_deep1_users_projects users_projects;'
screen -dm bash -c 'source activate cdea; python src/experiments/run_cf_recommender_test.py 5 train_autoencoder_64_deep1_users_projects users_projects;'
screen -dm bash -c 'source activate cdea; python src/experiments/run_cf_recommender_test.py 10 train_autoencoder_64_deep1_users_projects users_projects;'

screen -dm bash -c 'source activate cdea; python src/experiments/run_cf_recommender_test.py 1 train_autoencoder_128_deep1_users_projects users_projects;'
screen -dm bash -c 'source activate cdea; python src/experiments/run_cf_recommender_test.py 5 train_autoencoder_128_deep1_users_projects users_projects;'
screen -dm bash -c 'source activate cdea; python src/experiments/run_cf_recommender_test.py 10 train_autoencoder_128_deep1_users_projects users_projects;'

