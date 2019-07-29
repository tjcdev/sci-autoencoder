import numpy as np
from json import loads
import requests
import time
import datetime
import os
import sys

response_folder = 'data/raw/'
try:
    os.mkdir(response_folder)
except:
    print("results folder already exists")

data_folder = 'data/raw/'

projectdata_name = 'new-sci-participation-data.txt'

# Start the projectdata file blank
open(response_folder + projectdata_name, 'w').close()

# Adding the start of the json
with open(response_folder + projectdata_name, 'a') as projectdata:
    projectdata.write('{"projects": [')

# Create new blank log file
log_name = response_folder + 'new-sci-participations-logs.txt'
open(log_name, "w").close()

# Loop over all projects
for i in range(0, 500):
    try:
        # Get the project data from SciStarter API
        participations_url = 'https://scistarter.org/api/stream-page?page=%d&key=5255cf33e739e9ecc20b9b260cb68567fbc81f6b1bfb4808ba2c39548501f0a1523e2e97d79563645cba40a09894bfdb277779d1145a596f237ebdc166afcf50' % (i)
        response = requests.get(participations_url)
        json_txt = response.text

        # Format the text
        json_txt = json_txt.replace('\n', ', ')

        # Adding string to the end of the file
        with open(response_folder + projectdata_name, 'a') as projectdata:
            projectdata.write(json_txt)
        
        # Log successful poll
        with open(log_name, "a") as logs:
            logs.write(str(i) + " successfully polled. \n")

    except Exception as e:
        # handle error
        with open(log_name, "a") as logs:
            logs.write(str(i) + " failed to poll. \n")
            logs.write(str(e) + '\n')
    time.sleep(20)

# Closing off the json
with open(response_folder + projectdata_name, 'a') as projectdata:
    projectdata.write(']}')

# Log the successful poll of all projects
with open(log_name, "a") as logs:
    logs.write("SUCCESSFULLY POLLED ALL PAGES")
print("SUCCESSFULLY POLLED")
