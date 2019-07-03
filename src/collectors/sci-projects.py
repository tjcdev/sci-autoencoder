import numpy as np
from json import loads
import requests
import time
import datetime
import os
import sys

response_folder = '../results/'
try:
    os.mkdir(response_folder)
except:
    print("results folder already exists")

data_folder = '../Data/'

# SciStarter API Key
key = '29ec4b02b5426297ad7e1515f66f295f7367c861839d1c7416827b93d3cc3ba1bc9878ecb708d73a7e62adecad9e5054632deec8eba0a0542864a7f1adb0ad71'

# Load the Project Ids
projects = np.load(data_folder + str(sys.argv[1]) + '.npy')

projectdata_name = 'projectdata' + str(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S'))  + '.txt'

try:
    # Load the polled Project Ids
    polled_projects = np.load(response_folder + 'polled_projects.npy')
except:
    polled_projects = np.array([], dtype=int)

# Calculate the remaining projects
remaining_projects = projects
for project in polled_projects:
    remaining_projects = remaining_projects.remove(project)

# Start the projectdata file blank
open(response_folder + projectdata_name, 'w').close()
# Adding the start of the json
with open(response_folder + projectdata_name, 'a') as projectdata:
    projectdata.write('{"projects": [')

# Create new blank log file
log_name = response_folder + 'logs-' + str(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S')) + '.txt'
open(log_name, "w").close()

# Loop over all projects
for project_id in remaining_projects:
    try:
        # Get the project data from SciStarter API
        project_url = 'https://scistarter.com/api/project/{}?&key={}'.format(str(project_id), key)
        response = requests.get(project_url)
        json_txt = response.text

        # Format the text
        json_txt = json_txt + ', \n'

        # Adding string to the end of the file
        with open(response_folder + projectdata_name, 'a') as projectdata:
            projectdata.write(json_txt)
        
        # Add project id to the list of projects polled
        polled_projects = np.append(polled_projects, project_id)
        np.save(response_folder + 'polled_projects', polled_projects)

        # Log successful poll
        with open(log_name, "a") as logs:
            logs.write(str(project_id) + " successfully polled. \n")
    except Exception as e:
        # handle error
        with open(log_name, "a") as logs:
            logs.write(str(project_id) + " failed to poll. \n")
            logs.write(str(e) + '\n')
    time.sleep(20)

# Closing off the json
with open(response_folder + projectdata_name, 'a') as projectdata:
    projectdata.write(']}')

# Delete the polled projects
os.remove(response_folder + 'polled_projects.npy')

# Log the successful poll of all projects
with open(log_name, "a") as logs:
    logs.write("SUCCESSFULLY POLLED ALL PROJECTS")
print("SUCCESSFULLY POLLED")
