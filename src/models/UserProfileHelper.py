import pandas as pd
from scipy import sparse

class UserProfileHelper:
    def __init__(self):
        # initialise the class
        self.projects = None
    
    def importProjects(self, projectsFile, ignored_project_id):
        self.projects = pd.read_pickle(projectsFile)
        self.projects = self.projects[self.projects['project_id'] != ignored_project_id]


    def generateProfile(self, userProjects):
        # Get the ids of projects that the user has interacted with
        project_ids = list(userProjects[userProjects == 1].index)

        project_titles = []

        for project in project_ids:       
            project_titles.extend(self.projects[self.projects['project_id'] == int(project)]['title'].tolist())

        # Make a 'user profile' out of these projects
        projects_fields_combined = ' '.join(project_titles)

        # Set this part of the profile to the user profile
        return projects_fields_combined