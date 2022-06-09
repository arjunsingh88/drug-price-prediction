import os
from inspect import getsourcefile
from os.path import abspath
from pathlib import Path

def change_dir():
    """
    Change the working directory to the folder where the script is located
    """
    # Getting the path of the file where the function is located.
    file_dir =abspath(getsourcefile(lambda:0))
    # This is a way to get the path of the parent directory 0 means immediate and 1 is its super.
    goal_dir = Path(file_dir).parents[0]
    # change the current working directory to the parent directory
    os.chdir(goal_dir)

change_dir()
#print(os.getcwd())
# This command basically executes sequence of commands to run our entire project
# Starts with 
# 1. Updating the pip
# 2. Configuring python environment
# 3. Installing the required packages
# 4. Running the final.py script to clean and generate dataset for modeling
# 5. Running the train_model.py script to train, evaluate the model and save the predictions as submission.csv
os.system("python3 -m pip install --upgrade pip;python3 src/config/environment.py;pip install .;pip list;python3 src/final/final.py; python3 src/models/train_model.py")