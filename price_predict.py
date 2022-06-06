import os
from inspect import getsourcefile
from os.path import abspath
from pathlib import Path

def change_dir():
    """
    Change the working directory to the folder where the script is located
    """
    file_dir =abspath(getsourcefile(lambda:0))
    # This is a way to get the path of the parent directory 0 means immediate and 1 is its super.
    goal_dir = Path(file_dir).parents[0]
    # change the current working directory to the parent directory
    os.chdir(goal_dir)

change_dir()
print(os.getcwd())

os.system("python3 src/config/environment.py; pip install -e .;pip list;python3 src/final.py; python3 src/models/train_model.py")