import os
import sys
import venv

def isvirtualenv():
    return sys.prefix != sys.base_prefix

def findfile(startdir, pattern):
    for root, dirs, files in os.walk(startdir):
        for name in files:
            if name.find(pattern) >= 0:
                return root + os.sep + name

    return None

venv_path = 'price_predict_env'            # This is an environment path

if isvirtualenv():
    print('Already in virtual environment.')
else:
    if findfile(os.getcwd(), 'activate') is None:
        print('No virtual environment found. Creating one.')
        env = venv.EnvBuilder(with_pip = True)
        env.create(venv_path)
        print('Virtual environment created.')
        print('Installing dependencies.')
        
    else:
        print('Not in virtual environment. Virtual environment directory found.')

    # This is the heart of this script that puts you inside the virtual environment. 
    # There is no need to undo this. When this script ends, your original path will 
    # be restored.
    os.environ['PATH'] = os.path.dirname(findfile(os.getcwd(), 'activate')) + os.pathsep + os.environ['PATH']
    sys.path.insert(1, os.path.dirname(findfile(venv_path, 'easy_install.py')))

# Run your script inside the virtual environment from here
print(os.environ['PATH'])
#os.system("pip install -e .; pip list")