from setuptools import find_packages, setup

with open("requirements.txt") as requirement_file:
    requirements = requirement_file.read().split()
    print(requirements)

setup(
    name="src",
    description='The project entails an indepth isight into indicators responsible for drug prices in pharmaceutical industry. On top of that we build a ML model to predict drug prices bases on available indicators.',
    version="0.1.0",
    author="Arjun Singh",
    author_email="arjun.s.0717@gmail.com",
    install_requires=requirements,
    packages=find_packages(exclude=["src/visualization"]) # package = any folder with an __init__.py file
)


#pip install -e .
#pip install .