#!/bin/bash
echo "Starting The Price Prediction for pharmaceutical drugs"

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
#echo $SCRIPTPATH
cd "$SCRIPTPATH"


echo "Testing Python environment"
python3 test_environment.py
echo "setting up Virtual environment"
python3 src/config/environment.py
echo "Environment is all set up"
sleep 1

echo "Starting the Data preprocessing and Feature engineering for the data available"
python3 src/final.py

echo "training and deploying our price prediction model"
python3 src/models/train_model.py

sleep 3
echo "Summary of the Project after option 1"
echo "The end-to-end of the model finished 
      1. Raw data in raw folder, 
      2. Processed data in processed folder 
      3. Final Predictions in predictions folder"

sleep 2
echo "Summary of the Project after option 2"
echo "The end-to-end of the model finished 
      1. Raw data in raw folder, 
      2. Processed data in processed folder 
      3. Adjusted R2 score of the training data"

echo "exiting the project"