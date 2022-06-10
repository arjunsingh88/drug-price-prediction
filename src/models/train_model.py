from inspect import getsourcefile
from os.path import abspath
from pathlib import Path
import os
#from ensurepip import bootstrap
from pathlib import Path
import sys
from inspect import getsourcefile
from os.path import abspath
from pathlib import Path
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import mean_squared_error as mse, r2_score as r2
import warnings
import pandas as pd
import numpy as np


class Model:
    
    def __init__(self, df, df_test, model_type = None):
        """
        The function takes in a dataframe, a target column, and a boolean value. If the boolean value is
        False, the function will split the dataframe into a training set and a testing set. If the
        boolean value is True, the function will split the dataframe into a training set, a testing set,
        and a validation set
        
        :param df: This is the training data
        :param df_test: This is the dataframe that you want to predict on
        :param model_type: This is the type of model that you want to use. You can choose between a
        random forest model and a gradient boosting model
        """
        self.df = df
        self.df_test = df_test
        
        # This is the code that is used to create the model. 
        # The user can choose between a random forest model and a gradient boosting model.
        
        if model_type == 'rf':
            self.user_model = RandomForestRegressor(bootstrap= True, max_features= None, n_estimators= 70, max_depth= 120, min_samples_leaf = 4, min_samples_split = 10)
            
        if model_type == 'gb':
            self.user_model = GradientBoostingRegressor(random_state=111, learning_rate = 1, max_leaf_nodes = 2, n_estimators = 50)
        else:
            pass
            
    def split(self, test_size = None, target = None, testing = None):
        """
        The function takes in a dataframe, a target column, and a boolean value. If the boolean value is
        False, the function will split the dataframe into a training set and a testing set. If the boolean
        value is True, the function will split the dataframe into a training set, a testing set, and a
        validation set
        
        :param test_size: This is the size of the test set. It should be a float between 0 and 1
        :param target: The column name of the target variable
        :param testing: If you want to test the model, set this to True. If you want to use the model to
        predict on the test set, set this to False
        """
        if testing == False:
            self.X_train, self.X_test, self.y_train = self.df.drop(target, axis = 1), self.df_test, self.df[target]
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(  self.df.drop(columns=[target], axis = 1),   self.df[target],
                                                            test_size = 0.20, random_state=1)  

    def fit(self):
        """
        > The function takes in a model, a training set, and a test set, and returns a trained model
        """
        self.model =  self.user_model.fit(self.X_train, self.y_train)

    def predict(self, input_value):
        """
        The function takes in a string, and if the string is 'model_testing', it will return the predictions
        of the model on the test data. Otherwise, it will return the predictions of the model on the input
        value
        
        :param input_value: This is the input value that you want to predict if model testing then it predicts values for test dataset
                            else it predicts value for a single instance
        :return: The result of the prediction.
        """
        if input_value == 'model_testing':
            result = self.model.predict(self.X_test)
        else:
            result = self.model.predict(np.array(input_value))
        return result

    def adj_r2(estimator, X, y_true):
        n, p = X.shape
        pred = estimator.predict(X)
        return 1 - ((1 - r2(y_true, pred)) * (n - 1))/(n-p-1)

    def score(self, result, testing = None):
        """
        It takes in a model, trains it on the training data, and then returns the score of the model on the
        test data
        we have option to try 3 evaluation metric depending on the need
        
        :param result: the predicted values
        :return: The score of the model.
        """
        if testing is False:
            scores = cross_val_score(self.user_model, self.X_train, self.y_train, scoring=Model.adj_r2 ,cv=5, n_jobs=-1 ,verbose=10)
            return scores
        else:
            #r2(self.y_test,result)
            return  1 - ( 1- self.model.score(self.X_test, self.y_test) ) * ( len(self.y_test) - 1 ) / ( len(self.y_test) - self.X_test.shape[1] - 1 )        

if __name__ == '__main__':

    df_train = pd.read_csv("data/processed/cleaned_train.csv", sep = ',' , header = 0)
    df_test = pd.read_csv("data/processed/cleaned_test.csv", sep = ',' , header = 0)

    columns_to_drop = ['drug_id','description']
    train = df_train.drop(columns = columns_to_drop, axis = 1)
    test = df_test.drop(columns = columns_to_drop, axis = 1)
    
    # Creating an instance of the class Model.
    model_instance = Model(train, test, model_type = 'gb')
    obtain_choice = input("Press '1' to Train and Validate model(Predict values for test data and store results).\nPress '2' to Train & Test the model\n")
    try:
        if obtain_choice == "1"  :
            # choice 1 is to train and validate the model on entire traning data
            test = False
        if obtain_choice == "2"  :
            # choice 2 is to split taining data into train and validate data and then train test the model
            test = True

        # Splitting the data into training and testing set.
        model_instance.split(testing =test, target = 'price')
        # Fitting the model on the training data.
        model_instance.fit()
        # Predicting the values for the test dataset
        result = model_instance.predict(input_value = 'model_testing')

        # Calculating the score of the model on the test data
        #print our results
        
        try: 
            if test is True:
                print('The Adjusted R2 score of the model on the test data is: {:.2f} '.format(model_instance.score(result, testing = test)))
            else:
                scores = model_instance.score(result, testing = test)
                print("The Adjusted R2 score of the model on the training data is: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)) 
                final_results = pd.Series(np.round_(result, decimals = 2))
                df_submission = pd.concat([df_test['drug_id'],final_results.rename('price') ], axis = 1)
                print("Writing csv data of predicted values to the file")
                df_submission.to_csv('data/predictions/submission.csv', sep = ',', index = False)
        except:
            print('Check the code')
        
    except:
        print('Try again, wrong option')


