import os
from inspect import getsourcefile
from os.path import abspath
from pathlib import Path
import logging
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from feature_engine.imputation import AddMissingIndicator
from pyLDAvis import display
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler, OneHotEncoder
from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.pipeline import Pipeline
import re
import unidecode
from src.data.make_dataset import preprocessing
from src.features.build_features import feature_engineering


def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    # Lets read the data available in the raw folder
    #columns_to_skip = 'drug_id'
    columns_to_skip = ' '
    # This is a way to read the data from the csv file and skip the columns that are not needed.
    df_train_data = pd.read_csv('data/raw/drugs_train.csv',usecols=lambda x: x not in columns_to_skip)
    df_test_data = pd.read_csv('data/raw/drugs_test.csv',usecols=lambda x: x not in columns_to_skip)
    df_features = pd.read_csv('data/raw/drug_label_feature_eng.csv')
    
    #Lets Fix the FEATURES data quickly
    #df_features['description'] = df_features['description'].map(lambda x: re.sub('  ',' ',x))
    #df_features = df_features.drop_duplicates(keep='first').reset_index(drop=True)
    prep_features = preprocessing(df_features,df_features)
    df_features = prep_features.fix_duplicates()
    
    # TRAINING DATA preprocessing starts here
    # Lets fix the missing values in the training data
    prep_train_data = preprocessing(df_train_data,df_features)
    df_train = prep_train_data.fix_duplicates()
    df_train = prep_train_data.date_features()   
    df_train = prep_train_data.fix_missing_label_count() 

    # Add new features to the dataframe
    # Feature engineering training data

    # Feature engineering function is called with sollowing steps
    #step 1: initialize object
    feature_train_data = feature_engineering(df_train)

    #step 2: add new features, the Authorization period
    df_train = feature_train_data.authorization_period()

    #step 3: add new features, the cleaning of pharmaceutical companies using dirty category logic
    #dirty_learn, df_train = feature_train_data.dirty_categories()
    dirty_learn_company, na_val_comp, dirty_learn_desc, na_learn_desc, df_train = feature_train_data.dirty_categories_alt()
    
    #step 4: add feature encoding and final preparation for ML modeling
    #features_learned, transformer, df_train_feature = feature_train_data.feature_engineering()
    features_learned, transformer, df_train_feature = feature_train_data.feature_engineering_alt()

    #step 5: preparing data in a format suitable for ML modeling
    cols = transformer.get_feature_names_out()
    ml_train = pd.DataFrame(df_train_feature,columns=cols)    
    dropper = ['label_plaquette', 'label_ampoule', 'label_flacon', 'label_tube', 'label_stylo', 'label_seringue', 'label_pilulier', 'label_sachet', 'label_comprime', 'label_gelule', 'label_film', 'label_poche', 'label_capsule'] + ['marketing_declaration_date', 'marketing_authorization_date'] + ['administrative_status','approved_for_hospital_use', 'reimbursement_rate','marketing_authorization_process','pharmaceutical_companies'] # + ['authorization_processing_period']
    df_train = df_train.drop(dropper, axis=1)
    df_train = pd.concat([df_train ,ml_train], axis = 1)
    #Finally dataframe is ready for ML modeling

    Y_train = df_train['price'] # Split the Label or dependent variable from the dataframe
    df_train = df_train.drop(['price'], axis=1)

    #Merging the Y label with rest of the Dataframe
    df_train = pd.merge(df_train, pd.Series(Y_train), left_index=True, right_index=True)
    
    # TEST DATA preprocessing starts here
    # Lets fix the missing values in the test data
    prep_test_data = preprocessing(df_test_data,df_features)
    df_test = prep_test_data.fix_duplicates()
    df_test = prep_test_data.date_features()   
    df_test = prep_test_data.fix_missing_label_count() 
    # Feature engineering test data, applying learning of transformers to test data where needed
    #step 1: initialize object
    feature_test_data = feature_engineering(df_test)
    #step 2: add new features, the Authorization period
    df_test = feature_test_data.authorization_period()
    #step 3: Apply the learning for the cleaning of pharmaceutical companies using dirty category logic
    #df_test = dirty_learn.transform(df_test)
    dirty_learn_company, na_val_comp, dirty_learn_desc, na_learn_desc

    df_test['pharmaceutical_companies'] = df_test['pharmaceutical_companies'].map(dirty_learn_company)
    df_test['pharmaceutical_companies'] = df_test['pharmaceutical_companies'].fillna(na_val_comp)
    df_test['description'] = df_test['description'].map(dirty_learn_desc)
    df_test['description'] = df_test['description'].fillna(na_learn_desc)
    #step 4: Apply learning for feature encoding done on traning data and final preparation for ML modeling
    df_test_feature = features_learned.transform(df_test)
    #step 5: preparing data in a format suitable for ML modeling
    ml_test = pd.DataFrame(df_test_feature,columns=cols)    
    df_test = df_test.drop(dropper, axis=1)
    df_test = pd.concat([df_test ,ml_test], axis = 1)
    #Finally test dataframe is ready for ML modeling

    # Check the shape to verify our preprocessing is done correctly
    print(df_train.shape)
    print(df_test.shape)

    # Lets save the cleaned data to dataframes
    df_train.to_csv('data/processed/cleaned_train.csv', sep = ',',  index=False)
    df_test.to_csv('data/processed/cleaned_test.csv', sep = ',',  index=False)
    df_features.to_csv('data/processed/cleaned_features.csv', sep = ',',  index=False)

 
if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()

