# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from feature_engine.imputation import AddMissingIndicator
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler, OneHotEncoder
from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.pipeline import Pipeline
import re
import unidecode
import sys, os, warnings
#clear raw data into usable data for ML and feature engineering

class preprocessing:

    def __init__(self, df, df_features):
        self.df = df
        self.df_features = df_features

    def fix_duplicates(self):
        """
        It removes duplicate rows from the dataframe, keeping the last row of the duplicates
        :return: The dataframe with the duplicates removed.
        """
        # a simple cleaning of extra space in the textual columns
        self.df['description'] = self.df['description'].map(lambda x: re.sub('  ',' ',x))
        self.df['description'] = self.df['description'].map(lambda x: x[1:-1] if x.startswith('"') and x.endswith('"') else x)
        #self.df['pharmaceutical_companies'] = self.df['pharmaceutical_companies'].str.strip(' ')
        if 'pharmaceutical_companies' in self.df.columns:
            self.df['pharmaceutical_companies'] = self.df['pharmaceutical_companies'].map(lambda x: x[1:-1] if x.startswith('"') or x.endswith('"')else x)
            self.df['pharmaceutical_companies'] = self.df['pharmaceutical_companies'].map(lambda x: x[1:-1] if x.startswith(' ') or x.endswith(' ')else x)
        self.df = self.df.drop_duplicates(keep='last',subset=self.df.columns.difference(['drug_id'])).reset_index(drop=True)
        #if 'price' in self.df.columns:
        #    self.df = self.df[~self.df.duplicated(keep='last',subset=self.df.columns.difference(['price']))==True].sort_values(by=['description']).reset_index(drop=True)
        #else:
        #    self.df =self.df[~self.df.duplicated(keep='last')==True].sort_values(by=['description'])
        return self.df  
        
    # Converting date columns to datetime and to year
    def date_features(self):
        """
        We are converting the dates to datetime format and then extracting the year from the datetime object
        :return: The dataframe with the new columns added.
        """
        # We could convert the date to datetime but we will use year as month and date are both 01, 01 e.g. 2014-01-01
        self.df['marketing_declaration_date'] = pd.to_datetime(self.df['marketing_declaration_date'], format='%Y%m%d').dt.year
        self.df['marketing_authorization_date'] = pd.to_datetime(self.df['marketing_authorization_date'], format='%Y%m%d').dt.year
        return self.df

    def fix_missing_label_count_values_file(self):
        """
        It takes the dataframe of the labels and the dataframe of the features and merges them together
        :return: The dataframe with the missing values filled in.
        """
        # fixing features from CSV file, in our case its default functionality       
        self.df = self.df.merge(self.df_features, how='left', on='description')
        #print(self.df.columns)
        return self.df
         

    def fix_missing_label_count_values_logic(self):
        """
        The function takes in a dataframe and returns a dataframe with missing values in the label_ and
        count_ features filled in using the manual search logic
        :return: The dataframe with the missing values filled in.
        """
        # fixing features using logic, secondary case when we still have missing values after merging with features
        test = self.df[[col for col in self.df.columns if 'label_' in col or 'count_' in col]]
        label_items = [re.sub('label_', '',l)for l in test if 'label_' in l]
        count_items = [re.sub('count_', '',l)for l in test if 'count_' in l]
        new_value_dictionary = dict()
        # Fixing missing values in label_ features in the data
        for idx, text in self.df.loc[self.df['label_plaquette'].isna(),'description'].iteritems():
            for item in label_items:
                if item in text:
                    #li1.append({'label_'+item:1})
                    new_value_dictionary['label_'+item] = 1
                else:
                    #li1.append({'label_'+item:0})
                    new_value_dictionary['label_'+item] = 0 
        
        # Fixing missing values in counts_ features in the data
            text = unidecode.unidecode(text)
            list_of_words = text.split()
            sentence = [re.sub(r'\([s)]*\)','', word) for word in list_of_words]
            for word in count_items:
                if word in sentence:
                    ind = sentence.index(word)
                    if ind == 0:
                        #li.append({'count_'+word:0})
                        new_value_dictionary['count_'+word] = 0
                    else:
                        try:
                            #if sentence[ind-1].isdigit() == True:
                            #li.append({'count_'+word:int(sentence[ind-1])})
                            new_value_dictionary['count_'+word] = int(sentence[ind-1])
                        except:
                            #li.append({'count_'+word:0})
                            new_value_dictionary['count_'+word] = 0
                else:
                    #li.append({'count_'+word:0})
                    new_value_dictionary['count_'+word] = 0
            
            for key in new_value_dictionary.keys():
                try:
                    self.df.loc[idx,key] = new_value_dictionary.get(key)
                except:
                    pass
        return self.df

    def fix_missing_label_count(self):
        """
        It takes the dataframe, checks if there are any missing values in the label_plaquette column, and if
        there are, it checks if the description column has any unique values. If it does, it runs the
        fix_missing_label_count_values_file function, which fixes the missing values by using the values
        from the CSV file. If there are no unique values in the description column, it runs the
        fix_missing_label_count_values_logic function, which fixes the missing values by using the search logic
        described in fix_missing_label_count_values_logic function.
        :return: The dataframe with the missing values fixed.
        """
        # Fix features from CSV file
        self.df = preprocessing.fix_missing_label_count_values_file(self)
        # check if we still have missing value for those features added from file and fix them with logic
        if self.df.loc[self.df['label_plaquette'].isna(),'description'].nunique() != 0:
            self.df = preprocessing.fix_missing_label_count_values_logic(self)
        return self.df