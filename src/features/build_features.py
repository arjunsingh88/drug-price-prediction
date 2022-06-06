"""
@Authors:
* Arjun Singh (arjun.s.0717@gmail.com)
"""
from email import encoders
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from feature_engine.imputation import AddMissingIndicator
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler, OneHotEncoder, KBinsDiscretizer, LabelEncoder
from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.pipeline import Pipeline
import re
import unidecode
from feature_engine.encoding import RareLabelEncoder
import sys, os, warnings

class feature_engineering:
    
    def __init__(self, df):
        self.df = df

    def authorization_period(self):
        """
        This function takes the marketing authorization date and the marketing declaration date and
        subtracts them to get the authorization processing period
        :return: The dataframe with the new column added.
        """
        #self.df['authorization_processing_period'] = (self.df['marketing_declaration_date'] - self.df['marketing_authorization_date'])
        return self.df

    def dirty_categories(self):
        """
        The function takes in a dataframe and returns a dataframe with the rare labels encoded for dirt category columns
        the column in this case is pharmaceutical_companies
        the transformer encodes the categories with low value count into Rare/low_val category
        and the rest remain the same
        :return: The learning object for test data and the dataframe
        """
        self.df = self.df[[col for col in self.df.columns if 'price' not in col]]
        categorical_transformer_dirty = RareLabelEncoder(tol=0.0005, n_categories=25, max_n_categories=None, replace_with='Low_count', variables='pharmaceutical_companies', ignore_format=False)
        learning = categorical_transformer_dirty.fit(self.df)
        self.df = learning.transform(self.df)
        return learning, self.df

    def dirty_categories_alt(self):
        """
        We're going to group the data by the pharmaceutical_companies and description columns, and then
        aggregate the price column by the mean. 
        This will give us encoded columns by mean price which we can then bin discretize for our model
        
        :return: A dictionary of the mean price for each company, the mean price for all companies, a
        dictionary of the mean price for each description, the median price for all descriptions, and the
        dataframe.
        """
        learning_companies = self.df.groupby(self.df['pharmaceutical_companies'])['price'].aggregate('mean').to_dict()
        null_company_val = self.df['price'].median()
        self.df['pharmaceutical_companies'] = self.df['pharmaceutical_companies'].map(learning_companies)

        learning_description = self.df.groupby(self.df['description'])['price'].aggregate('mean').to_dict()
        null_desc_val = self.df['price'].median()
        self.df['description'] = self.df['description'].map(learning_description)
        return learning_companies, null_company_val, learning_description, null_desc_val, self.df


    #def consolidated_companies(self):
    #    filter = (self.df['pharmaceutical_companies'].value_counts()/self.df['pharmaceutical_companies'].count()).head(15).index
    #    return np.where(self.df['pharmaceutical_companies'].isin(filter) , self.df['pharmaceutical_companies'], 'Low_value').tolist()

        
    def feature_engineering(self):
        """
        The function takes in a dataframe and returns a list of three objects: a learning object, a
        transformer object and a matrix of the transformed data
        :return: The learning object is the transformer object, which is the ColumnTransformer object.
        The transformed data and the transformer as well.
        """
        #This particular try except block is to handle the error that arises when the dataframe doesnot contain the predicted label price
        try: 
            self.df = self.df[[col for col in self.df.columns if 'price' not in col]]
        except:
            pass

        ordinal_encoder_features = ['label_plaquette', 'label_ampoule', 'label_flacon', 'label_tube', 'label_stylo', 'label_seringue', 'label_pilulier', 'label_sachet', 'label_comprime', 'label_gelule', 'label_film', 'label_poche', 'label_capsule']
        OE_transformer = OrdinalEncoder(categories = 'auto' )

        binning_features = ['marketing_declaration_date', 'marketing_authorization_date'] #+ ['authorization_processing_period']
        KBinsDiscretizer_transformer = KBinsDiscretizer(n_bins=15, encode='onehot', strategy='kmeans', random_state=42)

        categorical_features = ['administrative_status','approved_for_hospital_use', 'reimbursement_rate','marketing_authorization_process','pharmaceutical_companies']
        OHE_transformer = OneHotEncoder(sparse=False,  drop = 'first', handle_unknown='error')

        transformer = ColumnTransformer(transformers = [("OE", OE_transformer, ordinal_encoder_features),
                                                    ("KBin", KBinsDiscretizer_transformer, binning_features),
                                                    ("OHE", OHE_transformer, categorical_features)])
        
        learning = transformer.fit(self.df)
        ml = transformer.transform(self.df)
        #print(feature.get_feature_names())

        return learning, transformer, ml

    def feature_engineering_alt(self):
        """
        The function takes in a dataframe and returns a list of three objects: a learning object, a
        transformer object and a matrix of the transformed data
        :return: The learning object is the transformer object, which is the ColumnTransformer object.
        The transformed data and the transformer as well.
        """
        #This particular try except block is to handle the error that arises when the dataframe doesnot contain the predicted label price
        #try: 
        #    self.df =  self.df[[col for col in  self.df.columns if 'price' not in col]]
        #except:
        #    pass
        ordinal_encoder_features = ['label_plaquette', 'label_ampoule', 'label_flacon', 'label_tube', 'label_stylo', 'label_seringue', 'label_pilulier', 'label_sachet', 'label_comprime', 'label_gelule', 'label_film', 'label_poche', 'label_capsule']
        OE_transformer = OrdinalEncoder(categories = 'auto' )

        binning_features = ['marketing_declaration_date']  #+ ['authorization_processing_period']
        KBinsDiscretizer_transformer = KBinsDiscretizer(n_bins=15, encode='onehot', strategy='kmeans', random_state=42)

        binning_features_1 = ['marketing_authorization_date'] +['pharmaceutical_companies'] + ['description'] #+ ['authorization_processing_period']
        KBinsDiscretizer_transformer_1 = KBinsDiscretizer(n_bins=15, encode='onehot', strategy='kmeans', random_state=42)

        categorical_features = ['administrative_status','approved_for_hospital_use', 'reimbursement_rate','marketing_authorization_process']
        OHE_transformer = OneHotEncoder(sparse=False,   handle_unknown='error')

        transformer = ColumnTransformer(transformers = [("OE", OE_transformer, ordinal_encoder_features),
                                                    ("KBin", KBinsDiscretizer_transformer, binning_features),
                                                    ("KBin1", KBinsDiscretizer_transformer_1, binning_features_1),
                                                    ("OHE", OHE_transformer, categorical_features)])
        
        learning = transformer.fit(self.df)
        ml = transformer.transform(self.df)
        #print(feature.get_feature_names())

        return learning, transformer, ml
          