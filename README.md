Drugs Price Prediction
==============================

The project entails an indepth isight into indicators responsible for drug prices in pharmaceutical industry. On top of that we build a ML model to predict drug prices bases on available indicators.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- cleaned_train.csv, cleaned_test.csv, cleaned_features.csv
    │   ├── raw            <- The original, immutable data dump. drug_train, drug_test, drug_label_feature_eng
    │   └── predictions    <- The final result in submission.csv
    │
    ├── notebooks          <- Jupyter notebooks. test_logic.ipynb (All the inital research & experiemntation is here)
    │
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   ├── final.py       <- Combines both make dataset and feature engineering scripts for the cleaned and prepared data for ML modeling
    │   │
    │   ├── config           <- Scripts to identify platform(linux, mac, window), create/verify virtual environment, activate environment
    │   │   └── environment.py
    │   │    
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │    
    ├── setup.py           <- Script to setup the entire list of library as a package
    │    
    ├── price_predict.py   <- Script to capture the directory of project and execute 4 scripts in tandem
    │                      1. environment.py
    │                      2. pip install -e (standard package install)
    │                      3. final.py
    │                      4. train_model.py
    │
-------

Filenames: `drugs_train.csv` and `drugs_test.csv`

|Field|Description|
|---|---|
| `drug_id` | Unique identifier for the drug. |
| `description` | Drug label. |
| `administrative_status` | Administrative status of the drug. |
| `approved_for_hospital_use` | Whether the drug is approved for hospital use (`oui`, `non` or `inconnu`). |
| `reimbursement_rate` | Reimbursement rate of the drug. |
| `marketing_declaration_date` | Marketing declaration date. |
| `marketing_authorization_date` | Marketing authorization date. |
| `marketing_authorization_process` | Marketing authorization process. |
| `pharmaceutical_companies` | Companies owning a license to sell the drug. Comma-separated when several companies sell the same drug. |
| `price` | Price of the drug (i.e. the output variable to predict). |

**Note:** the `price` column only exists for the train data set.

### Text Description Feature Engineering

Filename: `drug_label_feature_eng.csv`

This file is here to help you and provide some feature engineering on the drug labels.

| Field | Description |
| --- | --- |
| `description` | Drug label. |
| `label_XXXX` | Dummy coding using the words in the drug label (e.g. `label_ampoule` = `1` if the drug label contains the word `ampoule` - vial in French). |
| `count_XXXX` | Extract the quantity from the description (e.g. `count_ampoule` = `32` if the drug label  the sequence `32 ampoules`). |

**Note:** This data has duplicate records and some descriptions in `drugs_train.csv` or `drugs_test.csv` might not be present in this file.

1. Instructions on how to run your code.</br>
    - `OPTION 1`: Execute the price_predict.py python script. once excecuted, it will run the the following
      - python3 src/config/environment.py
      - pip install -e .
      - pip list
      - python3 src/final.py
      - python3 src/models/train_model.py
    - `OPTION 2`: Incase option 1 does not work(tested only on mac systems). Try pip install -r requirements.txt and try Option 1 again

   `Additional Info:`
    - final.py script carries two step 1. Data Preprocessing(make_dataset.py) step 2. Feature engineering(build_features.py)
    - The results of final.py i.e. cleaned_train, cleaned_test, cleaned_features files are generated in data/preprocessed/ dir
    - After few seconds of delay the model file train_model.py is executed

   once executed, it will read the processed data from data/preprocessed/ directory. Then it will seek user input to execute the cases:
    - `case 1`. this case involves training model(parameters are tuned in jupyter notebook) on entire training dataset and applied to test data, the result being mean adjusted R2 with +/- standard deviation and stored in data/predictions as submission.csv
    - `case 2`. This particular case is to split train data into train and test and then build and execute model for this data

2. The selection of model is purely instinct and intuition based backed by research and experience.
   For the problem of pricing prediction, i have used ensemble learning, that aggregates the results of all the tries we do on our data.<br>
   Ensemble learning is the process by which multiple models, such as classifiers or experts, are strategically generated and combined to solve a particular computational intelligence problem. Ensemble learning is primarily used to improve the (classification, prediction, function approximation, etc.) performance of a model, or reduce the likelihood of an unfortunate selection of a poor one. Other applications of ensemble learning include assigning a confidence to the decision made by the model, selecting optimal (or near optimal) features, data fusion, incremental learning, nonstationary learning and error-correcting.
  
    | Algorithm | Description |
    | --- | --- |
    | Random Forest | Random Forest is an ensembling method and one of the most popular and powerful algorithm in Machine Learning. The random forest is a model made up of many decision trees. Rather than just simply averaging the prediction of trees (which we could call a “forest”), this model uses two key concepts that gives it the name random:<br>1. Random sampling of training data points when building trees<br>2. Random subsets of features considered when splitting nodes.</p> </details> |
    | Gradient Boosting | Gradient boosting is a machine learning boosting type. It strongly relies on the prediction that the next model will reduce prediction errors when blended with previous ones. The main idea is to establish target outcomes for this upcoming model to minimize errors.So every case’s outcome depends on the number of changes brought upon by the prediction and its effects on the prediction error.<br>1. If the prediction has a small change and causes a significant error drop, then the case’s expected target outcome will have a high value. Forecasts provided by new models could reduce the errors as long as they are near their targets.<br>2. If there are no error changes caused by a small prediction change, then the case’s next outcome will be zero. You cannot minimize the error by changing the prediction.</p> </details>  |

3. Overall performance of your algorithm(s).
  The algorithm in research performed excellent, powered by detailed feature engineering
    | Algorithm | CV Results | Hypertune Experiments | Hypertuned results |
    | --- | --- | --- | --- |
    | **`Gradient Boosting`** | Adjusted R2_score = 0.81 (+/- 0.03) | <img width="1119" alt="Gradient Boosting HiPlot" src="https://user-images.githubusercontent.com/45566835/172601913-a70d5946-f3d7-4630-b885-61217af0aa1f.png"> | **0.83** |
    | **`Random Forest`** | Adjusted R2_score = 0.78 (+/- 0.03) | <img width="1124" alt="Random Forest HiPlot" src="https://user-images.githubusercontent.com/45566835/172601970-b036ff97-0561-4594-9879-f9e53482f9f1.png"> | **0.81** |

4. Next steps and potential improvements.
    1. In Data preprocessing and understanding we can introduce the outlier/anomaly detection algorithm to study and filter the records which can bring down our models' accuracy.
    2. Feature engineering: Ideating additional features based on the understanding of pharmaceutical industry. For example, we have market authorization and market declaration date and there could be a potential for new feature
    3. Feature selection: Use of more high-level feature selection method, wrapper methods are more exhaustive but computationally expensive and time consuming. A small research study has been carried out using VIF i.e Variance inflation factor to find the features with VIF less than 5. Higher the value, a VIF above 10 indicates high correlation and is cause for concern but for better modeling anything above 2.5 requires attention.
    4. Hyperparameter tuning: It can be even more exhaustive, for the said use case I explored grid search, but we have other options such as Randomized search as well as Bayesian search.
    5. Model Building: Designing more complex model to identify the user based on the keystroke value implemented
    6. Future research: Deep Learning techniques like Deep Neural Network can be a potential to explore and find the implementation for our problem as they learn from data and can help improve upon to minimize loss and find the best features for our model.
