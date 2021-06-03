#!/usr/bin/env python
# coding: utf-8

# # Enron Person of Interest Classification



# Read and clean the data, add new features, split into testing/training subsets

# Import libraries
import pickle
from tester import dump_classifier_and_data
from dos_to_unix import d2ux
import pandas as pd
import numpy as np
from math import inf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, make_scorer, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')
pd.options.display.max_columns = 50

# Remove obvious outliers (These were discovered by reviewing enron61702insiderpay.pdf)
def remove_outliers(data):
    data.pop('TOTAL', 0 ) # Remove TOTAL 
    data.pop('THE TRAVEL AGENCY IN THE PARK', 0) # Remove THE TRAVEL AGENCY IN THE PARK due to lack of relevance
    print('Removed the following outliers:')
    print('TOTAL')
    print('THE TRAVEL AGENCY IN THE PARK')
    
# Read data to dataframe
def to_dataframe(data):
    """
    Create pandas dataframe from dictionary of enron data
    """
    dataframe = pd.DataFrame(data).T
    dataframe.index = dataframe.index.rename('name')
    return dataframe

def new_columns(dataframe):
    """
    Create the following new columns:
    
    total_restricted_stock_diff: total stock value - restricted stock
    to_poi_ratio - ratio of emails to poi
    from_poi_ratio - ratio of emails from poi
    payments_to_stock_ratio - ratio of total payments to total stock value
    """
    dataframe['total_restricted_stock_diff'] = dataframe.total_stock_value - dataframe.restricted_stock 
    dataframe['to_poi_ratio'] = dataframe.from_this_person_to_poi/ dataframe.to_messages
    dataframe['from_poi_ratio'] = dataframe.from_poi_to_this_person / dataframe.from_messages
    dataframe['payments_to_stock_ratio'] = dataframe.total_payments / dataframe.total_stock_value
    # fill inf or NaN ratios in new columns with 0.0
    dataframe.loc[dataframe.to_poi_ratio.isna(), 'to_poi_ratio'] = 0.0
    dataframe.loc[dataframe.from_poi_ratio.isna(), 'from_poi_ratio'] = 0.0
    dataframe.loc[(dataframe.payments_to_stock_ratio == inf) | 
                  (dataframe.payments_to_stock_ratio.isna()), 'payments_to_stock_ratio'] = 0.0 
    return dataframe

def clean_data(dataframe):
    """
    Add new columns, convert "poi" from boolean to binary, convert pandas objects to floats (fill NaN with 0.0), 
    split X and y data
    """
    nan_dict = {} # Create dictionary to store total NaN values by column label
    for col in dataframe.columns:
        # Adjust string "NaN" values to the numpy representation, count total nan values for each label and save to nan_dict
        dataframe.loc[dataframe[col] == 'NaN', col] = np.nan
        nan_dict.update({col : dataframe[col].isnull().sum()})
        if col == 'email_address':
            pass
        elif col == 'poi':
            dataframe[col] = dataframe[col].astype(int) # Convert poi to binary
        else:
            dataframe[col] = dataframe[col].astype(float) # All other numeric values as floats
            dataframe[col] = dataframe[col].fillna(0.0) # Data imputation (fill missing data)   
    # Sort nan dict, and print
    nan_dict = dict(sorted(nan_dict.items(), key = lambda item: item[1]))
    print('\nTOTAL NAN VALUES BY LABEL:\n')
    for k in nan_dict.keys():
        print(k, ':', nan_dict[k])
    dataframe = new_columns(dataframe) # Add new columns
    # Create separate variables for X and y
    print('\nDataset after cleaning data and adding new features:\n')
    dataframe.info()
    X_features = [feature for feature in dataframe.columns if feature not in ['poi', 'email_address']]
    X_data = dataframe[X_features]
    y_data = dataframe.poi
    return X_data, y_data

# Convert dos linefeed to unix, return new .pkl file name ("_unix" appended)
dataset_file = d2ux("final_project_dataset.pkl") 

# Load the dictionary containing the dataset
with open(dataset_file, "rb") as data_file:
    data_dict = pickle.load(data_file)

# Remove outliers
remove_outliers(data_dict)

# Read data to dataframe for easier data manipulation, and clean data
df = to_dataframe(data_dict)
# Print the total number of people in the dataset after removing outliers
print(f'\nThe total number of people in the dataset after removing outliers: {len(df)}')
# Print the total number of POIs, and non-POIs
print(f'Total Persons of Interest: {df.poi.sum()}\nTotal who are not Persons of Interest: {len(df) - df.poi.sum()}')
X, y = clean_data(df)

print('\nSample of data after data cleaning, and prior to final feature selection and standardization:')
pd.concat([y, X], axis = 1).head(5)




# Test a variety of classifiers to achieve better than 0.3 precision and recall

# Choose if data is going to be scaled, and if/how many features will be selected
def scale_selectk(X_data, y_data, scale = True, selectk = True, k = 'all'):
    """
    Specifies whether to scale the data, and which best columns to keep
    """
    # Scale data (keep as dataframe) if scale = True
    if scale == True:
        scaler = MinMaxScaler()
        X_data = pd.DataFrame(data = scaler.fit_transform(X_data),
                              columns = X_data.columns,
                              index = X_data.index)
    # Keep selected X features if selectk = True
    if selectk == True:
        best = SelectKBest(f_classif, k = k).fit(X_data, y_data)
        X_mask = best.get_support()
        X_kept_cols = X_data.columns[X_mask].tolist()
        print('The kept features are:', X_kept_cols)
        print('The total number of kept features are:', len(X_kept_cols))
        print('The feature scores are:')
        for c, b in zip(X_data.columns, best.scores_):
            print(c, ':', b)
        X_data = X_data[X_kept_cols]
    return X_data

# Print different performance metrics for a classifier
def metrics(predictions, y_val, estimator_descr = 'estimator'):
    """
    Prints the accuracy, f1_score, recall, and precision for a given prediction
    """
    print(f'\nAccuracy Score for {estimator_descr}: {accuracy_score(predictions, y_val) * 100}%')
    print(f'F1 Score for {estimator_descr}: {f1_score(predictions, y_val) * 100}%')
    print(f'Precision Score for {estimator_descr}: {precision_score(predictions, y_val) * 100}%')
    print(f'Recall Score for {estimator_descr}: {recall_score(predictions, y_val) * 100}%\n')
    
# Define performance reporting function to print the best performing parameters when using GridSearchCV
def best_performance(classifier):
    """
    Reports performance and parameters of the best classifier of a parameter search/CV
    """
    print('Best Parameters: ' + str(classifier.best_params_) + '\n')
    print('Best Score: ' + str(classifier.best_score_))

# Optimize estimator using cross-validation
def EstimatorCV(X_data, y_data, pipeline, params): 
    # define parameters to test using grid search
    estimator = GridSearchCV(pipeline, 
                             param_grid = params, 
                             cv = 5, 
                             scoring = make_scorer(f1_score, pos_label = 1),
                             verbose = True, 
                             n_jobs = -1)
    # find best training parameters
    best_estimator = estimator.fit(X_data, y_data)
    best_performance(best_estimator)
    return best_estimator




# Naive Bayes Classifier

# Scale using Standard Scaler
nb_scaler = StandardScaler()
nb_X = nb_scaler.fit_transform(X)

# Split into training/testing sets (Best performance when there is no scaling/removing features)
X_train, X_test, y_train, y_test = train_test_split(nb_X, y, test_size = 0.2, random_state = 0)

pca = PCA()
nbc = GaussianNB()

nb_pipeline = Pipeline([('pca', pca),
                         ('nbc', nbc)])

# Naive Bayes scores with no parameter optimization, no PCA:
nbc = nbc.fit(X_train, y_train)
nbc_pred = nbc.predict(X_test)
metrics(nbc_pred, y_test, 'Naive Bayes')

# Naive Bayes scores with no parameter optimization, but with PCA
nbc_w_pca = nb_pipeline.fit(X_train, y_train)
nbc_w_pca_pred = nbc_w_pca.predict(X_test)
metrics(nbc_w_pca_pred, y_test, 'Naive Bayes with PCA')

# Naive Bayes cross-validation
nb_param_grid = {'pca__n_components' : [None, 2, 5, 10],
                 'pca__whiten' : [True, False]}

nb_clf = EstimatorCV(X_train, y_train, nb_pipeline, nb_param_grid)
nb_clf_pred = nb_clf.predict(X_test)
metrics(nb_clf_pred, y_test, 'Naive Bayes with optimized PCA parameters')

print('\nConfusion Matrix:')
print(confusion_matrix(y_test, nb_clf_pred),'\n')




# Random Forest Classifier

# Scale = True, select 5 features
rf_X = scale_selectk(X, y, scale = True, selectk = True, k = 5)
# Split into training/testing sets
X_train, X_test, y_train, y_test = train_test_split(rf_X, y, test_size = 0.2, random_state = 0)

pca = PCA()
rfc = RandomForestClassifier(criterion = 'entropy',
                             random_state = 0,
                             max_features = 'auto',
                             warm_start = True)

rf_pipeline = Pipeline([('pca', pca), 
                        ('rfc', rfc)])

# Random Forest scores with no parameter optimization, no PCA
rfc = rfc.fit(X_train, y_train)
rfc_pred = rfc.predict(X_test)
metrics(rfc_pred, y_test, 'Random Forest')

# Random Forest scores with no parameter optimization, but with PCA
rfc_w_pca = rf_pipeline.fit(X_train, y_train)
rfc_w_pca_pred = rfc_w_pca.predict(X_test)
metrics(rfc_w_pca_pred, y_test, 'Random Forest with PCA')

# Random forest cross-validation
rf_param_grid = {'pca__n_components' : [None, 2, 5],
                 'pca__whiten' : [True, False],
                 'rfc__min_samples_leaf': [1, 2, 4, 6],
                 'rfc__class_weight' : [None, 'balanced']}

rf_clf = EstimatorCV(X_train, y_train, rf_pipeline, rf_param_grid)
rf_clf_pred = rf_clf.predict(X_test)
metrics(rf_clf_pred, y_test, 'Random Forest with PCA and optimized parameters')

print('\nConfusion Matrix:')
print(confusion_matrix(y_test, rf_clf_pred),'\n')




# SVM Classifier


# Scale = True, select all features
svm_X = scale_selectk(X, y, scale = True, selectk = True, k = 'all')
# Split into training/testing sets
X_train, X_test, y_train, y_test = train_test_split(svm_X, y, test_size = 0.2, random_state = 0)

pca = PCA(whiten = True)
svc = SVC(random_state = 0,
          probability = True,
          decision_function_shape = 'ovo')

svc_pipeline = Pipeline([('pca', pca), 
                         ('svc', svc)])

# SVM scores with no parameter optimization, no PCA
svc = svc.fit(X_train, y_train)
svc_pred = svc.predict(X_test)
metrics(svc_pred, y_test, 'SVM')

# SVM scores with no parameter optimization, but with PCA
svc_w_pca = svc_pipeline.fit(X_train, y_train)
svc_w_pca_pred = svc_w_pca.predict(X_test)
metrics(svc_w_pca_pred, y_test, 'SVM with PCA')

# SVM cross-validation
svm_param_grid = {'pca__n_components' : [None, 2, 5],
                  'svc__C' : [0.1, 10, 50, 100],
                  'svc__kernel' : ['linear', 'rbf', 'sigmoid']}

svm_clf = EstimatorCV(X_train, y_train, svc_pipeline, svm_param_grid)
svm_clf_pred = svm_clf.predict(X_test)
metrics(svm_clf_pred, y_test, 'Random Forest with PCA and optimized parameters')

print('\nConfusion Matrix:')
print(confusion_matrix(y_test, svm_clf_pred),'\n')




# Gradient Boost Classifier


# Scale = True, select 4 features
gb_X = scale_selectk(X, y, scale = True, selectk = True, k = 4)

# Split into training/testing sets
X_train, X_test, y_train, y_test = train_test_split(gb_X, y, test_size = 0.2, random_state = 0)

pca = PCA(n_components = None)
gbc = GradientBoostingClassifier(loss = 'exponential',
                                 random_state = 0, 
                                 max_features = 'auto',
                                 warm_start = True)

gbc_pipeline = Pipeline([('pca', pca), 
                         ('gbc', gbc)])

# Gradient Boost Classifier scores with no parameter optimization, no PCA
gbc = gbc.fit(X_train, y_train)
gbc_pred = gbc.predict(X_test)
metrics(gbc_pred, y_test, 'Gradient Booster')

# Gradient Boost Classifier scores with no parameter optimization, but with PCA
gbc_w_pca = gbc_pipeline.fit(X_train, y_train)
gbc_w_pca_pred = gbc_w_pca.predict(X_test)
metrics(gbc_w_pca_pred, y_test, 'Gradient Booster with PCA')

# Gradient Boost Classifier cross-validation
gbc_param_grid = {'gbc__n_estimators' : [100, 250, 300, 350, 400, 450],
                  'gbc__subsample' : [0.1, 0.25, 0.3, 0.4, 0.5, 1.0],
                  'gbc__max_depth' : [1, 2, 3]}

gb_clf = EstimatorCV(X_train, y_train, gbc_pipeline, gbc_param_grid)
gb_clf_pred = gb_clf.predict(X_test)
metrics(gb_clf_pred, y_test, 'Random Forest with PCA and optimized parameters')

print('\nConfusion Matrix:')
print(confusion_matrix(y_test, gb_clf_pred))




# Dump classifier, dataset, and feature list

# Save best parameters to clf variable for final model
pca = PCA(n_components = None)
gbc = GradientBoostingClassifier(loss = 'exponential',
                                 max_depth = 1,
                                 n_estimators = 250,
                                 random_state = 0, 
                                 max_features = 'auto',
                                 warm_start = True,
                                 subsample = 0.4)
gbc_pipeline = Pipeline([('pca', pca), 
                         ('gbc', gbc)])

clf = gbc_pipeline.fit(X_train, y_train)
# Format dataset and feature_list for final submission
dataset = pd.concat([y.astype(bool), gb_X], axis = 1).to_dict(orient = 'index')
feature_list = ['poi'] + gb_X.columns.tolist()

# Dump to pickles
dump_classifier_and_data(clf, dataset, feature_list)




# Final test of model (Goal is at least 0.3 for recall and accuracy scores)

from tester import test_classifier
test_classifier(clf, dataset, feature_list, folds = 1000)

