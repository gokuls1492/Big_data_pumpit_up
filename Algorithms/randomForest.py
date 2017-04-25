'''
Created on Apr 21, 2017

@author: gokul
'''  
'''Accuracy = 81.46% on 80:20 ratio'''
'''Use Random Forest'''
import numpy as np
import pandas as pd
import random
from Preprocessing.pre_processing import *
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import time
from sklearn.grid_search import GridSearchCV

def train_rf(X_train, Y_train):
    rf = RandomForestClassifier();
    rf.set_params(**getBestParams(X_train, Y_train['status_group']))
    rf.fit(X_train, Y_train['status_group'])     
    return rf

def predict_validation_result(model,X_validate,Y_validate):
    labels = model.predict(X_validate)
    print(metrics.confusion_matrix(Y_validate,labels))

    return metrics.accuracy_score(Y_validate,labels),labels

def scale_data(X_train, X_validate, test_df):
    '''
    It is important to scale data for the SVM algorithm
    Use sklearn scaler to scale validation/testing data based on training data
    '''
    
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_validate = scaler.transform(X_validate)
    test_df = scaler.transform(test_df)
    
    return X_train, X_validate,test_df

def create_submission(model,test_df, test_df_ix):
    '''
    Run prediction using model and test_df and create submission file
    
    '''
    predictions = model.predict(test_df)
    predictions_label=[]
    for pred in predictions:
        predictions_label.append(map_int_to_label(pred))
    
    submission = pd.DataFrame(data=predictions_label,  # values
                              index=test_df_ix,  # 1st column as index
                              columns=["status_group"])  # 1st row as the column names
    timestr = time.strftime("%Y%m%d-%H%M%S")
    submission.to_csv("../data/submission_rf_"+timestr+".csv")

def getBestParams(X_train, y_train, rerun = False):
    '''
    if rerun = True : Get best parameters using cross validated grid search
    if rerun = False: return the best parameters so far
    :param X_train:
    :param y_train:
    :param rerun:
    :return:
    '''

    if rerun:
        rf_grid = {'max_depth': [None],
                   'max_features': [20],
                   'min_samples_split': [1, 3],
                   'min_samples_leaf': [1, 3],
                   'bootstrap': [True],
                   'n_estimators': [50],
                   'random_state': [1]}

        grid_cv = GridSearchCV(RandomForestClassifier(), rf_grid, n_jobs=-1, verbose=True,
                               scoring='mean_squared_error').fit(X_train, y_train)
        best_params = grid_cv.best_params_
    else:
        best_params = {'bootstrap': True,
                     'max_depth': None,
                     'max_features': 30,
                     'min_samples_leaf': 1,
                     'min_samples_split': 3,
                     'n_estimators': 100,
                     'random_state': 1}

    return best_params



if __name__ == '__main__':
    '''
    Parameters to initialize:
       - If FINAL_RUN is True then run model training on full data set
       - Otherwise, set DEBUG_SMALL to True if willing to obtain quick results on 5K of data
    Load data
    Add/remove features (columns)
    Convert categorical columns to numbers or dummy columns
    Split to training/testing sets
    Scale data sets
    '''
    DEBUG_SMALL = False
    FINAL_RUN = False
    train_df, train_lbl_df, test_df = load_data()
    print('Data is loaded')
    drop_add_features(train_df,test_df,train_lbl_df)
    train_df,test_df,train_lbl_df = pre_process_data(train_df, test_df, train_lbl_df)
    random.seed(1234)
    test_df_ix=test_df['id']
    test_df.drop('id',axis=1,inplace=True)
    '''
    X_train - data frame to be used for training
    Y_train - label data corresponding to X_train
    '''
    X_train, X_validate, Y_train, Y_validate = train_test_split(train_df, train_lbl_df, test_size=0.20,random_state = 2015)
    '''X_train,X_validate,test_df = scale_data(X_train,X_validate,test_df)'''
    
    if FINAL_RUN == False:
        if DEBUG_SMALL:
            print('Running training on small sample')
            clf = train_rf(X_train[0:5000], Y_train[0:5000])
        else:
            print('Running training on full data')
            clf = train_rf(X_train, Y_train)
        accuracy, predictions = predict_validation_result(clf,X_validate,Y_validate)
        print("Accuracy on validation set", accuracy)
        create_submission(clf, test_df,test_df_ix )
    else:
        '''
        In case submission, run model on full data set
        '''
        rf = train_rf(train_df, train_lbl_df)
        create_submission(rf, test_df,test_df_ix )
