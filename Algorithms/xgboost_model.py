'''
Created on Apr 15, 2017

@author: uri
'''

'''
Use XgBoost to classify data
Use hyperopt to tune parameters
Parameters to initialize:
       - If GET_PARAMETERS is set to True, tuning of parameters via hyperopt will be triggered. >>>>>>>>>>It will take several hours<<<<<
       - If FINAL_RUN is True then run model training on full data set
       - If FINAL_RUN is False, set DEBUG_SMALL to True if willing to obtain quick results on 5K of data
       - In below setting example example, training will run on 80% data set and validation on 20%. Submission file will be created based on that model:
           DEBUG_SMALL = False
           FINAL_RUN = False
           GET_PARAMETERS=False
Flow:
    Load data
    Add/remove features (columns)
    Convert categorical columns to numbers or dummy columns
    Split to training/testing sets
    >>>Important<<<< : If GET_PARAMETERS is set to True, tuning of parameters via hyperopt will be triggered. >>>>>>>>>>It will take several hours<<<<<
    Scale data sets

Best Result:
Submission accuracy on 100% 82.48%
Rank on Leader-board as of 4/28/2017 - 15th
Best parameters:                  
{'colsample_bytree': 0.55, 'max_depth': 19.0, 'min_child_weight': 1.0, 'subsample': 0.9500000000000001, 'n_estimators': 78.0, 'gamma': 0.5, 'eta': 0.05}  
[[5874  480  106]
 [ 958 3580   56]
 [ 444  115  267]]
Accuracy 0.818 on 80/20 Split                    
'''
DEBUG_SMALL = False
FINAL_RUN = False
GET_PARAMETERS=False

import numpy as np
import pandas as pd
import random
from Preprocessing.pre_processing import *
from sklearn import svm 
from sklearn import metrics
import time
import xgboost as xgb
from hyperopt import hp
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

def optimize(trials):
    space = {
             'n_estimators' : hp.quniform('n_estimators', 10, 150, 1),
             'eta' : hp.quniform('eta', 0.025, 0.5, 0.025),
             'max_depth' : hp.quniform('max_depth', 1, 20, 1),
             'min_child_weight' : hp.quniform('min_child_weight', 1, 6, 1),
             'subsample' : hp.quniform('subsample', 0.5, 1, 0.05),
             'gamma' : hp.quniform('gamma', 0.5, 1, 0.05),
             'colsample_bytree' : hp.quniform('colsample_bytree', 0.5, 1, 0.05),
             'num_class' : 3,
             'eval_metric': 'merror',
             'objective': 'multi:softmax',
             'nthread' : 4,
             'silent' : 1
             }

    best = fmin(score, space, algo=tpe.suggest, trials=trials, max_evals=250)

    print ('Best parameters',best)

def score(params):
    print ("Training with params : ")
    print(params)
    num_round = int(params['n_estimators'])
    del params['n_estimators']
    params['max_depth'] = int(params['max_depth'])
    dtrain = xgb.DMatrix(X_train, label=Y_train)
    dvalid = xgb.DMatrix(X_validate, label=Y_validate)

    model = xgb.train(params, dtrain, num_round)
    accuracy, predictions = predict_validation_result(model,dvalid,Y_validate)
    score = 1 - accuracy
    print ("Accuracy",accuracy)
    return {'loss': score, 'status': STATUS_OK}
    
def predict_validation_result(model,X_validate,Y_validate):
    '''
    Translation to labels needed only when using reg:linear objective.
    If multi:softmax is used labels are classified correctly. Still the translation will leave labels unchanged
    '''
    labels = model.predict(X_validate)
    print('Prediction type and sample',type(labels),labels)
    for i,label in enumerate(labels):
        if label <= 0.5:
            labels[i]=0
        elif label > 0.5 and label < 1.5:
            labels[i] = 1
        else: # 2
            labels[i] = 2
            
    print(metrics.confusion_matrix(Y_validate,labels))

    return metrics.accuracy_score(Y_validate,labels),labels

def create_submission(model,test_df, test_df_ix):
    '''
    Run prediction using model and test_df and create submission file
    
    '''
    predictions = model.predict(test_df)
    predictions_label=[]
    for pred in predictions:
        predictions_label.append(map_float_to_label(pred))
    
    submission = pd.DataFrame(data=predictions_label,  # values
                              index=test_df_ix,  # 1st column as index
                              columns=["status_group"])  # 1st row as the column names
    timestr = time.strftime("%Y%m%d-%H%M%S")
    submission.to_csv("../data/submission_xgb_"+timestr+".csv")

if __name__ == '__main__':
    '''
    Parameters to initialize:
       - If GET_PARAMETERS is set to True, tuning of parameters via hyperopt will be triggered
       - If FINAL_RUN is True then run model training on full data set
       - Otherwise, set DEBUG_SMALL to True if willing to obtain quick results on 5K of data
    Load data
    Add/remove features (columns)
    Convert categorical columns to numbers or dummy columns
    Split to training/testing sets
    Scale data sets
    '''
    
    ''' Set parameter for the XgBoost training. It is past as an input outside of the params dictionary
    '''
    num_round=78
    
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
    if GET_PARAMETERS:
        global X_train, X_validate, Y_train, Y_validate
    
    X_train, X_validate, Y_train, Y_validate = train_test_split(train_df, train_lbl_df, test_size=0.20,random_state = 2015)
    
    #X_train=X_train.astype(float)
    '''XgBoost parameters
    '''
    param = {'max_depth':19, 'eta':0.05, 'silent':1, 'min_child_weight':1, 'subsample' : 0.95 ,
                      'num_class' : 3, 'nthread' : 4,"objective"   : "multi:softmax",
                      'eval_metric': 'merror','colsample_bytree':0.55, 'gamma': 0.5}
    
    if GET_PARAMETERS:
        '''
        If running to evaluate best parameters and exit
        '''
        trials = Trials()
        optimize(trials)
        exit
    
    if FINAL_RUN == False:
        if DEBUG_SMALL:
            print('Running training on small sample')
            dtrain=xgb.DMatrix(X_train[0:5000],label=Y_train[0:5000],missing=np.NaN)
            dtest=xgb.DMatrix(X_validate,missing=np.NaN)
            watchlist  = [(dtrain,'train')]
            bst = xgb.train(param, dtrain, num_round, watchlist)
            #y_test_bst=bst.predict(dtest)

        else:
            print('Running training on full sample data')
            dtrain=xgb.DMatrix(X_train,label=Y_train,missing=np.NaN)
            dtest=xgb.DMatrix(X_validate,missing=np.NaN)
            watchlist  = [(dtrain,'train')]
            bst = xgb.train(param, dtrain, num_round, watchlist)
            
        accuracy, predictions = predict_validation_result(bst,dtest,Y_validate)
        print("Accuracy on validation set", accuracy)
        
        dtest_sub=xgb.DMatrix(test_df,missing=np.NaN)
        create_submission(bst, dtest_sub,test_df_ix )
    else:
        '''
        In case submission, run model on full data set
        '''
        print('Running training on full data')
        dtrain=xgb.DMatrix(train_df,label=train_lbl_df,missing=np.NaN)
        bst = xgb.train(param, dtrain, num_round)
        dtest_sub=xgb.DMatrix(test_df,missing=np.NaN)
        create_submission(bst, dtest_sub,test_df_ix )
        print('Completed Running training on full data')
    
    
    