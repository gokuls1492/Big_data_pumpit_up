'''
Created on Apr 15, 2017

@author: uri
'''

'''
Use XgBoost to classify data

Result:

rounds = 5000

[[5206 1220   34]
 [ 717 3838   39]
 [ 235  427  164]]
Accuracy on validation set 0.775084175084
Submission accuracy 80% 77.39%
                    100% 78.19%
'''
import numpy as np
import pandas as pd
import random
from Preprocessing.pre_processing import *
from sklearn import svm 
from sklearn import metrics
import time
import xgboost as xgb


def predict_validation_result(model,X_validate,Y_validate):
    labels = model.predict(X_validate)
    print(type(labels),labels)
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
       - If FINAL_RUN is True then run model training on full data set
       - Otherwise, set DEBUG_SMALL to True if willing to obtain quick results on 5K of data
    Load data
    Add/remove features (columns)
    Convert categorical columns to numbers or dummy columns
    Split to training/testing sets
    Scale data sets
    '''
    DEBUG_SMALL = False
    FINAL_RUN = True
    num_round=5000
    
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
    
    #X_train=X_train.astype(float)
    '''XgBoost parameters
    '''
    param = {'max_depth':10, 'eta':10**-2, 'silent':1, 'min_child_weight':1, 'subsample' : 0.7 ,"early_stopping_rounds":10,
                      "objective"   : "reg:linear",'eval_metric': 'rmse','colsample_bytree':0.8}
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
    
    
    