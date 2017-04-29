# Databricks notebook source
# Data Bricks notebook url: https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/3717366174502156/3827989904050330/8808643783127734/latest.html
# MAGIC %md
# MAGIC #![Spark Logo](http://spark-mooc.github.io/web-assets/images/ta_Spark-logo-small.png) + ![Python Logo](http://spark-mooc.github.io/web-assets/images/python-logo-master-v3-TM-flattened_small.png)
# MAGIC # **Big data class - Final project**
# MAGIC  
# MAGIC ## **Predicting Pumps status - Pump It Up Data Driven competition**
# MAGIC  
# MAGIC Can you predict which water pumps are faulty?
# MAGIC 
# MAGIC Using data from Taarifa and the Tanzanian Ministry of Water, can you predict which pumps are functional, which need some repairs, and which don't work at all? This is an intermediate-level practice competition. Predict one of these three classes based on a number of variables about what kind of pump is operating, when it was installed, and how it is managed. A smart understanding of which waterpoints will fail can improve maintenance operations and ensure that clean, potable water is available to communities across Tanzania.
# MAGIC https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/page/23/
# MAGIC  
# MAGIC In this Project:
# MAGIC * *Part 1*: Load Data
# MAGIC * *Part 1*: Pre-process data
# MAGIC * *Part 2*: Train and tune models
# MAGIC * *Part 3*: Predict model accuracy using SVM (sklearn)
# MAGIC * *Part 4*: Predict model accuracy using Random Forest (MLlib)
# MAGIC 
# MAGIC Look for below variable to control training data set size
# MAGIC * DEBUG_SMALL = True (current value)- will train on 5K dataset, 20% of full training set is set for validation
# MAGIC * DEBUG_SMALL = False - will train on full 80% of 50K dataset, 20% is set for validation
# MAGIC  

# COMMAND ----------

projectVersion = '1.0.0'

# COMMAND ----------

# MAGIC %md
# MAGIC #### Code and Libraries
# MAGIC - Python and Spark
# MAGIC - XGBoost
# MAGIC - MLib

# COMMAND ----------

# Data files for this assignment can be found at:
#Two locations:
#  s9rayjpv1493320587137
#  5hbwc80e1492906464748
#/FileStore/tables/s9rayjpv1493320587137/Test_set_values.csv
#/FileStore/tables/s9rayjpv1493320587137/Training_set_labels.csv
#/FileStore/tables/s9rayjpv1493320587137/Training_data_values.csv
display(dbutils.fs.ls('/FileStore/tables/5hbwc80e1492906464748'))

# COMMAND ----------

# MAGIC %md **WARNING:** If *problem with import*, required in the cell below, is not installed, follow the instructions [here](https://databricks-staging-cloudfront.staging.cloud.databricks.com/public/c65da9a2fa40e45a2028cddebe45b54c/8637560089690848/4187311313936645/6977722904629137/05f3c2ecc3.html).
# MAGIC 
# MAGIC **Pre-processing functions**

# COMMAND ----------

import sys
import os
from test_helper import Test
import pandas as pd
import sklearn
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split

sklearn.__version__

# COMMAND ----------

import sys
import os
from test_helper import Test
import pandas as pd
import numpy as np
import sklearn
from sklearn import preprocessing
import random
#from sklearn.model_selection import train_test_split

def load_data():
    train_df = pd.read_csv("/dbfs/FileStore/tables/5hbwc80e1492906464748/Training_data_values.csv")
    train_label_df = pd.read_csv("/dbfs/FileStore/tables/5hbwc80e1492906464748/Training_set_labels.csv")
    test_df    = pd.read_csv("/dbfs/FileStore/tables/5hbwc80e1492906464748/Test_set_values.csv")
    
    #train_df = train_df.drop(["id"],axis=1)
    #test_df = test_df.drop(["Id"],axis=1)
    return train_df, train_label_df, test_df

def drop_add_features(train_df,test_df,train_lbl_df):
    '''
    Remove columns:
    1. recorded_by - same value for each instance
    2. payment because same as payment_type 
    3. quantity_group same as quantity
    4. waterpoint_type_group same as waterpoint_type
    
    Drop "Id" column from Train data and Train labels
    
    Add columns:
    1. Add year, month and day based on date_recorded
    '''
    drop_columns=['recorded_by','payment','quantity_group','waterpoint_type_group']
    date_columns = ['date_recorded']
    
    for df in [train_df, test_df]:
        for col in date_columns:
            df[col] = pd.to_datetime(df[col])
            df[col + "_day"] = df[col].dt.dayofyear
            df[col + "_month"] = df[col].dt.month
            df[col + "_year"] = df[col].dt.year
        df.drop(drop_columns+date_columns,axis=1,inplace=True)

    for df in [train_df, train_lbl_df]:
        df.drop('id',axis=1,inplace=True) 
     
def map_label_to_int(label):
    if label=='functional':
        return 0
    elif label =='non functional':
        return 1
    else: # functional needs repair
        return 2

def map_int_to_label(label):
    if label == 0:
        return 'functional'
    elif label == 1:
        return 'non functional'
    else: # 2
        return 'functional needs repair'

def map_float_to_label(label):
    if label <= 0.5:
        return 'functional'
    elif label > 0.5 and label < 1.5:
        return 'non functional'
    else: # 2
        return 'functional needs repair'
    
def pre_process_data(train_df,test_df,train_lbl_df):
    ''' 
        Encode all categorical columns (defined as 'object' in data-frame
    '''     
    
    missing_columns=['public_meeting','permit','waterpoint_type']
    for f in missing_columns:
        train_df[f].fillna('Missing', inplace=True)
        test_df[f].fillna('Missing', inplace=True)
    
    print("Before Categorical pre-processing - number of features",len(train_df.columns))
    categorical=['permit','public_meeting','source_class','quantity','management_group','quality_group','waterpoint_type','source_type','payment_type',\
                 'extraction_type_class','water_quality','basin','source']
    train_df = pd.get_dummies(train_df,prefix_sep='_',columns=categorical)
    test_df = pd.get_dummies(test_df,prefix_sep='_',columns=categorical)
    print("After Categorical pre-processing - number of features",len(train_df.columns))
    
    for f in train_df.columns:
        if train_df[f].dtype == 'object':
            #print("processing Column:..",f)
            lbl = preprocessing.LabelEncoder()
            lbl.fit(np.unique(list(train_df[f].values) + list(test_df[f].values)))
            train_df[f] = lbl.transform(list(train_df[f].values))
            test_df[f]       = lbl.transform(list(test_df[f].values))
    
    # fill NaN values
    for f in train_df.columns:
        if train_df[f].dtype in ['float64','int64']:
            train_df[f].fillna(train_df[f].mean(), inplace=True)
            test_df[f].fillna(test_df[f].mean(), inplace=True)
    
    train_lbl_df['status_group'] = train_lbl_df['status_group'].apply(map_label_to_int)
        
    return train_df,test_df, train_lbl_df
train_df, train_lbl_df, test_df = load_data()
drop_add_features(train_df,test_df,train_lbl_df)
train_df,test_df,train_lbl_df = pre_process_data(train_df, test_df, train_lbl_df)

# COMMAND ----------

# MAGIC %md
# MAGIC #### **SVM model**
# MAGIC Use sklearn library SVM model to perform multi-class classification. For demo perposes the flag is set to run on small reaining set
# MAGIC * DEBUG_SMALL = True - if set to True only 5K samples will be used for training
# MAGIC * FINAL_RUN = False  - if set to False will not train model on full data set

# COMMAND ----------

# MAGIC %md
# MAGIC #### *Helper Functions*

# COMMAND ----------

from sklearn import metrics
import time
def train_svc(X_train, Y_train):
    clf = svm.SVC(decision_function_shape='ovo')
    clf.fit(X_train, Y_train['status_group']) 
#     SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
#         decision_function_shape='ovo', degree=3, gamma='auto', kernel='rbf',
#         max_iter=-1, probability=False, random_state=None, shrinking=True,
#         tol=0.001, verbose=False)
    
    return clf

def predict_validation_result(model,X_validate,Y_validate):
    labels = model.predict(X_validate)
    print(metrics.confusion_matrix(Y_validate,labels))

    return metrics.accuracy_score(Y_validate,labels),labels

def scale_data(train_df, test_df):
    '''
    It is important to scale data for the SVM algorithm
    Use sklearn scaler to scale validation/testing data based on training data
    '''
    
    scaler = preprocessing.StandardScaler().fit(train_df)
    train_df = scaler.transform(train_df)
    test_df = scaler.transform(test_df)
    
    return train_df,test_df

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
    submission.to_csv("/dbfs/FileStore/tables/5hbwc80e1492906464748/submission_svm_"+timestr+".csv")

# COMMAND ----------

# MAGIC %md
# MAGIC #### **SVM - main run**

# COMMAND ----------

from sklearn import svm

DEBUG_SMALL = True
FINAL_RUN = False
train_df, train_lbl_df, test_df = load_data()
print('Data is loaded')
drop_add_features(train_df,test_df,train_lbl_df)
train_df,test_df,train_lbl_df = pre_process_data(train_df, test_df, train_lbl_df)
test_df_ix=test_df['id']
test_df.drop('id',axis=1,inplace=True)
train_df,test_df = scale_data(train_df,test_df)
random.seed(1234)

'''
X_train - data frame to be used for training
Y_train - label data corresponding to X_train
'''

X_train, X_validate, Y_train, Y_validate = train_test_split(train_df, train_lbl_df, test_size=0.20,random_state = 2015)

if FINAL_RUN == False:
    if DEBUG_SMALL:
        print('Running training on small sample')
        clf = train_svc(X_train[0:5000], Y_train[0:5000])
    else:
        print('Running training on full sample data')
        clf = train_svc(X_train, Y_train)
    accuracy, predictions = predict_validation_result(clf,X_validate,Y_validate)
    print("Accuracy on validation set", accuracy)
    #create_submission(clf, test_df,test_df_ix )
else:
    '''
    In case submission, run model on full data set
    '''
    print('Running training on full data')
    clf = train_svc(train_df, train_lbl_df)
    #create_submission(clf, test_df,test_df_ix)

# COMMAND ----------

# MAGIC %md
# MAGIC #### **Random Forest model**
# MAGIC Use MLlib Spark library and pipeline to perform multi-class classification. For demo perposes the flag is set to run on small reaining set
# MAGIC * DEBUG_SMALL = True - if set to True only 5K samples will be used for training
# MAGIC * FINAL_RUN = False  - if set to False will not train model on full data set

# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler

DEBUG_SMALL = True
FINAL_RUN = False
train_df, train_lbl_df, test_df = load_data()
print('Data is loaded')
drop_add_features(train_df,test_df,train_lbl_df)
train_df,test_df,train_lbl_df = pre_process_data(train_df, test_df, train_lbl_df)
#print(train_lbl_df)
train_columns = list(train_df.columns)
#print(train_columns)
train_df['status_group'] = train_lbl_df['status_group']
#print(train_df.head())
test_df_ix=test_df['id']
test_df.drop('id',axis=1,inplace=True)

''' Create Spark Data Frame and add features/labels for the MLlib
'''
if DEBUG_SMALL:
  print("Running training on small data-set")
  traindf = sqlContext.createDataFrame(train_df[0:5000])
else:
  print("Running training on 80% data-set")
  traindf = sqlContext.createDataFrame(train_df)

# Below transformations are done in order to brind data-frame to the format MLlib is requiring.
# MLlib requires data-frame with two columns: labels and features. While features column is collection of all features
#
labelIndexer = StringIndexer(inputCol="status_group", outputCol="indexedLabel").fit(traindf)
assembler = VectorAssembler(inputCols=train_columns, outputCol="features")
traindf = assembler.transform(traindf)
featureIndexer =VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=3).fit(traindf)

(trainingData, testData) = traindf.randomSplit([0.8, 0.2])


#
#Best params from sklearn {'n_estimators': 120, 'random_state': 1, 'min_samples_split': 5, 'max_features': 50, 'bootstrap': True, 'max_depth': None, 'min_samples_leaf': 1}
#
# MLlib RandomForestClassifier input (from documentation)
# class pyspark.ml.classification.RandomForestClassifier(self, featuresCol="features", labelCol="label", predictionCol="prediction", probabilityCol="probability", rawPredictionCol="rawPrediction", maxDepth=5, 
#maxBins=32, minInstancesPerNode=1, minInfoGain=0.0, maxMemoryInMB=256, cacheNodeIds=False, checkpointInterval=10, impurity="gini", numTrees=20, featureSubsetStrategy="auto", seed=None, subsamplingRate=1.0)

#########################################################################
# Below code implementation is adopted from Spark MLlib main guide
# https://spark.apache.org/docs/2.0.2/ml-classification-regression.html
##########################################################################
#Train a RandomForest model.
#
rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", maxDepth=19,numTrees=100, seed=1)

# Convert indexed labels back to original labels.
labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel",
                               labels=labelIndexer.labels)

# Chain indexers and forest in a Pipeline
pipeline = Pipeline(stages=[labelIndexer, featureIndexer, rf, labelConverter])

# Train model.  This also runs the indexers.
model = pipeline.fit(trainingData)

# Make predictions.
predictions = model.transform(testData)

# Select example rows to display.
predictions.select("predictedLabel", "status_group", "features").show(5)

# Select (prediction, true label) and compute test error
evaluator = MulticlassClassificationEvaluator(
    labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test Accuracy = %g" % (accuracy))

rfModel = model.stages[2]
print(rfModel)  # summary only

