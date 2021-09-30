#!/usr/bin/env python
# coding: utf-8

# ## PREPROCESSING ML

# In[1]:


import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import  OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from plotly.subplots import make_subplots


# In[2]:


ks_18 = pd.read_csv("ks_18_step1.csv")


# - Separating the variables

# In[3]:


# Separate target variable Y from features X

print("Separating labels from features...")
Y = ks_18.loc[:,"state"]
X = ks_18.loc[:, ks_18.columns !="state"]
print("...Done.")
print(Y.head())
print()
print(X.head())
print()


# In[4]:


# Convert pandas DataFrames to numpy arrays before using scikit-learn
print("Convert pandas DataFrames to numpy arrays...")
X = X.values
Y = Y.tolist()


# In[5]:


print("Dividing into train and test sets...")
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)


# - Creating Pipeline

# In[6]:


# Create pipeline for numeric features
numeric_features = [3,4,5,7] # Positions of numeric columns in X_train/X_test
numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])


# In[7]:


# Create pipeline for categorical features
categorical_features = [0,1,2,6,8,9,10,11]
categorical_transformer = Pipeline(steps=[('encoder', OneHotEncoder(drop='first'))])


# In[8]:


# Use ColumnTransformer to make a preprocessor object that describes all the treatments to be done
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])


# In[9]:


# Preprocessings on train set
print("Performing preprocessings on train set...")
print(X_train[0:5, 0:5])
X_train = preprocessor.fit_transform(X_train)
print('...Done.')
print(X_train[0:5, 0:5])
print()

# Preprocessings on test set
print("Performing preprocessings on test set...")
print(X_test[0:5, 0:5])
X_test = preprocessor.transform(X_test) 
print('...Done.')
print(X_test[0:5, 0:5])
print()


# ## MACHINE LEARNING

# In[10]:


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve,  classification_report
from sklearn.ensemble import StackingClassifier, VotingClassifier
from xgboost import XGBClassifier
import xgboost as xgb

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) # to avoid deprecation warnings

import matplotlib.pyplot as plt
import seaborn as sns 
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
# setting Jedha color palette as default
pio.templates["jedha"] = go.layout.Template(
    layout_colorway=["#4B9AC7", "#4BE8E0", "#9DD4F3", "#97FBF6", "#2A7FAF", "#23B1AB", "#0E3449", "#015955"]
)
pio.templates.default = "jedha"
pio.renderers.default = "svg" # to be replaced by "iframe" if working on JULIE


# ### MODEL TRAINING

# In[13]:


model_dtc =DecisionTreeClassifier(criterion="entropy", max_depth=9, min_samples_split=20)


# In[14]:


model_dtc.fit(X_train,Y_train)


# ### PREDICTION

# In[15]:


# Predictions on training set
Y_train_pred_dtc = model_dtc.predict(X_train)

# Predictions on test set
Y_test_pred_dtc = model_dtc.predict(X_test)


# In[16]:


# Print scores

print("Accuracy_training_dtc =", accuracy_score(Y_train, Y_train_pred_dtc))
print("Accuracy_test_dtc =", accuracy_score(Y_test, Y_test_pred_dtc))
print("f1score_training_dtc =", f1_score(Y_train, Y_train_pred_dtc, average="micro"))
print("f1score_test_dtc =", f1_score(Y_test, Y_test_pred_dtc, average="micro")
)


# In[ ]:


import joblib


# In[ ]:


joblib.dump(model_dtc, "model_dtc.joblib")


# In[ ]:


joblib.dump(model_dtc.columns, 'model_columns.joblib')

