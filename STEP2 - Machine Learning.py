#!/usr/bin/env python
# coding: utf-8

# ## PREPROCESSING ML

# In[2]:


import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import  OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from plotly.subplots import make_subplots


# In[3]:


ks_18 = pd.read_csv("ks_18_step1.csv")


# - Separating the variables

# In[4]:


# Separate target variable Y from features X

print("Separating labels from features...")
Y = ks_18.loc[:,"state"]
X = ks_18.loc[:, ks_18.columns !="state"]
print("...Done.")
print(Y.head())
print()
print(X.head())
print()


# In[5]:


# Convert pandas DataFrames to numpy arrays before using scikit-learn
print("Convert pandas DataFrames to numpy arrays...")
X = X.values
Y = Y.tolist()


# In[6]:


print("Dividing into train and test sets...")
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)


# - Creating Pipeline

# In[7]:


# Create pipeline for numeric features
numeric_features = [3,4,5,7] # Positions of numeric columns in X_train/X_test
numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])


# In[8]:


# Create pipeline for categorical features
categorical_features = [0,1,2,6,8,9,10,11]
categorical_transformer = Pipeline(steps=[('encoder', OneHotEncoder(drop='first'))])


# In[9]:


# Use ColumnTransformer to make a preprocessor object that describes all the treatments to be done
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])


# In[10]:


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


get_ipython().system('pip install xgboost')


# In[11]:


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


# In[12]:


corr = ks_18.corr()
sns.heatmap(corr)


# ### MODEL TRAINING

# In[13]:


model_reg = LogisticRegression()

model_gridsearch_dtc = DecisionTreeClassifier(criterion="entropy", max_depth=9, min_samples_split=20)

model_gridsearch_rfc = GridSearchCV(RandomForestClassifier(), 
                                    param_grid ={'max_depth': np.arange(1,10,2),
                                                 'min_samples_leaf': [1, 2, 5],
                                                 'min_samples_split': [2, 4, 8],
                                                 'n_estimators': [10, 20, 40]}, 
                                    cv = 3, verbose=1)

model_xgb = xgb.XGBClassifier(random_state=1,learning_rate=0.01)


# In[14]:


model_reg.fit(X_train,Y_train)


# In[ ]:


model_gridsearch_dtc.fit(X_train,Y_train)


# In[16]:


model_gridsearch_rfc.fit(X_train,Y_train)


# In[17]:


model_xgb.fit(X_train,Y_train)


# ### PREDICTION

# In[18]:


# Predictions on training set
Y_train_pred_reg = model_reg.predict(X_train)
Y_train_pred_dtc = model_gridsearch_dtc.predict(X_train)
Y_train_pred_rfc = model_gridsearch_rfc.predict(X_train)
Y_train_pred_xgb = model_xgb.predict(X_train)

# Predictions on test set
Y_test_pred_reg = model_reg.predict(X_test)
Y_test_pred_dtc = model_gridsearch_dtc.predict(X_test)
Y_test_pred_rfc = model_gridsearch_rfc.predict(X_test)
Y_test_pred_xgb = model_xgb.predict(X_test)


# In[62]:


# Print scores

Accuracy_training_reg = accuracy_score(Y_train, Y_train_pred_reg)
Accuracy_test_reg = accuracy_score(Y_test, Y_test_pred_reg)
f1score_training_reg =f1_score(Y_train, Y_train_pred_reg, average="micro")
f1score_test_reg = f1_score(Y_test, Y_test_pred_reg, average="micro")

Accuracy_training_dtc = accuracy_score(Y_train, Y_train_pred_dtc)
Accuracy_test_dtc = accuracy_score(Y_test, Y_test_pred_dtc)
f1score_training_dtc =f1_score(Y_train, Y_train_pred_dtc, average="micro")
f1score_test_dtc = f1_score(Y_test, Y_test_pred_dtc, average="micro")

Accuracy_training_rfc = accuracy_score(Y_train, Y_train_pred_rfc)
Accuracy_test_rfc = accuracy_score(Y_test, Y_test_pred_rfc)
f1score_training_rfc =f1_score(Y_train, Y_train_pred_rfc, average="micro")
f1score_test_rfc= f1_score(Y_test, Y_test_pred_rfc, average="micro")

Accuracy_training_xgb = accuracy_score(Y_train, Y_train_pred_xgb)
Accuracy_test_xgb = accuracy_score(Y_test, Y_test_pred_xgb)
f1score_training_xgb =f1_score(Y_train, Y_train_pred_xgb, average="micro")
f1score_test_xgb = f1_score(Y_test, Y_test_pred_xgb, average="micro")

score = {"Model":["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost"], 
         "Accuracy on training set" : [Accuracy_training_reg, Accuracy_training_dtc, Accuracy_training_rfc, Accuracy_training_xgb],
        "Accuracy on test set" : [Accuracy_test_reg, Accuracy_test_dtc, Accuracy_test_rfc, Accuracy_test_xgb],
        "F1-score on training set" : [f1score_training_reg, f1score_training_dtc, f1score_training_rfc, f1score_training_xgb],
         "F1-score on test set" : [f1score_test_reg, f1score_test_dtc, f1score_test_rfc, f1score_test_xgb]}

score = pd.DataFrame(score)
score


# ###  Stacking Classifier 

# In[21]:


st = StackingClassifier(estimators=[("Decision tree",model_gridsearch_dtc),
    ("Random forest",model_gridsearch_rfc),
    ("Logistic regression",model_reg),
    ("XGBoost",model_xgb)])


# In[22]:


st.fit(X_train, Y_train)

print("Score for the stacking classifier on the train set : {}".format(st.score(X_train, Y_train)))
print("\n")
print("Score for the stacking classifier on the test set : {}".format(st.score(X_test, Y_test)))


# In[29]:



vc = VotingClassifier(estimators=[("Decision tree",model_gridsearch_dtc),
    ("Random forest",model_gridsearch_rfc),
    ("Logistic regression",model_reg),
    ("XGBoost",model_xgb)])


# In[31]:



vc = vc.fit(X_train,Y_train)

print("Score for the stacking classifier on the train set : {}".format(vc.score(X_train, Y_train)))
print("\n")
print("Score for the stacking classifier on the test set : {}".format(vc.score(X_test, Y_test)))


# In[1]:


cm = confusion_matrix(Y_test, Y_test_pred_reg)
sns.heatmap(cm, annot=True, fmt="d")


# In[24]:


cm = confusion_matrix(Y_test, Y_test_pred_dtc)
sns.heatmap(cm, annot=True, fmt="d")


# In[25]:


cm = confusion_matrix(Y_test, Y_test_pred_rfc)
sns.heatmap(cm, annot=True, fmt="d")


# In[26]:


cm = confusion_matrix(Y_test, Y_test_pred_xgb)
sns.heatmap(cm, annot=True, fmt="d")


# In[55]:


print("Classification Report Logistic Regression")
print("Train")
y_true = Y_train
y_pred = model_reg.predict(X_train)
print(classification_report(y_true, y_pred))


# In[58]:


print("Classification Report Logistic Regression")
print("Test")
y_true = Y_test
y_pred = model_reg.predict(X_test)
print(classification_report(y_true, y_pred))


# In[60]:


print("Classification Report Decision Tree")
print("Train")
y_true = Y_train
y_pred = model_gridsearch_dtc.predict(X_train)
print(classification_report(y_true, y_pred))


# In[61]:


print("Classification Report Decision Tree")
print("Test")
y_true = Y_test
y_pred = model_gridsearch_dtc.predict(X_test)
print(classification_report(y_true, y_pred))


# In[64]:


print("Classification Report Random Forest")
print("Train")
y_true = Y_train
y_pred = model_gridsearch_rfc.predict(X_train)
print(classification_report(y_true, y_pred))


# In[65]:


print("Classification Report Random Forest")
print("Test")
y_true = Y_test
y_pred = model_gridsearch_rfc.predict(X_test)
print(classification_report(y_true, y_pred))


# In[66]:


print("Classification Report XGBoost")
print("Train")
y_true = Y_train
y_pred = model_xgb.predict(X_train)
print(classification_report(y_true, y_pred))


# In[67]:


print("Classification Report XGBoost")
print("Test")
y_true = Y_test
y_pred = model_xgb.predict(X_test)
print(classification_report(y_true, y_pred))


# In[ ]:




