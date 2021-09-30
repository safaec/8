#!/usr/bin/env python
# coding: utf-8

# ## IMPORT LIBRARIES

# In[1]:


import pandas as pd 

from datetime import datetime
import calendar


# ## READ THE DATASET

# In[2]:


ks_18 = pd.read_csv("ks-projects-201801.csv", encoding="ISO-8859-1")


# ## ANALYSE THE DATASET

# In[4]:


ks_18.info()
print()
print("Total projects :", len(ks_18))
print("Total features :", ks_18.columns.nunique())


# In[ ]:


ks_18.describe()


# - Missing values

# In[ ]:


ks_18.isnull().sum()


# From Kaggle
# <br> usd_pledged: conversion in US dollars of the pledged column (conversion done by kickstarter).
# <br> usd pledge real: conversion in US dollars of the pledged column (conversion from Fixer.io API).
# <br> usd goal real: conversion in US dollars of the goal column (conversion from Fixer.io API).
# <br> we are going to keep the conversion from fixer.io

# In[3]:


ks_18 = ks_18.drop(["goal", "pledged","usd pledged"], axis=1)


# - State column

# In[ ]:


(ks_18.groupby(["state"]).agg({"ID":"nunique"})*100/ len(ks_18)).sort_values(by=["ID"],ascending=False)


# From Kickstarter Platform : "But if a creator decides now is not the time for their project, they have the ability to cancel and relaunch at a later date."
# <br> From Kickstarter Platform : "A project may be suspended if our Trust & Safety team uncovers evidence that it is in violation of Kickstarter's rules"
# <br> 
# <br> We are going to keep the value "failed", "successful", "canceled".

# In[4]:


ks_18.drop(ks_18[(ks_18["state"]=="suspended")].index, inplace=True)
ks_18.drop(ks_18[(ks_18["state"]=="live")].index, inplace=True)
ks_18.drop(ks_18[(ks_18["state"]=="undefined")].index, inplace=True)


# In[5]:


ks_18["state"] = ks_18["state"].replace(["failed"],0)
ks_18["state"] = ks_18["state"].replace(["successful"],1)
ks_18["state"] = ks_18["state"].replace(["canceled"],2)


# In[6]:


(ks_18.groupby(["state"]).agg({"ID":"nunique"})*100/ len(ks_18)).sort_values(by=["ID"],ascending=False)


# - Currency

# In[ ]:


print(ks_18["currency"].nunique(), "currency")


# In[ ]:


pd.set_option('display.max_rows', None)
(ks_18.groupby(["currency"]).agg({"ID":"nunique"})*100/ len(ks_18)).sort_values(by=["ID"],ascending=False)


# In[ ]:


ks_18["currency"] = ks_18["currency"].replace(['NOK','MXN','SEK','NZD','CHF','DKK','HKD','SGD','JPY'],"Rare")


# In[ ]:


(ks_18.groupby(["currency"]).agg({"ID":"nunique"})*100/ len(ks_18)).sort_values(by=["ID"],ascending=False)


# - Category

# In[ ]:


print(ks_18["main_category"].nunique(), "main_category")


# In[ ]:


print(ks_18["category"].nunique(), "category")


# In[ ]:


pd.set_option('display.max_rows', None)
(ks_18.groupby(["category"]).agg({"ID":"nunique"})*100/ len(ks_18)).sort_values(by=["ID"],ascending=False)


# In[ ]:


(ks_18.groupby(["main_category"]).agg({"ID":"nunique"})*100/len(ks_18)).sort_values(by=["ID"],ascending=False)


# In[25]:


ks_18 = ks_18.drop(["category"], axis=1)


# - Name column

# In[ ]:


ks_18_name = ks_18[ks_18["name"]=="Cancelled (Canceled)"]
ks_18_name


# In[53]:


# Remove the project with the word "Canceled" in their name

name_projects = ["Cancelled (Canceled)", "Project Cancelled", "(Canceled)", "Canceled", "Cancelled"]


# In[28]:


remove = r'\b(?:{})\b'.format('|'.join(name_projects))
ks_18['name'] = ks_18['name'].str.replace(remove, '')


# In[29]:


ks_18[ks_18.stack().str.contains("|".join(name_projects)).any(level=0)]


# In[33]:


ks_18[ks_18["state"]==2].head()


# In[34]:


ks_18["len_name"] = ks_18["name"].str.len()


# In[35]:


ks_18 = ks_18.drop(["name"], axis=1)


# In[44]:


ks_18["len_name"] = ks_18["len_name"].fillna(0)


# - Country

# In[57]:


print(ks_18["country"].nunique(), "country")


# In[58]:


(ks_18.groupby(["country"]).agg({"ID":"nunique"})*100/len(ks_18)).sort_values(by=["ID"],ascending=False)


# In[36]:


ks_18.drop(ks_18[(ks_18["country"]=='N,0"')].index, inplace=True)


# In[37]:


ks_18["country"] = ks_18["country"].replace(["US", "CA"], "AM")
ks_18["country"] = ks_18["country"].replace(["GB", "DE", "FR", "NL", "IT", "ES", "SE", "DK","IE", "CH", "NO", "BE", "AT", "LU"], "EU")
ks_18["country"] = ks_18["country"].replace(["AU", "NZ"], "OC")
ks_18["country"] = ks_18["country"].replace(["MX"], "SA")
ks_18["country"] = ks_18["country"].replace(["HK", "SG", "JP"], "AS")


# In[38]:


ks_18 = ks_18.rename(columns={"country":"continent"})


# In[ ]:


(ks_18.groupby(["continent"]).agg({"ID":"nunique"})*100/len(ks_18)).sort_values(by=["ID"],ascending=False)


# - Date columns

# In[ ]:


ks_18["launched"] = pd.to_datetime(ks_18["launched"])
ks_18["deadline"] = pd.to_datetime(ks_18["deadline"])


# In[ ]:


ks_18["crowdfunding_duration_days"] = ks_18["deadline"]- ks_18["launched"]
ks_18["crowdfunding_duration_days"] = ks_18["crowdfunding_duration_days"].dt.days


# In[39]:


#launched
ks_18['year_launched'] = pd.DatetimeIndex(ks_18['launched']).year
ks_18['month_launched'] = pd.DatetimeIndex(ks_18['launched']).month
ks_18['day_launched'] = pd.DatetimeIndex(ks_18['launched']).day
ks_18['hour_launched'] = pd.DatetimeIndex(ks_18['launched']).hour

#drop columns
ks_18 = ks_18.drop(["deadline", "launched"], axis=1)


# In[ ]:


ks_18["year_launched"].unique()


# In[ ]:


ks_18.drop(ks_18[(ks_18["year_launched"]==1970)].index, inplace=True)


# In[ ]:


ks_18["year_launched"].unique()


# - Other columns

# In[40]:


ks_18 = ks_18.drop(["ID", "backers", ], axis=1)


# In[45]:


ks_18.head()


# In[48]:


ks_18.isnull().sum()


# In[46]:


ks_18_step1 = ks_18


# In[47]:


ks_18_step1.to_csv("ks_18_step1.csv",index=False)

