#!/usr/bin/env python
# coding: utf-8

# # GRAPH ANALYS

# In[1]:


get_ipython().system(' pip install plotly')


# In[1]:


import plotly.express as px
import plotly.graph_objs as go
import pandas as pd


# In[2]:


datasample = (pd.read_csv("ks_18_step1.csv")).sample(30000)


# In[12]:


datasample


# In[16]:



fig = px.bar(data_frame=datasample, x="usd_goal_real", y="state")
fig.show()


# In[14]:



fig = px.pie(data_frame=datasample, names="state", title="Répartition du dataset", 
             labels={"0":"Failed", "1":"Succed", "2":"Canceled"}, )

fig.show()


# In[3]:



fig = go.Figure(go.Sunburst(labels="main_category"))

print(fig.show())


# In[10]:



fig = px.pie(data_frame=datasample, names="country_US", title="Continent de provenance des projets")
fig.update_traces(textposition='inside', textinfo='percent')
fig.show()


# In[9]:



fig = px.pie(data_frame=datasample, names="country_US", title="Proportion des projets provenant des USA")
fig.show()


# In[ ]:


- Corellation between the features and the state ?


# In[33]:


fig1 = px.box(data_frame=datasample, x="continent", y="state", color="continent")
fig1.show("iframe")


# In[39]:



# create trace1 
trace1 = go.Bar(
                x = datasample.continent,
                y = datasample.state==0,
                name = "Failed",
                text = datasample.continent)
# create trace2 
trace2 = go.Bar(
                x = datasample.continent,
                y = datasample.state==1,
                name = "Succed",
                text = datasample.continent)
# create trace2 
trace3 = go.Bar(
                x = datasample.continent,
                y = datasample.state==2,
                name = "Canceled",
                text = datasample.continent)
data = [trace1, trace2, trace3]
layout = go.Layout(barmode = "group")
fig = go.Figure(data = data, layout = layout)
fig.show("iframe")


# In[27]:


fig3 = px.histogram(data_frame=datasample, x="main_category", y="state", color="main_category")
fig3.show("iframe")


# In[28]:



fig6 = px.bar(data_frame=datasample, x="state", y="month_launched", color="month_launched")
fig6.show("iframe")


# In[30]:


fig7 = px.histogram(data_frame=datasample, x="usd_pledged_real", y="state", color="state")
fig7.show("iframe")


# In[ ]:




