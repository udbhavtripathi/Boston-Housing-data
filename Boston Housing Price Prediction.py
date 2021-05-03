#!/usr/bin/env python
# coding: utf-8

# ## Boston housing data

# In[1]:


import pandas as pd


# In[2]:


data=pd.read_csv("Boston.csv")
data


# ## Removing first column(unwanted)

# In[3]:


df = pd.DataFrame(data)
new_data=df.drop(['Unnamed: 0'], axis = 1)


# ## Reviewing data

# In[4]:


new_data.head()


# In[5]:


new_data.describe()


# ## Plotting Data as histogram

# In[6]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
new_data.hist(bins=30,figsize=(60,50))


# In[7]:


# What happens inside test train split
# import numpy as np
# def split_train_test(data,test_ratio):
#     np.random.seed(21)
#     shuffled=np.random.permutation(len(new_data))
#     test_set_size=int(len(new_data)*test_ratio)
#     test_indices=shuffled[:test_set_size]
#     train_indices=shuffled[test_set_size:]
#     return data.iloc[train_indices],data.iloc[test_indices]
    


# In[8]:


# train_set,test_set=split_train_test(new_data,0.2)
# train_set


# ## Splitting data

# In[9]:


from sklearn.model_selection import train_test_split
train_set,test_set=train_test_split(new_data,test_size=0.2,random_state=25)
train_set.shape


# ## Performing Stratified Split for column 'chas'

# In[10]:


from sklearn.model_selection import StratifiedShuffleSplit
split=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=22)

for train_index,test_index in split.split(new_data,new_data['chas']):
    strat_train_set=new_data.loc[train_index]
    strat_test_set=new_data.loc[test_index]


# In[11]:


strat_test_set['chas'].value_counts()


# In[12]:


strat_train_set['chas'].value_counts()


# ## Ratio of splitting is equal

# In[13]:


95/7


# In[14]:


376/28


# ## Looking for correlation

# In[15]:


corr=new_data.corr()
corr['medv'].sort_values(ascending=False)


# ## Plotting correlation graph

# In[16]:


from pandas.plotting import scatter_matrix
attr=['medv','rm','zn','ptratio','lstat']
scatter_matrix(new_data[attr],figsize=(15,8))


# In[17]:


new_data.plot(kind="scatter",x="rm",y="medv",alpha=0.9)


# ## Trying attribute combination

# In[18]:


new_data['taxroom']=new_data['tax']/new_data['rm']
new_data.head()


# In[19]:


new_data.plot(kind="scatter",x="taxroom",y="medv",alpha=0.9)


# In[20]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer( strategy='median')
imputer.fit(train_set)
X=imputer.transform(train_set)
train_up=pd.DataFrame(X,columns=train_set.columns)


# In[21]:


housing=train_up.drop("medv",axis=1)
housing_label=train_up["medv"].copy()
housing.shape


# ## Scikit learn design

# Primarily three type of objects
# 1.Estimators-It estimates somes parameters based on a dataset.Eg Imputer.
# It has a fit and transform method.
# 
# 2.Transformers- transform method takes input and returns output based on the learnings from fit(). It alos has a convenience function called fit_transform() which fits and then transform.
# 
# 3.Predictors- ML models for predictions.

# ## Feature scaling

# Primarily two type of feature Scaling method-
# 1. Min-max Scaling (Normalization)
#   (value-min)/(max-min)
#   Sklearn provides a class called Minmaxscaler for this.
#   
# 2. Standarization  
#      (value-mean)/std dev
#      Sklearn provides a class called Standard Scaler for this.

# ## Creating pipelines

# In[22]:


# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler

# my_pipeline=Pipeline([
#     ('imputer',SimpleImputer(strategy="median")),
#     ('std_scaler', StandardScaler())
# ])


# ## Selecting ML-model

# In[23]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
#model=LinearRegression()
#model=DecisionTreeRegressor()
model=RandomForestRegressor()
model.fit(housing,housing_label)


# In[24]:


test_up=test_set.drop("medv",axis=1)
test_up.shape
test_label=test_set["medv"]


# In[25]:


model.predict(test_up)


# In[26]:


from sklearn.metrics import mean_squared_error
import numpy as np
housing_pred=model.predict(test_up)
mse=mean_squared_error(test_label,housing_pred)
rmse=np.sqrt(mse)
rmse


# ## ------------------------ THE END -----------------------------------
