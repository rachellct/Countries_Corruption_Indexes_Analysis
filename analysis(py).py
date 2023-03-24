#!/usr/bin/env python
# coding: utf-8

# ## Countries’ Corruption Indexes Analysis

# ## I. Economic Theory
# <br/>
# 
# * What is Market capitalization and Gross Domestic Product (GDP)?
# * Market capitalization reflects the value of a company
# * Market cap to GDP ratio
# * Corruption Perceptions Index (CPI)

# In[2]:


# packages intended to use
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt


# In[12]:


# read csv file using pandas package
gdp_data_uncleaned = pd.read_csv('/Users/rache/Downloads/files/gdp_data.csv')
gdp_data_uncleaned.set_index('Country Name', inplace=True)
gdp_data_uncleaned = gdp_data_uncleaned.iloc[:, 3:]
gdp_data_uncleaned.head()


# In[13]:


# read csv file using pandas package
market_cap_data_uncleaned = pd.read_csv('/Users/rache/Downloads/files/market_cap_usd.csv')
market_cap_data_uncleaned.set_index('Country Name', inplace=True)
market_cap_data_uncleaned = market_cap_data_uncleaned.iloc[:, 3:]
market_cap_data_uncleaned.sort_values('2019', ascending=False)


# ## II. Data and Summary Statistics
# 

# In[14]:


market_cap_to_gdp = market_cap_data_uncleaned.div(gdp_data_uncleaned)
market_cap_to_gdp.sort_values('2018', ascending=False, inplace=True)
market_cap_to_gdp.head(20)
market_cap_to_gdp.describe


# In[21]:


df = pd.read_csv('/Users/rache/Downloads/files/index.csv')
df.head(50)


# In[23]:


# list of column names
market_cap_to_gdp.columns


# ## III. Variables Descriptions
# <br/>
# 
# * Corruption Perceptions Index (CPI)
# * Global Insight Country Risk Ratings
# * IMD World Competitiveness Yearbook
# * World Justice Project Rule of Law Index
# * Economist Intelligence Unit Country Ratings

# In[24]:


market_cap_to_gdp.describe


# ### Equation
# 
# Market cap to GDP ratio = a + bXi + cXi + dXi + eXi + fXi + ϵ
# 
# 
# Where:
# * Corruption Perceptions Index (CPI) = bX1
# * Global Insight Country Risk Ratings = cX2
# * IMD World Competitiveness Yearbook = dX3
# * World Justice Project Rule of Law Index = eX4
# * Economist Intelligence Unit Country Ratings = fX5
# * Residual (error) = ϵ 

# In[25]:


#Original
from pandas import DataFrame
import statsmodels.api as sm

corruption_index_data = df.fillna(method='ffill')
corruption_index_data.rename(columns={'Country': 'Country Name'}, inplace=True)

merged_data = pd.merge(corruption_index_data, market_cap_to_gdp, how='inner', on=['Country Name'])
merged_data.set_index('Country Name', inplace=True)
merged_data.dropna(subset = ['Corruption Perceptions Index (CPI)','Global Insight Country Risk Ratings','2017'],inplace=True)

x = merged_data[['Corruption Perceptions Index (CPI)','Global Insight Country Risk Ratings']]
x = sm.add_constant(x) #adding a constant

y = market_cap_to_gdp
y = merged_data[['2017']]

model = sm.OLS(y, x).fit()
predictions = model.predict(x)

print_model = model.summary()
print(print_model)


# In[27]:


get_ipython().system('pip install linearmodels')


# In[28]:


#COPY using another model 
import pandas as pd
import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt
import statsmodels.api as sm
from linearmodels import PanelOLS


corruption_index_data = df.fillna(method='ffill')
corruption_index_data.rename(columns={'Country': 'Country Name'}, inplace=True)

merged_data = pd.merge(corruption_index_data, market_cap_to_gdp, how='inner', on=['Country Name'])
merged_data.set_index('Country Name', inplace=True)
merged_data.dropna(subset = ['Corruption Perceptions Index (CPI)','Global Insight Country Risk Ratings','2017'],inplace=True)

# exog_vars = ['Corruption Perceptions Index (CPI)', 'Global Insight Country Risk Ratings']
# exog = sm.add_constant(merged_data[exog_vars])

# merged2 = merged_data.rename(columns = {'2017': 'Year'}, inplace = False)


# mod = PanelOLS(y = merged2['Year'], x = merged2[['Corruption Perceptions Index (CPI)', 'Global Insight Country Risk Ratings']], time_effects=True)

# # mod = PooledOLS(merged2.Year, exog)
# pooled_res = mod.fit()
# print(pooled_res)

x = merged_data[['Corruption Perceptions Index (CPI)','Global Insight Country Risk Ratings']]
x = sm.add_constant(x) #adding a constant

y = market_cap_to_gdp
y = merged_data[['2017']]

model = sm.OLS(y, x).fit()
predictions = model.predict(x)

print_model = model.summary()
print(print_model)


# In[29]:


#Copy for adding more variables
from pandas import DataFrame
import statsmodels.api as sm

corruption_index_data = df.fillna(method='ffill')
corruption_index_data.rename(columns={'Country': 'Country Name'}, inplace=True)

merged_data = pd.merge(corruption_index_data, market_cap_to_gdp, how='inner', on=['Country Name'])
merged_data.set_index('Country Name', inplace=True)
merged_data.dropna(subset = ['Corruption Perceptions Index (CPI)','Global Insight Country Risk Ratings','IMD World Competitiveness Yearbook','World Justice Project Rule of Law Index','Economist Intelligence Unit Country Ratings','2017'],inplace=True)

x = merged_data[['Corruption Perceptions Index (CPI)','Global Insight Country Risk Ratings','IMD World Competitiveness Yearbook','World Justice Project Rule of Law Index','Economist Intelligence Unit Country Ratings']]
x = sm.add_constant(x) #adding a constant

y = market_cap_to_gdp
y = merged_data[['2017']]

model = sm.OLS(y, x).fit()
predictions = model.predict(x)

print_model = model.summary()
print(print_model)


# ## IV. Regression Model Results
# <br/>
# 
# * We expect an average of 0.0152 increase in market cap to GDP ratio if other independent variables is constant.
# * Global Insight Country Ratings, World Competitive Yearbook, and EIU ratings have a positive effects on Market cap to GDP ratio
# * CPI and Rule of Law Index both have a negative effect on Market cap to GDP ratio

# ## V. Conclusion 
# 
# R2 from the model was low (0.238) which means the independent variables are not explaining much in the change of Market cap to GDP ratio
# 
