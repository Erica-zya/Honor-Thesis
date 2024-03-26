#!/usr/bin/env python
# coding: utf-8

# In[17]:


get_ipython().system('pip install pandas')
import pandas as pd
import numpy as np


# # AR Model

# In[18]:


carbon_price_url = 'https://raw.githubusercontent.com/Erica-zya/Honor-Thesis/main/Data/Data_New/Daily%20Carbon%20Price.csv'
carbon_price = pd.read_csv(carbon_price_url, header=0)
carbon_price['Date'] = pd.to_datetime(carbon_price['Date'])


# In[19]:


carbon_price = carbon_price.replace([np.inf, -np.inf], np.nan)
carbon_price.dropna(subset=['Carbon'], inplace=True)


# ### AR(1) Model

# In[20]:


from statsmodels.tsa.ar_model import AutoReg

y = carbon_price['Carbon']

modelAR1 = AutoReg(y, lags=1)
resultAR1 = modelAR1.fit()

print(resultAR1.summary())


# ### Unit Root Test

# In[21]:


from statsmodels.tsa.stattools import adfuller

adf_result = adfuller(y)

print('ADF Statistic: %f' % adf_result[0])
print('p-value: %f' % adf_result[1])
print('Critical Values:')
for key, value in adf_result[4].items():
    print('\t%s: %.3f' % (key, value))


# ### Differencing

# In[22]:


y_diff = y.diff().dropna()


modelAR1_diff = AutoReg(y_diff, lags=1)
resultAR1_diff = modelAR1_diff.fit()

print(resultAR1_diff.summary())


# In[23]:


y_diff = y.diff().dropna()

adf_result_diff = adfuller(y_diff)

print('ADF Statistic: %f' % adf_result_diff[0])
print('p-value: %f' % adf_result_diff[1])
print('Critical Values:')
for key, value in adf_result_diff[4].items():
    print('\t%s: %.3f' % (key, value))


# ### AR(2) Model

# In[66]:


modelAR2 = AutoReg(y, lags=2)
resultAR2 = modelAR2.fit()

print(resultAR2.summary())


# # ARIMA Model

# In[53]:


carbon_price_copy = pd.read_csv(carbon_price_url, header=0)
carbon_price_copy['Date'] = pd.to_datetime(carbon_price['Date'])

energy_price_url = 'https://raw.githubusercontent.com/Erica-zya/Honor-Thesis/main/Data/Data_New/Daily%20Energy.csv'
energy_price = pd.read_csv(energy_price_url, header=0)
energy_price['Date'] = pd.to_datetime(energy_price['Date'])

# Perform the merges
merged_df = pd.merge(carbon_price_copy, energy_price, on='Date', how='inner')


# In[54]:


merged_df = merged_df.replace([np.inf, -np.inf], np.nan)
merged_df.dropna(subset=['Carbon'], inplace=True)
merged_df.dropna(subset=['Coal'], inplace=True)
merged_df.dropna(subset=['Crude Oil'], inplace=True)
merged_df.dropna(subset=['Natural Gas'], inplace=True)


# In[55]:


from statsmodels.tsa.arima.model import ARIMA

X = merged_df[['Coal', 'Crude Oil', 'Natural Gas']]
_y = merged_df['Carbon']


# ### ARIMA(0,0,0)

# In[58]:


model_arima000 = ARIMA(_y, exog=X, order=(0,0,0))
model_arima000_fit = model_arima000.fit()

print(model_arima000_fit.summary())


# ### ARIMA(1,0,0)

# In[59]:


model_arima100 = ARIMA(_y, exog=X, order=(1,0,0))
model_arima100_fit = model_arima100.fit()

print(model_arima100_fit.summary())


# ### ARIMA(0,1,0)

# In[60]:


model_arima010 = ARIMA(_y, exog=X, order=(0,1,0))
model_arima010_fit = model_arima010.fit()

print(model_arima010_fit.summary())


# ### ARIMA(0,0,1)

# In[61]:


model_arima001 = ARIMA(_y, exog=X, order=(0,0,1))
model_arima001_fit = model_arima001.fit()

print(model_arima001_fit.summary())


# ### ARIMA(0,1,1)

# In[62]:


model_arima011 = ARIMA(_y, exog=X, order=(0,1,1))
model_arima011_fit = model_arima011.fit()

print(model_arima011_fit.summary())


# ### ARIMA(1,0,1)

# In[63]:


model_arima101 = ARIMA(_y, exog=X, order=(1,0,1))
model_arima101_fit = model_arima101.fit()

print(model_arima101_fit.summary())


# ### ARIMA(1,1,0); theoretical optimal model

# In[64]:


model_arima110 = ARIMA(_y, exog=X, order=(1,1,0))
model_arima110_fit = model_arima110.fit()

print(model_arima110_fit.summary())


# ### ARIMA(1,1,1); best performing model

# In[65]:


model_arima111 = ARIMA(_y, exog=X, order=(1,1,1))
model_arima111_fit = model_arima111.fit()

print(model_arima111_fit.summary())


# In[68]:


import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

plt.figure(figsize=(10, 6))
plot_acf(merged_df['Carbon'], lags=5, alpha=0.05)
plt.title('Autocorrelation Function for Carbon Price')
plt.show()


# In[70]:


from statsmodels.graphics.tsaplots import plot_pacf

plt.figure(figsize=(10, 6))
plot_pacf(_y, lags=5, alpha=0.05)
plt.title('Partial Autocorrelation Function')
plt.show()

