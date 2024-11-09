#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
from datetime import date

import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import pandas as pd


# In[2]:


START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")


# In[3]:


st.title('Stock Forecast App')


# In[4]:


stocks = ('GOOG', 'AAPL', 'MSFT', 'GME')
selected_stock = st.selectbox('Select dataset for prediction', stocks)


# In[5]:


n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365


# In[6]:


def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data


# In[7]:


data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')


# In[8]:


st.subheader('Raw data')
st.write(data.tail())


# In[9]:


# Plot raw data
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
    fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()


# In[10]:


# Predict forecast with Prophet.
df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})


# In[11]:


print(df_train.head())
print(df_train.info())


# In[12]:


df_train.columns = ['ds', 'y']


# In[13]:


df_train


# In[14]:


min_value = df_train['y'].min()
max_value = df_train['y'].max()
print(f'min_value: {min_value} & max_value: {max_value}')


# In[15]:


# Convert 'ds' column to datetime
df_train['ds'] = pd.to_datetime(df_train['ds'])

# Remove timezone information from 'ds'
df_train['ds'] = df_train['ds'].dt.tz_localize(None)


# In[16]:


m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)


# In[17]:


# Show and plot forecast
st.subheader('Forecast data')
st.write(forecast.tail())


# In[18]:


st.write(f'Forecast plot for {n_years} years')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)


# In[41]:


st.write("Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)

