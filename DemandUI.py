#!/usr/bin/env python
# coding: utf-8

# In[3]:


import streamlit as st
from pycaret.regression import load_model,predict_model
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
from functools import reduce
import plotly.express as px


# In[18]:


def temp_model():
    tempModel=load_model("TemperatureForecastingModel")
    return tempModel
def humidity_model():
    humidityModel=load_model("HumidityForecastingModel")
    return humidityModel
def rainfall_model():
    rainfallModel=load_model("AvgRainfallForecastingModel")
    return rainfallModel
def windspeed_model():
    windspeedModel=load_model("AvgWindspeedForecastingModel")
    return windspeedModel
def ur_model():
    urModel=load_model("URForecastingModel")
    return urModel
def demand_model():
    demandModel=load_model("DemandForecastingModel")
    return demandModel
    
def app_layout():
    st.title("Demand Forecasting")
    FutureMonths=st.number_input('Months',min_value=1,max_value=24,value=5)
    return FutureMonths
def fig(df,date,y):
    newDF=df
    newDF['Date']=date
    fig = px.line(newDF, x='Date', y=[y], template = 'plotly_dark',width=500,height=300)
    return fig
    
    

def predict():
    start='2022-07-01'
    starttime=datetime.strptime(start,'%Y-%m-%d')
    future_date = starttime + relativedelta(months=app_layout())
    end=future_date.strftime('%Y-%m-%d')
    future_dates = pd.date_range(start = start, end = end, freq = 'MS')
    future_df = pd.DataFrame()
    future_df['Month'] = [i.month for i in future_dates]
    future_df['Year'] = [i.year for i in future_dates]    
    future_df['Series'] = np.arange(79,(79+len(future_dates)))
    model1=temp_model()
    temp_future = predict_model(model1, data=future_df)
    temp_future.columns=['Month','Year','Series','ForecastedTemperature']
    
    st.subheader("Temperature Forecasting")
    col1,col2=st.columns(2)
    col1.dataframe(temp_future)
    image=fig(temp_future,future_dates,'ForecastedTemperature')
    col2.plotly_chart(image)
    
    st.subheader("Humidity Forecasting")
    col1,col2=st.columns(2)
    model2=humidity_model()
    humidity_future = predict_model(model2, data=future_df)
    humidity_future.columns=['Month','Year','Series','ForecastedHumidity']
    col1.dataframe(humidity_future)
    image=fig(humidity_future,future_dates,'ForecastedHumidity')
    col2.plotly_chart(image)
    
    st.subheader("Rainfall Forecasting")
    col1,col2=st.columns(2)
    model3=rainfall_model()
    rainfall_future = predict_model(model3, data=future_df)
    rainfall_future.columns=['Month','Year','Series','ForecastedRainfall']
    col1.dataframe(rainfall_future)
    image=fig(rainfall_future,future_dates,'ForecastedRainfall')
    col2.plotly_chart(image)
    
    st.subheader("WindSpeed Forecasting")
    col1,col2=st.columns(2)
    model4=windspeed_model()
    windspeed_future = predict_model(model4, data=future_df)
    windspeed_future.columns=['Month','Year','Series','ForecastedWindSpeed']
    col1.dataframe(windspeed_future)
    image=fig(windspeed_future,future_dates,'ForecastedWindSpeed')
    col2.plotly_chart(image)

    st.subheader("Unemployment Rate Forecasting")
    col1,col2=st.columns(2)
    model5=ur_model()
    ur_future = predict_model(model5, data=future_df)
    ur_future.columns=['Month','Year','Series','ForecastedUnemploymentRate']
    col1.dataframe(ur_future)
    image=fig(ur_future,future_dates,'ForecastedUnemploymentRate')
    col2.plotly_chart(image)  
    
    st.subheader("Demand Forecasting")
    col1,col2=st.columns(2)
    model6=demand_model()
    newDF=[temp_future,humidity_future,windspeed_future,rainfall_future,ur_future]
    final_df = reduce(lambda  left,right: pd.merge(left,right,on=['Series','Year','Month'],
                                            how='inner'), newDF)
    
    predict_df=final_df[['ForecastedTemperature','ForecastedHumidity','ForecastedWindSpeed','ForecastedRainfall','ForecastedUnemploymentRate','Month','Year','Series']]
    predict_df.columns=['AvgTemp', 'AvgHumidity', 'AvgWindspeed', 'AvgRainfall','UnemploymentRate', 'Month', 'Year', 'Series']
    #['AvgTemp', 'AvgHumidity', 'AvgWindspeed', 'AvgRainfall','UnemploymentRate', 'Month', 'Year', 'Series']
    
    demand_future = predict_model(model6, data=predict_df)
    demand_future.columns=['AvgTemp', 'AvgHumidity', 'AvgWindspeed', 'AvgRainfall','UnemploymentRate', 'Month', 'Year', 'Series','ForecastedDemand']
    col1.dataframe(demand_future)
    image=fig(demand_future,future_dates,'ForecastedDemand')
    col2.plotly_chart(image)


# In[ ]:


predict()

