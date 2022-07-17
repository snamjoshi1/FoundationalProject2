
import streamlit as st
from pycaret.regression import load_model,predict_model
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
from functools import reduce
import plotly.express as px



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
  
    
def term():
    st.title("Energy Consumption Prediction")
    option = st.selectbox('Select Term for Forecast',('Short Term', 'Long Term'))
    return option
    
def app_layout(option):
    if option=='Short Term':
        FutureMonths=st.number_input('Months',min_value=1,max_value=12,value=3)
        return FutureMonths
    elif option=='Long Term':
        FutureMonths=st.number_input('Months',min_value=13,max_value=24,value=13)
        return FutureMonths
        
def fig(df,date,y):
    newDF=df
    newDF['Date']=date
    fig = px.line(newDF, x='Date', y=[y], template = 'plotly_dark',width=500,height=300)
    newDF.drop(['Date'],axis=1,inplace=True)
    return fig
    
    

def predict():
    start='2022-07-01'
    starttime=datetime.strptime(start,'%Y-%m-%d')
    option=term()
    fmonths=app_layout(option)
    future_date = starttime + relativedelta(months=fmonths-1)
    end=future_date.strftime('%Y-%m-%d')
    future_dates = pd.date_range(start = start, end = end, freq = 'MS')
    future_df = pd.DataFrame()
    future_df['Series'] = np.arange(151,(151+len(future_dates)))
    future_df['Month'] = [i.month for i in future_dates]
    future_df['Year'] = [i.year for i in future_dates]    
    model1=temp_model()
    temp_future = predict_model(model1, data=future_df)
    temp_future.columns=['Series','Month','Year','ForecastedTemperature']
                     
    
    st.subheader("Temperature Forecasting")
    col1,col2=st.columns(2)
    col1.dataframe(temp_future.style.format({'ForecastedTemperature':'{:.2f}'}))
    image=fig(temp_future,future_dates,'ForecastedTemperature')
    col2.plotly_chart(image)
    
    st.subheader("Humidity Forecasting")
    col1,col2=st.columns(2)
    model2=humidity_model()
    humidity_future = predict_model(model2, data=future_df)
    humidity_future.columns=['Series','Month','Year','ForecastedHumidity']
    col1.dataframe(humidity_future.style.format({'ForecastedHumidity':'{:.2f}'}))
    image=fig(humidity_future,future_dates,'ForecastedHumidity')
    col2.plotly_chart(image)
    
    st.subheader("Rainfall Forecasting")
    col1,col2=st.columns(2)
    model3=rainfall_model()
    rainfall_future = predict_model(model3, data=future_df)
    rainfall_future.columns=['Series','Month','Year','ForecastedRainfall']
    col1.dataframe(rainfall_future.style.format({'ForecastedRainfall':'{:.2f}'}))
    image=fig(rainfall_future,future_dates,'ForecastedRainfall')
    col2.plotly_chart(image)
    
    st.subheader("WindSpeed Forecasting")
    col1,col2=st.columns(2)
    model4=windspeed_model()
    windspeed_future = predict_model(model4, data=future_df)
    windspeed_future.columns=['Series','Month','Year','ForecastedWindSpeed']
    col1.dataframe(windspeed_future.style.format({'ForecastedWindSpeed':'{:.2f}'}))
    image=fig(windspeed_future,future_dates,'ForecastedWindSpeed')
    col2.plotly_chart(image)

    st.subheader("Unemployment Rate Forecasting")
    col1,col2=st.columns(2)
    model5=ur_model()
    ur_future = predict_model(model5, data=future_df)
    ur_future.columns=['Series','Month','Year','ForecastedUnemploymentRate']
    col1.dataframe(ur_future.style.format({'ForecastedUnemploymentRate':'{:.2f}'}))
    image=fig(ur_future,future_dates,'ForecastedUnemploymentRate')
    col2.plotly_chart(image)  
    
    st.subheader("Demand Prediction")
    col1,col2=st.columns(2)
    model6=demand_model()
    newDF1=[temp_future,humidity_future,windspeed_future,rainfall_future,ur_future]
    final_df = reduce(lambda  left,right: pd.merge(left,right,on=['Series','Month','Year'],
                                            how='inner'), newDF1)
    
    
    predict_df=final_df[['Series','Month', 'Year','ForecastedTemperature','ForecastedHumidity','ForecastedWindSpeed','ForecastedRainfall','ForecastedUnemploymentRate']]
    predict_df.columns=[ 'Series','Month', 'Year','AvgTemp', 'AvgHumidity', 'AvgWindspeed', 'AvgRainfall','UnemploymentRate']
    #['AvgTemp', 'AvgHumidity', 'AvgWindspeed', 'AvgRainfall','UnemploymentRate', 'Month', 'Year', 'Series']
    
    demand_future = predict_model(model6, data=predict_df)
    demand_future.columns=['Series','Month', 'Year','AvgTemp', 'AvgHumidity', 'AvgWindspeed', 'AvgRainfall','UnemploymentRate','ForecastedDemand']
    col1.dataframe(demand_future.style.format({'AvgTemp':'{:.2f}','AvgHumidity':'{:.2f}','AvgWindspeed':'{:.2f}','UnemploymentRate':'{:.2f}','ForecastedDemand':'{:.2f}'}))
    image=fig(demand_future,future_dates,'ForecastedDemand')
    col2.plotly_chart(image)



predict()

