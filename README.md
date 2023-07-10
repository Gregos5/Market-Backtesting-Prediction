# Market prediction with Machine learning modelling and backtesting

## Overview and Motivation
The initial aim of this project was to achieve a cryptocurrency prediction model with a precision above 60%.

The goal is to provide a user interface on which the user can select the predictors and optimise backtesting parameters to generate a cryptocurrency market model (random forest classifiers) (BTC_USD) to predict the direction of the next hourly candle using several predictors and backtesting on a very large amount of data (BTC 1H)

## Data 
All data is taken using Bybit API. However, since bybit only allows small amounts requests at a time, I have included data requests functions in PriceAction Class which is present in data.py file. These functions allow you to download any amount of Cryptocurrency data since its listing on Bybit Platform (since 2017, Lot of data).

In this case, I have chosen to use only hourly data because it was faster to download and the dataset is very large.

## Predictors not used
When the predictors are not used, the precision is around 50%, which is basically similar to gambling.

![image](https://github.com/Gregos5/Market-Backtesting-Prediction/assets/78451671/ee87ff6b-23d8-4d95-9c29-396c959f709e)

## All Predictors used
When all of them are used , the precision reaches 60% and above which means the model can generate profit.

![image](https://github.com/Gregos5/Market-Backtesting-Prediction/assets/78451671/b06a055a-cdfc-4350-be86-bcc8d3a601eb)

