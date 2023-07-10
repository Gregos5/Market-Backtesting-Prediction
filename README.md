# Market prediction with Machine learning modelling and backtesting

## Overview and Motivation
The initial aim of this project was to achieve a cryptocurrency prediction model with a precision above 60%.

The goal is to provide a user interface on which the user can select the predictors and optimise backtesting parameters to generate a cryptocurrency market model (random forest classifiers) (BTC_USD) to predict the direction of the next hourly candle using several predictors and backtesting on a very large amount of data (BTC 1H)

## Data 
All data is taken using Bybit API. However, since bybit only allows small amounts requests at a time, I have included data requests functions in PriceAction Class which is present in data.py file. These functions allow you to download any amount of Cryptocurrency data since its listing on Bybit Platform (since 2017, Lot of data).

In this case, I have chosen to use only hourly data because it was faster to download and the dataset is very large.

## Predictors not used
When the predictors are not used, the precision is around 50%, which is basically similar to gambling.

![Screenshot 2023-07-10 183815](https://github.com/Gregos5/Market-Backtesting-Prediction/assets/78451671/b020bf55-5dc4-4b0a-9b6c-b4c0fd7dc814)

## All Predictors used
When all of them are used , the precision reaches 60% and above which means the model can generate profit.

![Screenshot 2023-07-10 183815](https://github.com/Gregos5/Market-Backtesting-Prediction/assets/78451671/5a46bee1-5c79-4c03-8a06-c427b146ef0b)
