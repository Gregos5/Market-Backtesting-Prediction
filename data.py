import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, confusion_matrix
import calendar, os
import matplotlib.pyplot as plt
from pybit import inverse_perpetual
import plotly.graph_objects as go

class PriceAction():
    def __init__(self, session = inverse_perpetual.HTTP(endpoint="https://api.bybit.com"), TF = '1H', ticker='BTCUSD'):
        '''
        TF = 'D | '1H' | '30min' | '15min' | '1min'
        R_since : Download price, volume
        save_data : into csv
        plot_ohlc : plot candlestick
        '''
        self.session= session
        self.ticker = ticker
        self.TF = TF
        self.TF2min()
        self.path = os.path.dirname(os.path.realpath(__file__))
        self.df = pd.DataFrame()
        self.figs = []
         
    def TF2min(self):
        
        match self.TF:
            case '1min':
                ticks = 1
            case '5min':
                ticks = 5
            case '15min':
                ticks = 15
            case '30min':
                ticks = 30
            case '1H':
                ticks = 60
            case '2H':
                ticks = 120
            case _:
                ticks = 60
        self.tick_interval = ticks
        return ticks 
    
    def load_data(self):
        """
        Load BTC price
        """
        try:
            self.df = pd.read_csv(os.path.join(self.path, (self.ticker+'_'+self.TF)), index_col='open_time')
            return self.df
        except FileNotFoundError:
            print('file {} not found'.format((self.ticker+'_'+self.TF)))

    def update_data(self):
        old_df = self.load_data()
        last_open_time = old_df.index[-1]
        add_df = self.R_since_unix(self, last_open_time)
        self.df = pd.concat([self.df, add_df])
        self.save_data()

    def save_data(self):
        """ create/save data in csv """
        self.df.to_csv(os.path.join(self.path, (self.ticker+'_'+self.TF)), encoding='utf-8')
    
    def R_day(self, day_look_back=1):
        now = datetime.utcnow()
        midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
        unixtime_pday_end = calendar.timegm(midnight.utctimetuple())

        since = unixtime_pday_end - 60*60*24*day_look_back
        # bybit only sends you 200 candles at a time
        # if resolution = 1min; 24*60/200 = 7.2 steps = 7 full steps + 0.2 of a step 
        steps = 24*60/(200*self.tick_interval)
        step_int = int(np.floor(steps))
        step_fl = steps-step_int
        D_sep = [pd.DataFrame()]*(step_int+1)
        # i full steps 
        for i in range(step_int):
            try:
                response = self.session.query_kline(symbol= self.ticker, interval= self.tick_interval, **{'from': str(since+i*200*60*self.tick_interval)})
                index = np.arange(0+200*i,200+200*i)
                D_sep[i] = pd.DataFrame(response['result']).set_index(index)
            except:
                pass
        #0.2 of a step = 40 minutes left
        limit= round(step_fl*200) 
        response = self.session.query_kline(symbol= self.ticker, interval= int(self.tick_interval), **{'from': str(since+(step_int)*200*self.tick_interval*60)}, limit=limit)
        index = np.arange(step_int*200,step_int*200+limit)
        D_sep[step_int] = pd.DataFrame(response['result']).set_index(index)

        #concatenate segmented df
        self.df = pd.concat(D_sep)
        self.rename_data()
        return self.df

    def R_today(self, tick_interval=1):
        now = datetime.utcnow()
        midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
        since = calendar.timegm(midnight.utctimetuple())
        until = calendar.timegm(now.utctimetuple())

        step = 200*60*tick_interval
        steps = (until-since)/step
        
        step_int = int(np.floor(steps))
        step_fl = steps-step_int

        D_sep = [pd.DataFrame()]*(step_int+1)
        # i full steps 
        for i in range(step_int):
            response = self.session.query_kline(symbol= self.ticker, interval= int(tick_interval), **{'from': str(since+i*step)})
            D_sep[i] = pd.DataFrame(response['result'])
        # fracional step
        limit= round(step_fl*200)
        response = self.session.query_kline(symbol= self.ticker, interval= int(tick_interval), **{'from': str(since+(step_int)*step)}, limit=limit)
        D_sep[step_int] = pd.DataFrame(response['result'])

        #concatenate segmented df
        self.df = pd.concat(D_sep)
        self.rename_data()
        return self.df

    def R_since(self, day_look_back=1):

        now = datetime.utcnow()
        midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
        unixtime_pday_end = calendar.timegm(midnight.utctimetuple())

        since = unixtime_pday_end - 60*60*24*day_look_back
        until = calendar.timegm(now.utctimetuple())

        # bybit only sends you 200 candles at a time
        step = 200*60*self.tick_interval
        steps = (until - since)/(step)
        step_int = int(np.floor(steps))
        step_fl = steps-step_int

        D_sep = [pd.DataFrame()]*(step_int+1)
        # i full steps 
        print(step_int)
        for i in range(step_int):
            response = self.session.query_kline(symbol= self.ticker, interval= int(self.tick_interval), **{'from': str(since+i*step)})
            D_sep[i] = pd.DataFrame(response['result'])
            print(i*100/step_int)
        # fracional step
        limit= round(step_fl*200)
        response = self.session.query_kline(symbol= self.ticker, interval= int(self.tick_interval), **{'from': str(since+(step_int)*step)}, limit=limit)
        D_sep[step_int] = pd.DataFrame(response['result'])

        #concatenate segmented df
        self.df = pd.concat(D_sep)
        # print(self.df.tail())
        self.rename_data()
        return self.df

    def R_since_unix(self, since):

        now = datetime.utcnow()
        until = calendar.timegm(now.utctimetuple())

        # bybit only sends you 200 candles at a time
        step = 200*60*self.tick_interval
        steps = (until - since)/(step)
        step_int = int(np.floor(steps))
        step_fl = steps-step_int

        D_sep = [pd.DataFrame()]*(step_int+1)
        # i full steps 
        print(step_int)
        for i in range(step_int):
            response = self.session.query_kline(symbol= self.ticker, interval= int(self.tick_interval), **{'from': str(since+i*step)})
            D_sep[i] = pd.DataFrame(response['result'])
            print(i*100/step_int)
        # fracional step
        limit= round(step_fl*200)
        response = self.session.query_kline(symbol= self.ticker, interval= int(self.tick_interval), **{'from': str(since+(step_int)*step)}, limit=limit)
        D_sep[step_int] = pd.DataFrame(response['result'])

        #concatenate segmented df
        self.df = pd.concat(D_sep)
        # print(self.df.tail())
        self.rename_data()
        return self.df

    def rename_data(self):
        self.df = self.df.set_index('open_time')
        self.df = self.df[['open', 'high', 'low', 'close', 'volume', 'open_interest']]
        self.df.rename(columns = { 'open':'Open' , 'high': 'High', 'low': 'Low', 'close':'Close', 'volume':'Volume', 'open_interest': 'OI'}, inplace=True)
        self.df = self.df.astype(float)
        return self.df
    
    def plot_ohlc(self):
        """ Change to matplotlib """
        df = self.df
        df['Date'] =  pd.to_datetime(df['open_time'], unit='s', utc=True)
        
        df = df.set_index('Date')
        
        self.figs.append(go.Candlestick(
            x=df.index,
            open=df.Open,
            high=df.High,
            low=df.Low,
            close=df.Close,
            xaxis='x',
            yaxis='y2',
            visible=True,
            showlegend=False
        ))
        
        layout = go.Layout(
            title=go.layout.Title(text="BTCUSD today"),
            xaxis=go.layout.XAxis(
                side="bottom",
                title="Date",
                rangeslider=go.layout.xaxis.Rangeslider(visible=False)
            ),
            yaxis=go.layout.YAxis(
            side="right",
            title='Volume',
            showticklabels=False,
            domain=[0, 0.2]
            ),
            yaxis2=go.layout.YAxis(
                side="right",
                title='Price',
                domain=[0.2, 1.0]
            ),
            xaxis2=go.layout.XAxis(
                side="top",
                showgrid=False,
                ticks='',
                showticklabels=False,
                range=[0, int(self.vol_bars.max() * 5)],
                overlaying="x"
            ),
            yaxis3=go.layout.YAxis(
                side="left",
                range=[self.cmin_int, self.cmax_int],
                showticklabels=False,
                overlaying="y2"
            ),
        )
        fig = go.Figure(data=[self.figs[1], self.figs[0]], layout=layout)
        fig.show()

    def value_area(self, price_pace=5, percent=0.68):
        value_areas = pd.DataFrame()

        self.df['Date'] =  pd.to_datetime(self.df['open_time'], unit='s', utc=True)
        days = int((len(self.df)-len(self.df)%1440)/1440)

        for day in range(days-12):
            df = self.df[1440*day:1440*(day+1)]
            df = df.reset_index()
            df['avg'] = df[['Open', 'Close']].mean(axis=1).astype(int)
            cmin = min(df.Close)
            cmax = max(df.Close)
            cmin_int = int(cmin / price_pace) * price_pace  # int(0.9) = 0
            cmax_int = int(cmax / price_pace) * price_pace

            if cmax_int < cmax:
                cmax_int += price_pace
            cmax_int += price_pace  # right bracket is not included in arrange

            cmin_int=np.ceil(cmin_int)
            cmax_int=np.floor(cmax_int)
            price_buckets = np.arange(cmin_int, cmax_int, price_pace)
            price_coors = pd.Series(price_buckets).rolling(2).mean().dropna()
            vol_bars = np.histogram(df.avg, bins=price_buckets, weights=df.Volume)[0]
            self.vol_bars = vol_bars
            #initialise
            indices = np.argsort(vol_bars)[::-1][1]
            # print('other max', indices)
            idx_poc = np.argmax(vol_bars)
            # print(max(vol_bars))
            # print(vol_bars[indices])
            # idx_poc = indices
            # print('poc index', idx_poc)

            idx_L = idx_poc-1
            idx_H = idx_poc+1
            total_volume = sum(vol_bars)
            value_area_vol = vol_bars[idx_poc]

            while value_area_vol < percent*total_volume:
                
                #Addition of 2 at top and bottom
                if idx_H+2<=len(vol_bars):
                    sum_top = vol_bars[idx_H]+vol_bars[idx_H+1]
                else:
                    sum_top = 0.0
                
                if idx_L-2>=0:
                    sum_bot = vol_bars[idx_L]+vol_bars[idx_L-1]
                else:
                    sum_bot=0.0
                
                #compare and update index of value area
                if sum_top>sum_bot:
                    idx_H+=2
                elif sum_top<sum_bot:
                    idx_L-=2
                else:
                    # skip tick if both are equal volume 
                    # not so rare if Vol = 0 for 2 consecutive prices
                    idx_H+=1
                    idx_L-=1

                idx_H = min(len(price_coors),idx_H)
                idx_L = max(0,idx_L)    
                
                #update total value area volume
                value_area_vol = sum(vol_bars[idx_L+1:idx_H])
            
            poc = price_coors[idx_poc]
            val = price_coors[idx_L+1]
            vah = price_coors[idx_H]
            dClose = df['Close'].iloc[-1]

        print('vah ', vah)
        print('poc ', poc)
        print('val ', val)

        
        fig1 = go.Candlestick(
        x=df.Date,
        open=df.Open,
        high=df.High,
        low=df.Low,
        close=df.Close,
        xaxis='x',
        yaxis='y',
        visible=True,
        showlegend=False
        )

        low = cmin_int
        high = cmax_int
        layout = go.Layout(
            title=go.layout.Title(text="Volume Profile"),
            xaxis=go.layout.XAxis(
                side="bottom",
                title="Date",
                rangeslider=go.layout.xaxis.Rangeslider(visible=False)
            ),
            yaxis=go.layout.YAxis(
                side="right",
                title='Price',
                range=[low, high],
                domain=[0.2, 1.0]
            ),
        )
    
        #price of max volume
        
        
        fig = go.Figure(data=[fig1], layout=layout)
        
        fig.add_hline(
            y=poc,
            line_width=2, line_color="red",
            row=1, col=1
        )
        fig.add_hline(
            y=val,
            line_width=2, line_color="red",
            row=1, col=3
        )
        fig.show()
        return fig

class Predictors():
    def __init__(self, btc1H):

        data = pd.DataFrame(btc1H[['Close']])
        data = data.rename(columns = {'Close':'Actual_Close'})

        # Setup our target.  This identifies if the price went up or down
        data["Target"] = btc1H.rolling(2).apply(lambda x: x.iloc[1] > x.iloc[0])["Close"]

        # Shift stock prices forward one day, so we're predicting tomorrow's stock prices from today's prices.
        btc_prev = btc1H.copy()
        btc_prev = btc_prev.shift(1)

        # Create our training data
        predictors = ["Close", "Volume", "Open", "High", "Low"]
        data = data.join(btc_prev[predictors]).iloc[1:]

        daily_mean = data.rolling(24).mean()
        weekly_mean = data.rolling(24*7).mean()
        weekly_trend = data.shift(1).rolling(24*7).mean()["Target"]

        # Setup our target.  This identifies if the price went up or down
        data["Target"] = btc1H.rolling(2).apply(lambda x: x.iloc[1] > x.iloc[0])["Close"]

        data["daily_mean"] = daily_mean["Close"] / data["Close"]
        data["weekly_mean"] = weekly_mean["Close"] / data["Close"]
        data["weekly_trend"] = weekly_trend

        data["open_close_ratio"] = data["Open"] / data["Close"]
        data["high_close_ratio"] = data["High"] / data["Close"]
        data["low_close_ratio"] = data["Low"] / data["Close"]

        data["EMA"] = data['Close'].ewm(span=50, adjust=False).mean()
        self.data = data
        self.RFC()
        self.predictors = ["Close", "Volume", "Open", "High", "Low", "open_close_ratio", "high_close_ratio", "low_close_ratio", "daily_mean", "weekly_mean", "weekly_trend", "EMA"]


    def RFC(self, n_estimators=100, min_samples_split=500):
        # Create a random forest classification model.  Set min_samples_split high to ensure we don't overfit.
        self.model = RandomForestClassifier(n_estimators=n_estimators, min_samples_split=min_samples_split, random_state=1)

    # backtest strategies
    def backtest(self, start=9000, step=3000, predictors = ["Close", "Volume", "Open", "High", "Low", "open_close_ratio", "high_close_ratio", "low_close_ratio", "daily_mean", "weekly_mean", "weekly_trend", "EMA"]):
        predictions = []
        self.predictors = predictors
        data = self.data.iloc[200:]
        # Loop over the dataset in increments
        for i in range(start, data.shape[0], step):
            # Split into train and test sets
            train = data.iloc[0:i].copy()
            test = data.iloc[i:(i+step)].copy()
            
            # Fit the random forest model
            self.model.fit(train[self.predictors], train["Target"])
            
            # Make predictions
            preds = self.model.predict_proba(test[self.predictors])[:,1]
            preds = pd.Series(preds, index=test.index)
            preds[preds > .6] = 1
            preds[preds<=.6] = 0
            
            # Combine predictions and test values
            combined = pd.concat({"Target": test["Target"],"Predictions": preds}, axis=1)
            
            predictions.append(combined)
        self.predicions = pd.concat(predictions)
        return  self.predicions

    def score(self, predictions):
        score = precision_score(predictions["Target"], predictions["Predictions"])
        
        tn, fp, fn, tp = confusion_matrix(predictions["Target"], predictions["Predictions"]).ravel()
        print('precision score', tp / (tp + fp))
        print('amount of tp', tp)
        print('amount of fp', fp)
        predictions["Predictions"].value_counts()
        predictions.iloc[-100:].plot()

        return score, tn, fp, fn, tp 

