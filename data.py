import pandas as pd
import numpy as np
from datetime import datetime
import requests
from time import time 
import calendar 
from pybit import inverse_perpetual
import plotly.graph_objects as go

class User():
    def __init__(self):
        ''' Load account '''
        self.name
        self.balance
        self.open_position
        self.records

class PriceAction():
    def __init__(self, session):
        ''' Load and update json file '''
        self.session= session
    
    def plot(self):
        ''' Matplotlib '''
    
    def get_single_day(self, day_look_back=1, tick_interval=1, ticker='BTCUSD'):
        now = datetime.utcnow()
        midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
        unixtime_pday_end = calendar.timegm(midnight.utctimetuple())

        since = unixtime_pday_end - 60*60*24*day_look_back
        # bybit only sends you 200 candles at a time
        # if resolution = 1min; 24*60/200 = 7.2 steps = 7 full steps + 0.2 of a step 
        steps = 24*60/(200*tick_interval)
        step_int = int(np.floor(steps))
        step_fl = steps-step_int
        D_sep = [pd.DataFrame()]*(step_int+1)
        # i full steps 
        for i in range(step_int):
            response = self.session.query_kline(symbol= ticker, interval= int(tick_interval), **{'from': str(since+i*200*60*tick_interval)})
            index = np.arange(0+200*i,200+200*i)
            D_sep[i] = pd.DataFrame(response['result']).set_index(index)
        #0.2 of a step = 40 minutes left
        limit= round(step_fl*200)
        response = self.session.query_kline(symbol= ticker, interval= int(tick_interval), **{'from': str(since+(step_int)*200*tick_interval*60)}, limit=limit)
        index = np.arange(step_int*200,step_int*200+limit)
        D_sep[step_int] = pd.DataFrame(response['result']).set_index(index)

        #concatenate segmented df
        D_tot = pd.concat(D_sep)
        D_tot = D_tot[['open', 'high', 'low', 'close', 'volume']]
        D_tot.rename(columns = { 'open':'Open' , 'high': 'High', 'low': 'Low', 'close':'Close', 'volume':'Volume'}, inplace=True)
        D_tot = D_tot.astype(float)
        return D_tot

    def get_data_since(self, day_look_back=1, tick_interval=1, ticker = 'BTCUSD'):
        now = datetime.utcnow()
        midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
        unixtime_pday_end = calendar.timegm(midnight.utctimetuple())

        since = unixtime_pday_end - 60*60*24*day_look_back
        # bybit only sends you 200 candles at a time
        step = 200*60*tick_interval
        steps = day_look_back*24*60*60/(step)
        step_int = int(np.floor(steps))
        step_fl = steps-step_int

        D_sep = [pd.DataFrame()]*(step_int+1)
        # i full steps 
        for i in range(step_int):
            response = self.session.query_kline(symbol= ticker, interval= int(tick_interval), **{'from': str(since+i*step)})
            D_sep[i] = pd.DataFrame(response['result'])
        # fracional step
        limit= round(step_fl*200)
        response = self.session.query_kline(symbol= ticker, interval= int(tick_interval), **{'from': str(since+(step_int)*step)}, limit=limit)
        D_sep[step_int] = pd.DataFrame(response['result'])

        #concatenate segmented df
        D_tot = pd.concat(D_sep).set_index('open_time')
        D_tot = D_tot[['open', 'high', 'low', 'close', 'volume']]
        D_tot.rename(columns = { 'open':'Open' , 'high': 'High', 'low': 'Low', 'close':'Close', 'volume':'Volume'}, inplace=True)
        D_tot = D_tot.astype(float)
        return D_tot

    def get_current_day(self, tick_interval=1, ticker='BTCUSD'):
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
            response = self.session.query_kline(symbol= ticker, interval= int(tick_interval), **{'from': str(since+i*step)})
            D_sep[i] = pd.DataFrame(response['result'])
        # fracional step
        limit= round(step_fl*200)
        response = self.session.query_kline(symbol= ticker, interval= int(tick_interval), **{'from': str(since+(step_int)*step)}, limit=limit)
        D_sep[step_int] = pd.DataFrame(response['result'])

        #concatenate segmented df
        D_tot = pd.concat(D_sep)
        D_tot = self.clean_data(D_tot)
        return D_tot

    def clean_data(self, df):
        df = df.set_index('open_time')
        df = df[['open', 'high', 'low', 'close', 'volume']]
        df.rename(columns = { 'open':'Open' , 'high': 'High', 'low': 'Low', 'close':'Close', 'volume':'Volume'}, inplace=True)
        df = df.astype(float)
        return df
    
    def plot_ticks(self, df):
        
        df['date'] =  pd.to_datetime(df.index, unit='s', utc=True)
        
        df = df.set_index('date')
        fig1 = go.Candlestick(
            x=df.index,
            open=df.Open,
            high=df.High,
            low=df.Low,
            close=df.Close,
            xaxis='x',
            yaxis='y',
            visible=True,
            showlegend=False
        )
        layout = go.Layout(
            title=go.layout.Title(text="BTCUSD today"),
            xaxis=go.layout.XAxis(
                side="bottom",
                title="Date",
                rangeslider=go.layout.xaxis.Rangeslider(visible=False)
            ),
            yaxis2=go.layout.YAxis(
                side="right",
                title='Price',
                domain=[0.2, 1.0]
            )
        )
        fig = go.Figure(data=[fig1], layout=layout)
        return fig

    def value_area(df, price_pace=1, percent=0.68):
        df = df.reset_index()
        df['avg'] = df[['Open', 'Close']].mean(axis=1).astype(int)
        
        cmin = min(df.Close)
        cmax = max(df.Close)
        cmin_int = int(cmin / price_pace) * price_pace  # int(0.9) = 0
        cmax_int = int(cmax / price_pace) * price_pace
        if cmax_int < cmax:
            cmax_int += price_pace
        cmax_int += price_pace  # right bracket is not included in arrange

        price_buckets = np.arange(cmin_int, cmax_int, price_pace)
        price_coors = pd.Series(price_buckets).rolling(2).mean().dropna()
        vol_bars = np.histogram(df.avg, bins=price_buckets, weights=df.Volume)[0]

        #initialise
        indices = np.argsort(vol_bars)[::-1][1]
        print('other max', indices)
        idx_poc = np.argmax(vol_bars)
        print(max(vol_bars))
        print(vol_bars[indices])
        idx_poc = indices
        print('poc index', idx_poc)

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

            idx_H = min(len(df),idx_H)
            idx_L = max(0,idx_L)    
                
            #update total value area volume
            value_area_vol = sum(vol_bars[idx_L+1:idx_H])

        print('--------- DONE -----------')
        poc = price_coors[idx_poc]
        val = price_coors[idx_L+1]
        vah = price_coors[idx_H]
        dClose = df['Close'].iloc[-1]

        print('pdClose', dClose)
        print('vah ', vah)
        print('poc ', poc)
        print('val ', val)

        return poc, val, vah, dClose

public_session = inverse_perpetual.HTTP(endpoint="https://api.bybit.com")
P = PriceAction(public_session)
D = P.get_single_day(day_look_back=3,  tick_interval=1)
D = P.get_current_day(tick_interval=1)
print(D.tail())

# poc,val,vah, dOpen = value_area(D)
fig = P.plot_ticks(D)

# fig.add_hline(y=poc,
#               annotation_text= "POC", 
#               annotation_position="bottom right",
#               line_color='red')
# for level in [val,vah]:
#     fig.add_hline(y=level, line_color='blue')
fig.show()

