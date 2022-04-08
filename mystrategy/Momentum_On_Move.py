from datetime import datetime,timedelta
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams["figure.figsize"] = (10, 6) # (w, h)
plt.ioff()
import backtrader as bt
from scipy.stats import linregress
import math

class Momentum(bt.Indicator):
    lines = ('trend',)
    params = (('period', 90),)

    def __init__(self):
        self.addminperiod(self.params.period)

    def next(self):
        returns = np.log(self.data.get(size=self.p.period))
        x = np.arange(len(returns))
        slope, _, rvalue, _, _ = linregress(x, returns)
        #annualized = (1 + slope) ** 252
        #annualized = math.exp( slope * 365 ) - 1
        annualized = math.exp(slope * 12) - 1
        self.lines.trend[0] = annualized * (rvalue ** 2)


class Strategy(bt.Strategy):

    params = (
        ('stock_number', 10),  # limited to 10 stocks
        ('index_sma', 90),
        ('stock_sma', 30),
        ('momentum_window', 12),
        ('stock_atr', 20 ),
        ('rebal_weekday',[1,4,7]), # five is friday
    )

    def __init__(self):
        self.inds = {}
        self.index = self.datas[0]
        self.stocks = self.datas[1:]

        self.index_sma = bt.indicators.SimpleMovingAverage(self.index, period=self.p.index_sma)
        for d in self.stocks:
            self.inds[d] = {}
            self.inds[d]["momentum"] = Momentum(d, period=self.p.momentum_window)
            self.inds[d]["sma"] = bt.indicators.SimpleMovingAverage(d, period=self.p.stock_sma)
            self.inds[d]["atr"] = bt.indicators.ATR(d, period=self.p.stock_atr)

        # self.times = 0
        # self.add_timer(
        #     when=bt.Timer.SESSION_START,
        #     #offset=timedelta(minutes=30),
        #     #repeat=timedelta(days=3),
        #     weekdays=self.p.rebal_weekday,
        #     #weekcarry=True,  # if a day isn't there, execute on the next
        # )

    def prenext(self):
        # call next() even when data is not available for all tickers
        if not pd.isna(self.index_sma[0]): # index has readings
            print(
                f"PRENEXT-DATA0: {self.data.datetime.datetime(0):%Y-%m-%d %H:%M:%S}, INDEX-SMA: {self.index_sma[0]}, STRATEGY-LEN: {len(self)}")
            self.next()

    def nextstart(self):
        # call next() here all data is available
        if not self.index_sma[0]: # index has readings
            print(f"NEXTSTART-DATA0: {self.data.datetime.datetime(0):%Y-%m-%d %H:%M:%S}, INDEX-SMA: {self.index_sma[0]}, STRATEGY-LEN: {len(self)}")
            self.next()

    def next(self):
        print(f"NEXT-DATA0: {self.data.datetime.datetime(0):%Y-%m-%d %H:%M:%S}, INDEX-SMA: {self.index_sma[0]}, STRATEGY-LEN: {len(self)}")

        #if self.data.datetime.datetime(0) > datetime.fromisoformat('2021-04-16'):
        #if True:
        l = len(self)
        if l % 3 == 0:
            # 1. check close
            self.close_position()

            # 2. rebalance
            if l % 6 == 0:
                self.rebalance_positions()

            # 3. buy new if cash available
            self.open_new_positions()


    # def notify_timer(self, timer, when, *args, **kwargs):
    #
    #      print(
    #             f"DATA0: {self.data.datetime.datetime(0):%Y-%m-%d %H:%M:%S}, INDEX-SMA: {self.index_sma[0]}, STRATEGY-LEN: {len(self)}")


        # if not pd.isna(self.index_sma[0]):  # index has readings
        #     l = len(self)
        #     if (l+3) % 6 == 0:
        #         self.rebalance_portfolio()
        #     if l % 6 == 0:
        #         self.rebalance_positions()

        # if not pd.isna(self.index_sma[0]): # index has readings
        #     print(
        #         f"DATA0: {self.data.datetime.datetime(0):%Y-%m-%d %H:%M:%S}, INDEX-SMA: {self.index_sma[0]}, STRATEGY-LEN: {len(self)}")
        #     if self.times % 2 == 0:
        #         self.rebalance_portfolio()
        #     else:
        #         self.rebalance_positions()
        #     self.times += 1

    def close_position(self):

        # only look at data that we can have indicators for
        self.rankings = list(filter(lambda d: (len(d) > self.p.stock_sma), self.stocks))
        self.rankings.sort(key=lambda d: self.inds[d]["momentum"][0], reverse=True)

        num_stocks = len(self.rankings)
        num_stocks_candidate = max(int(num_stocks * 0.2), self.p.stock_number)

        # sell stocks based on criteria
        for i, d in enumerate(self.rankings):
            if self.getposition(d).size:
                # if i >= num_stocks_candidate or d < self.inds[d]["sma"] :
                if i >= num_stocks_candidate or d < self.inds[d]["sma"] or self.inds[d]["momentum"][0] < 0:
                    self.close(d)

    def open_new_positions(self):

        num_stocks = len(self.rankings)
        num_stocks_candidate = max(int(num_stocks * 0.2), self.p.stock_number)
        print(f"Num Of Stock Candidate: {num_stocks_candidate}")

        if self.index < self.index_sma:
            return

        # buy stocks with remaining cash
        for i, d in enumerate(self.rankings[:num_stocks_candidate]):
            cash = self.broker.get_cash()
            value = self.broker.get_value()
            if cash <= 0:
                break

            if self.inds[d]['momentum'] > 0 and (not self.getposition(d).size):
                size = value * 0.01 / self.inds[d]["atr"]
                self.buy(d, size=size)

        # print positions after
        print(f"REBALANCE-PORTFOLIO DATE: {self.data.datetime.datetime(0):%Y-%m-%d %H:%M:%S} ",
              f"POSITIONS {[ d._name for d in self.stocks if self.getposition(d).size > 0 ]} ",
              f"MOMENTUM {[ self.inds[d]['momentum'][0] for d in self.stocks if self.getposition(d).size > 0  ]}",
              f"SIZE {[self.getposition(d).size for d in self.stocks if self.getposition(d).size>0]} ",
              f"PRICE {[self.getposition(d).price for d in self.stocks if self.getposition(d).size>0]}"
              )

    def rebalance_positions(self):

        num_stocks = len(self.rankings)
        num_stocks_candidate = max(int(num_stocks * 0.2), self.p.stock_number)
        print(f"Num Of Stock Candidate: {num_stocks_candidate}")

        # rebalance all stocks
        for i, d in enumerate(self.rankings[:num_stocks_candidate]):
            cash = self.broker.get_cash()
            value = self.broker.get_value()
            if cash <= 0:
                break
            if self.getposition(d).size:
                size = value * 0.01 / self.inds[d]["atr"]
                if abs(self.getposition(d).size - size)/self.getposition(d).size > 1e-2: # change larger than 1%
                    self.order_target_size(d, size)

        print(f"REBALANCE-POSITIONS DATE: {self.data.datetime.datetime(0):%Y-%m-%d %H:%M:%S} ",
              f"POSITIONS {[d._name for d in self.stocks if self.getposition(d).size > 0]} ",
              f"MOMENTUM {[self.inds[d]['momentum'][0] for d in self.stocks if self.getposition(d).size > 0]}",
              f"SIZE {[self.getposition(d).size for d in self.stocks if self.getposition(d).size > 0]} ",
              f"PRICE {[self.getposition(d).price for d in self.stocks if self.getposition(d).size > 0]}"
              )

if __name__ == '__main__':

    # Get Tickers
    data_path ='../../algo-trading/My_Stuff/notebook/Binance_Index_Momentum/data'
    tickers = pd.read_csv(data_path +'/tickers.csv', header=None)[1].tolist()

    cerebro = bt.Cerebro(stdstats=False)
    #cerebro.broker.set_coc(True)

    # spy = bt.feeds.YahooFinanceData(dataname='^CMC200',
    #                                 fromdate=datetime(2020, 1, 1),
    #                                 todate=datetime(2022, 3, 30),
    #                                 timeframe=bt.TimeFrame.Days,
    #                                 plot=False)
    # cerebro.adddata(spy)  # add S&P 500 Index

    # add btc as data0, serve as index
    bitcoin = 'BTC'
    df = pd.read_csv(f"{data_path}/{bitcoin}.csv",
                     parse_dates=True,
                     index_col=0)
    if len(df) > 100:  # data must be long enough to compute 100 day SMA
        cerebro.adddata(bt.feeds.PandasData(dataname=df, name=bitcoin, plot=False))

    for ticker in tickers:
        df = pd.read_csv(f"{data_path}/{ticker}.csv",
                         parse_dates=True,
                         index_col=0)
        if len(df) > 100:  # data must be long enough to compute 100 day SMA
            cerebro.adddata(bt.feeds.PandasData(dataname=df, name=ticker, plot=False))

    cerebro.addobserver(bt.observers.Value)
    cerebro.addanalyzer(bt.analyzers.TimeReturn, timeframe=bt.TimeFrame.NoTimeFrame, _name='alltimereturn')
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, riskfreerate=0.0)
    cerebro.addanalyzer(bt.analyzers.Returns)
    cerebro.addanalyzer(bt.analyzers.DrawDown)
    # Add PyFolio, but this is quite problematic
    cerebro.addanalyzer(
        bt.analyzers.PyFolio,  # PyFlio only work with daily data
        timeframe=bt.TimeFrame.Days,
    )

    cerebro.addstrategy(Strategy)


    results = cerebro.run()

    pyfoliozer = results[0].analyzers.getbyname('pyfolio')
    returns, positions, transactions, gross_lev = pyfoliozer.get_pf_items()
    returns.to_csv("returns.csv")
    positions.to_csv("positions.csv")
    transactions.to_csv("transactions.csv")

    print(f"Return: {list(results[0].analyzers.alltimereturn.get_analysis().values())[0]:.3f}")
    print(f"Sharpe: {results[0].analyzers.sharperatio.get_analysis()['sharperatio']:.3f}")
    print(f"Norm. Annual Return: {results[0].analyzers.returns.get_analysis()['rnorm100']:.2f}%")
    print(f"Max Drawdown: {results[0].analyzers.drawdown.get_analysis()['max']['drawdown']:.2f}%")

    cerebro.plot()
