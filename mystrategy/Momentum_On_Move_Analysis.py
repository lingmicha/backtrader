

import pyfolio as pf
from scipy.stats import linregress
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas_ta as ta
import warnings
import mplfinance as mpf



warnings.filterwarnings('ignore')


def momentum(closes):
    returns = np.log(closes)
    x = np.arange(len(returns))
    slope, _, rvalue, _, _ = linregress(x, returns)
    annualized = math.exp(slope) - 1
    return annualized * (rvalue ** 2)  # annualize slope and multiply by R^2


def slope(closes):
    returns = np.log(closes)
    x = np.arange(len(returns))
    slope, _, rvalue, _, _ = linregress(x, returns)
    annualized = math.exp(slope) - 1
    return annualized  # annualize slope and multiply by R^2


def r_square(closes):
    returns = np.log(closes)
    x = np.arange(len(returns))
    slope, _, rvalue, _, _ = linregress(x, returns)
    return (rvalue ** 2)  # annualize slope and multiply by R^2


def atr20(x):
    return ta.atr(x.high, x.low, x.close, length=20)


def sma30(x):
    return ta.sma(x, length=30)


class StrategyAnalysis():

    __instance = None

    def __init__(self):
        """ Virtually private constructor. """
        if StrategyAnalysis.__instance != None:
                raise Exception("This class is a singleton!")
        else:
            StrategyAnalysis.__instance = self

            self.coins = None
            self.tickers = None
            self.tickers_file = None

            self.transactions = None
            self.transaction_file = None

            self.positions = None
            self.position_file = None

            self.returns = None
            self.return_file = None

    @staticmethod
    def getInstance():
        if StrategyAnalysis.__instance is None:
            StrategyAnalysis()

        return StrategyAnalysis.__instance

    def process_return_file(self, path, filename):
        self.return_file = filename
        self.returns = pd.read_csv(f"{path}/{self.return_file}",
                                   parse_dates=True,
                                   index_col='index').squeeze()
        return self.returns

    def process_position_file(self, path, filename):
        self.position_file = filename
        self.positions = pd.read_csv(f"{path}/{self.position_file}",
                                     parse_dates=True,
                                     index_col='Datetime')
        return self.positions

    def process_transaction_file(self, path, filename):
        self.transaction_file = filename
        self.transactions = pd.read_csv(f"{path}/{self.transaction_file}",
                                        parse_dates=True,
                                        index_col='date')
        return self.transactions

    def process_tickers(self, path, filename):
        self.tickers_file = filename
        raw_tickers = pd.read_csv(f"{path}/{filename}", header=None)[1].tolist()

        # filter tickers
        filtered = []
        for ticker in raw_tickers:
            df = pd.read_csv(f"{path}/{ticker}.csv",
                             parse_dates=True,
                             index_col=0)
            if len(df) > 100:  # data must be long enough to compute 100 day SMA
                filtered.append(ticker)

        self.tickers = sorted(filtered)

        # load all coins bars
        self.coins = (
            (pd.concat(
                [pd.read_csv(f"{path}/{ticker}.csv", index_col='date', parse_dates=True)
                 for ticker in self.tickers],
                axis=1, keys=self.tickers,
                sort=True)
            )
        )
        self.coins = self.coins.loc[:, ~self.coins.columns.duplicated()]
        return self.coins

    def process_indicators(self):
        for ticker in self.tickers:
            self.coins.loc[:, (ticker, 'momentum')] = self.coins[ticker]['close'].rolling(12).apply(momentum, raw=False)
            self.coins.loc[:, (ticker, 'slope')] = self.coins[ticker]['close'].rolling(12).apply(slope, raw=False)
            self.coins.loc[:, (ticker, 'rsquare')] = self.coins[ticker]['close'].rolling(12).apply(r_square, raw=False)
            self.coins.loc[:, (ticker, 'sma30')] = sma30(self.coins[ticker]['close'])
            self.coins.loc[:, (ticker, 'atr20')] = atr20(self.coins[ticker])
            self.coins.loc[:, (ticker, 'sizing')] = 0.1 / self.coins.loc[:, (ticker, 'atr20')]
            self.coins.loc[:, (ticker, 'position')] = self.positions[ticker]
            self.coins.loc[:, (ticker, 'txn_amount')] = self.transactions.loc[self.transactions['symbol'] == ticker][
                'amount']
            self.coins.loc[:, (ticker, 'txn_value')] = self.transactions.loc[self.transactions['symbol'] == ticker][
                'value']
            self.coins.loc[:, (ticker, 'txn_price')] = self.transactions.loc[self.transactions['symbol'] == ticker][
                'price']

        self.coins.sort_index(axis=1, inplace=True)
        return self.coins

    def get_tickers(self):
        return self.tickers

    def get_returns(self):
        return self.returns

    def get_positions(self):
        return self.positions

    def get_transactions(self):
        return self.transactions


    def print_daily_stats(self, date):
        columns = ['ticker', 'close','momentum', 'sma30', 'atr20', 'position']
        daily_table = pd.DataFrame(index=[], columns=columns)

        for ticker in self.tickers:
            s = self.coins.loc[date, ticker].filter(columns)
            s["ticker"] = ticker
            daily_table = daily_table.append(s)

        daily_table.reset_index(inplace=True)
        #daily_table.drop("index")
        daily_table.set_index('ticker', inplace=True)


        with pd.option_context('display.max_rows', 50,
                               'display.max_columns', 12,
                               'display.precision', 3,
                               ):
            print(f"Momentum & Position Stats for Date {date}")
            print(daily_table.sort_values(by=['momentum', 'position'], ascending=False).head(20))


    def print_daily_txn_stats(self, date):
        columns = ['ticker', 'momentum', 'position','txn_value','txn_price','txn_amount' ]
        daily_table = pd.DataFrame(index=[], columns=columns)

        for ticker in self.tickers:
            s = self.coins.loc[date, ticker].filter(columns)
            s["ticker"] = ticker
            daily_table = daily_table.append(s)

        daily_table.reset_index(inplace=True)
        #daily_table.drop("index")
        daily_table.set_index('ticker', inplace=True)


        with pd.option_context('display.max_rows', 50,
                               'display.max_columns', 12,
                               'display.precision', 3,
                               ):
            print(f"Transaction Stats for Date {date}")
            print(daily_table.sort_values(by=['txn_value', 'position'], ascending=False).head(20))

    def print_ticker_daily_stats(self, ticker, start=None, end=None):

        coin = self.coins.loc[:, ticker]

        if start is None:
            start = coin.index[0]
        if end is None:
            end = coin.index[-1]

        coin = coin.loc[start:end]

        columns = ['close','momentum', 'sma30', 'atr20',
                   'position', 'txn_value','txn_price','txn_amount']
        coin = coin.filter(columns)
        coin = coin[coin['position'] > 0 | ~ coin['txn_amount'].isna()]

        with pd.option_context('display.max_rows', 1000,
                               'display.max_columns', 12,
                               'display.precision', 3,
                               ):
            print(f"{ticker} Daily Stats from {start} to {end}")
            print(coin.sort_values(by=['momentum', 'position'], ascending=False).head(20))

    def plot_ticker(self, ticker, start=None, end=None ):
        coin = self.coins.loc[:, ticker]

        if start is None:
            start = coin.index[0]
        if end is None:
            end = coin.index[-1]

        coin = coin.loc[start:end]

        coin['buy'] = coin.apply(
            lambda x: x.low * 0.99 if x.txn_amount > 0 else np.nan,
            axis=1,
        )

        coin['sell'] = coin.apply(
            lambda x: x.high * 1.01 if x.txn_amount < 0 else np.nan,
            axis=1,
        )

        apdict = [
            mpf.make_addplot(coin['buy'][start:end], type='scatter', markersize=75, marker='^'),
            mpf.make_addplot(coin['sell'][start:end], type='scatter', markersize=75, marker='v'),
            mpf.make_addplot(coin['sma30'][start:end]),
            mpf.make_addplot(coin['momentum'][start:end], panel=1, ylabel='momentum'),
            mpf.make_addplot(coin['slope'][start:end], panel=2, ylabel='slope'),
            mpf.make_addplot(coin['rsquare'][start:end], panel=3, ylabel='Q^2'),

        ]

        mpf.plot(coin[start:end], style='yahoo', type='candle', ylabel='Price', volume=False, addplot=apdict,
                 figscale=1.5)


if __name__ == "__main__":

    result_path = '/Users/michael/eProjects/algo-trading/My_Stuff/notebook/Binance_Index_Momentum/result'
    data_path = '/Users/michael/eProjects/algo-trading/My_Stuff/notebook/Binance_Index_Momentum/data'
    analysis = StrategyAnalysis.getInstance()
    analysis.process_return_file(result_path, 'E8-returns.csv')
    analysis.process_position_file(result_path, 'E8-positions.csv')
    analysis.process_transaction_file(result_path, 'E8-transactions.csv')
    analysis.process_tickers(data_path, 'tickers.csv')
    analysis.process_indicators()
    analysis.print_daily_stats('2021-12-01')
    analysis.print_daily_txn_stats('2021-12-01')
    analysis.plot_ticker('BTC')
    analysis.print_ticker_daily_stats('SOL')