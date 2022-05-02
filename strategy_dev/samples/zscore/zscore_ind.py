import backtrader as bt
import numpy as np
import scipy.stats as stats
import pandas as pd

'''
    Test ZScore indicator
'''

class ZScore(bt.ind.BaseApplyN):
    lines = ('zscore',)
    params = (('period', 20),
              ('func', lambda x: stats.zscore(x)[-1]),
              )

class ZScoreTest(bt.Strategy):

    def __init__(self):
        self.zscore = ZScore(self.datas[0], period=20)
        self.df = pd.read_csv('2017-2022-BNB.csv', parse_dates=True, index_col=0)

    def next(self):

        l = self.df['close'].to_list()
        d = stats.zscore(l[len(self)-20:len(self)])[-1]
        print(f"DATE: {self.datetime.datetime(0)} "
              f"Zscore: {self.zscore[0]} " 
              f"Pandas: {d}")

if __name__ == "__main__":

    cerebro = bt.Cerebro(stdstats=False)

    df = pd.read_csv('2017-2022-BNB.csv', parse_dates=True, index_col=0)
    data = bt.feeds.PandasData(dataname=df, name='BNB', timeframe=bt.TimeFrame.Days)
    cerebro.adddata(data)
    cerebro.addstrategy(ZScoreTest)

    cerebro.run()




