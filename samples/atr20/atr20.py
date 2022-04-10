import backtrader as bt
import pandas as pd


class atr_start(bt.Strategy):

    def __init__(self):
        self.atr20 = bt.ind.ATR(period=20, movav=bt.ind.MovAv.Smoothed)

    def next(self):
        print(f"Date: {self.data.datetime.datetime(0):%Y-%m-%d} ",
              f"ATR20: {self.atr20.atr[0]}")

    def stop(self):
        # Dump indicator, take care: exactbars setting cause it may only store values within time windows
        atr = self.atr20.get(size=len(self.atr20))
        dt = map(bt.utils.num2date, self.data.datetime.get(size=len(self.data.datetime)))

        df = pd.DataFrame(list(zip(dt, atr)), columns=['datetime', 'atr20'])
        df.set_index('datetime', inplace=True)
        df.to_csv("atr20.csv")


if __name__ == '__main__':

    cerebro = bt.Cerebro(stdstats=False)

    df = pd.read_csv('2017-2022-BNB.csv', parse_dates=True, index_col=0)
    data = bt.feeds.PandasData(dataname=df, name='BNB', timeframe=bt.TimeFrame.Days)

    cerebro.adddata(data)
    cerebro.addstrategy(atr_start)

    cerebro.run()

    # END