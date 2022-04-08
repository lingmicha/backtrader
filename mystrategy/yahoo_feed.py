from datetime import datetime
import backtrader as bt

class Strategy(bt.Strategy):

    def __init__(self):
        self.sma200 = bt.indicators.SMA(self.data, period=200)

    def prenext(self):
        self.next()

    def nextstart(self):
        self.next()

    def next(self):
        print( f"DATA0:{len(self.data)}, {self.data.datetime.datetime(0):%Y-%m-%d %H:%M:%S} ",
               f"Index: {self.data[0]}, sma200: {self.sma200[0]}")

if __name__ == '__main__':
    # Get Tickers
    cerebro = bt.Cerebro()
    spy = bt.feeds.YahooFinanceData(dataname='^CMC200',
                                    fromdate=datetime(2020, 1, 1),
                                    todate=datetime(2022, 3, 30),
                                    #timeframe=bt.TimeFrame.Days,
                                    )
    cerebro.adddata(spy)  # add S&P 500 Index

    cerebro.addstrategy(Strategy)
    results = cerebro.run()
    cerebro.plot()
