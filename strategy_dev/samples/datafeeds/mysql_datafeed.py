import json

import backtrader as bt
from datetime import datetime

class MyStrategy(bt.Strategy):

    def __init__(self):
        return

    def next(self):
        print(f"NEXT-DATA0: {self.data.datetime.datetime(0):%Y-%m-%d %H:%M:%S},  "
              f"Ticks: {self.data.open[0], self.data.high[0], self.data.low[0], self.data.close[0], self.data.volume[0]} "
              f"STRATEGY-LEN: {len(self)}")

def run_backtest():

    # absolute dir the script is in
    config_file = f"../../config/params-production-future.json"
    with open(config_file, 'r') as f:
        params = json.load(f)

    cerebro = bt.Cerebro(stdstats=False, preload=False, runonce=False)

    ticker = "BTC/USDT"
    data = bt.feeds.MySQLData(name="BTC",
                              symbol="BTC/USDT",
                              fromdate=datetime(2021, 1, 1),
                              todate=datetime(2021, 12, 31),
                              dbHost = params["database"]["host"],
                              dbUser = params["database"]["user"],
                              dbPWD = params["database"]["pass"],
                              dbName = params["database"]["name"],
                              plot=False,
                              timeframe=bt.TimeFrame.Days)

    cerebro.adddata(data)
    cerebro.addstrategy(MyStrategy)
    cerebro.run()

if __name__ == '__main__':
    run_backtest()