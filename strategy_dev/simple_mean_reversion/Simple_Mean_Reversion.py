import backtrader as bt
import numpy as np
import pandas as pd
import os
import json
import time
from ccxtbt import CCXTStore
from datetime import datetime


class MeanReversion(bt.Strategy):

    def prenext(self):
        self.next()

    def nextstart(self):
        self.next()

    def next(self):
        # only look at data that existed yesterday
        available = list(filter(lambda d: len(d)>2, self.datas))

        rets = np.zeros(len(available))
        for i, d in enumerate(available):
            # calculate individual daily returns
            rets[i] = (d.close[0] - d.close[-1]) / d.close[-1]

        # calculate weights using formula
        market_ret = np.mean(rets)
        weights = -(rets - market_ret)

        weights = weights / np.sum(np.abs(weights))

        for i, d in enumerate(available):
            self.order_target_percent(d, target=weights[i])

def backtest():

    cerebro = bt.Cerebro(stdstats=False)
    cerebro.broker.set_coc(True)

    # read ticker file
    tickers_file = 'tickers_20200101.csv'
    tickers = pd.read_csv(f"data/{tickers_file}", header=None)[1].to_list()
    print(f"TOTAL {len(tickers)} CRYPTOS ON BINANCE BEFORE 2020-01-01")

    # absolute dir the script is in
    script_dir = os.path.dirname(__file__)
    abs_file_path = os.path.join(script_dir, '../config/params-production-spot.json')
    with open(abs_file_path, 'r') as f:
        params = json.load(f)

    # Create our store
    config = {'apiKey': params["binance"]["apikey"],
              'secret': params["binance"]["secret"],
              'enableRateLimit': True,
              'nonce': lambda: str(int(time.time() * 1000)),
              }

    store = CCXTStore(exchange='binance', currency='USDT', config=config, retries=5, debug=False, sandbox=False)

    # TODO: DATA0 Must have the earilest start datetime
    fromdate = datetime.strptime('2020-01-01', '%Y-%m-%d')
    todate = datetime.strptime('2022-03-30', '%Y-%m-%d')

    for ticker in tickers:
        data = store.getdata(dataname=f"{ticker}/USDT", name=ticker,
                             timeframe=bt.TimeFrame.Days,
                             fromdate=fromdate,
                             todate=todate,
                             compression=1,
                             ohlcv_limit=10000,
                             drop_newest=True)  # , historical=True)
        cerebro.adddata(data)
        data.plotinfo.plot = False

    cerebro.broker.setcash(10000)
    cerebro.addobserver(bt.observers.Value)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, riskfreerate=0.0)
    cerebro.addanalyzer(bt.analyzers.Returns)
    cerebro.addanalyzer(bt.analyzers.DrawDown)
    cerebro.addstrategy(MeanReversion)

    results = cerebro.run()

    print(f"Sharpe: {results[0].analyzers.sharperatio.get_analysis()['sharperatio']:.3f}")
    print(f"Norm. Annual Return: {results[0].analyzers.returns.get_analysis()['rnorm100']:.2f}%")
    print(f"Max Drawdown: {results[0].analyzers.drawdown.get_analysis()['max']['drawdown']:.2f}%")
    cerebro.plot()[0][0]


if __name__ == "__main__":
    backtest()