import backtrader as bt
import numpy as np
import pandas as pd
import os
import json
import time
from ccxtbt import CCXTStore
from datetime import datetime


def min_n(array, n):
    return np.argpartition(array, n)[:n]

def max_n(array, n):
    return np.argpartition(array, -n)[-n:]

class CrossSectionalMR(bt.Strategy):
    '''
        Mean-Reversion: use market mean return as benchmark to perform mena reversion.

        Params:
            - ``n``: (default:20)
            maximum number of cryptos position

            - pct: (default:2)
            period to calculate price pct change as return
            the largest n cryptos in terms of return diff
            with market return will be hold during the period

            - std: (default:20)
            period to calculate price standard deviation

            - sma: (default:20)
            period to calculate moving average

            - vol_filter: (default:False)
            enable/disable vol_filter, vol_filter would require the ticker
            to rank lowest n among all possible cryptos.
            it would reduce the number of positions to hold, and hence increase
            the concentration risk while greatly boost return during bull market

            - debug: (default: False)
            output debug information


    '''



    params = (
        ('n', 20),
        ('pct', 2),
        ('std', 20),
        ('sma', 20),
        ('vol_filter', False),
        ('debug', False),
    )

    def __init__(self):
        self.inds = {}
        for d in self.datas:
            self.inds[d] = {}
            self.inds[d]["pct"] = bt.indicators.PercentChange(d.close, period=self.p.pct)
            self.inds[d]["std"] = bt.indicators.StandardDeviation(d.close, period=self.p.std)
            self.inds[d]["sma"] = bt.indicators.SimpleMovingAverage(d.close, period=self.p.sma)

    def prenext(self):
        self.next()

    def nextstart(self):
        self.next()

    def next(self):
        # only look at data that existed last week
        available = list(filter(lambda d: len(d) > self.p.sma + 2, self.datas))

        if self.p.debug:
            for i, d in enumerate(available):
                size = self.getposition(d).size
                price = self.getposition(d).price
                value = size * price
                if size !=0:
                    print(f"{d.datetime.datetime(0)} POSITION-{d._name}: PRICE:{price}, SIZE:{size}, VALUE:{value}")

        rets = np.zeros(len(available))
        stds = np.zeros(len(available))
        smas = np.zeros(len(available))

        for i, d in enumerate(available):
            rets[i] = self.inds[d]['pct'][0]
            stds[i] = self.inds[d]['std'][0]
            smas[i] = self.inds[d]['sma'][0]

        market_ret = np.mean(rets)
        weights = -(rets - market_ret)
        print(f"{self.data.datetime.datetime(0)} MARKET RETURN:{market_ret}")

        for i, d in enumerate(available):
            # weights[i] = weights[i] if np.sign(weights[i]) == np.sign( d.close[0]  - smas[i]) \
            #                 else 0
            if weights[i] > 0:
                if d.close[0] < smas[i]:
                    weight = 0
                    if self.p.debug:
                        print(f"{d.datetime.datetime(0)} CANNOT GO LONG WITH {d._name}")
                else:
                    weight = weights[i]
            else:
                if d.close[0] > smas[i]:
                    weight = 0
                    if self.p.debug:
                        print(f"{d.datetime.datetime(0)} CANNOT GO SHORT WITH {d._name}")
                else:
                    weight = weights[i]

            weights[i] = weight

        max_weights_index = max_n(np.abs(weights), self.params.n)
        low_volality_index = min_n(stds, self.params.n)

        if self.p.vol_filter:
            selected_weights_index = np.intersect1d(max_weights_index,
                                                    low_volality_index)
        else:
            selected_weights_index = max_weights_index

        if not len(selected_weights_index):
            # no good trades today
            print(f"{self.data.datetime.datetime(0)} NO TRADES TODAY")
            for i, d in enumerate(available):
                self.close(d)
            return

        selected_weights = weights[selected_weights_index]
        weights = weights / np.sum(np.abs(selected_weights))

        for i, d in enumerate(available):
            if i in selected_weights_index:
                self.order_target_percent(d, target=weights[i])
            else:
                self.order_target_percent(d, 0)

        number_of_position_today = len(list(filter( lambda i: weights[i] != 0, selected_weights_index)))
        print(f"{self.data.datetime.datetime(0)}: POSITIONS TAKEN:{number_of_position_today} ")

    def notify_trade(self, trade):
        """Execute after each trade
        Calcuate Gross and Net Profit/loss"""

        # trade closed
        if trade.isclosed:
            print(
                f"{trade.data.datetime.datetime(0)} OPERATIONAL PROFIT, GROSS: {trade.data._name}, {trade.pnl:.2f}, Net: {trade.pnlcomm:.2f}"
            )

def spot_backtest():

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
    todate = datetime.strptime('2021-01-30', '%Y-%m-%d')

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
    cerebro.addstrategy(CrossSectionalMR,
                        n=20,
                        pct=2,
                        std=20,
                        sma=20,
                        vol_filter=False,
                        debug=True,
                        )

    results = cerebro.run()

    print(f"Sharpe: {results[0].analyzers.sharperatio.get_analysis()['sharperatio']:.3f}")
    print(f"Norm. Annual Return: {results[0].analyzers.returns.get_analysis()['rnorm100']:.2f}%")
    print(f"Max Drawdown: {results[0].analyzers.drawdown.get_analysis()['max']['drawdown']:.2f}%")
    cerebro.plot()[0][0]


if __name__ == "__main__":
    spot_backtest()