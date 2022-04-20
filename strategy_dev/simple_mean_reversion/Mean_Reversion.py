import signal
import backtrader as bt
import numpy as np
import pandas as pd
import os
import json
import time
from ccxtbt import CCXTStore
from datetime import datetime
from strategy_dev import AlertEmailer, DataSet, BinancePerpetualFutureCommInfo, CryptoStrategy
from tabulate import tabulate

def min_n(array, n):
    return np.argpartition(array, n)[:n]

def max_n(array, n):
    return np.argpartition(array, -n)[-n:]

class CrossSectionalMR(CryptoStrategy):
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
    )

    def __init__(self):

        self.inds = {}
        for d in self.datas:
            self.inds[d] = {}
            self.inds[d]["pct"] = bt.indicators.PercentChange(d.close, period=self.p.pct)
            self.inds[d]["std"] = bt.indicators.StandardDeviation(d.close, period=self.p.std)
            self.inds[d]["sma"] = bt.indicators.SimpleMovingAverage(d.close, period=self.p.sma)

        super().__init__() # calling CryptoStrategy init

    def prenext(self):
        self.next()

    def nextstart(self):
        self.next()

    def next(self):

        if self.p.live_trading and (not self.live_data):
            return  # prevent live trading with delayed data

        self.manual_update_balance()
        self.print_period_stats()

        # only look at data that existed last week
        available = list(filter(lambda d: len(d) > self.p.sma + 2, self.datas))

        if len(available) > self.p.n:
            print( f"{available[0].datetime.date(0)}: AVAILABLE TICKERS: {len(available)}")
        else:
            print(f"NOT ENOUGH TICKERS FOR THE STRATEGY, CURRENTLY {len(available)} TICKERS")
            return

        rets = np.zeros(len(available))
        stds = np.zeros(len(available))
        smas = np.zeros(len(available))
        regimes = np.zeros(len(available))

        for i, d in enumerate(available):
            rets[i] = self.inds[d]['pct'][0]
            stds[i] = self.inds[d]['std'][0]
            smas[i] = self.inds[d]['sma'][0]
            regimes[i] = d.close[0] - self.inds[d]['sma'][0]

        market_ret = np.mean(rets)
        weights = -(rets - market_ret)

        not_allowed_shorts = np.intersect1d(np.nonzero(regimes > 0), np.nonzero(weights < 0))
        not_allowed_longs = np.intersect1d(np.nonzero(regimes < 0), np.nonzero(weights > 0))
        not_allowed = np.union1d(not_allowed_shorts, not_allowed_longs)

        weights[not_allowed] = 0 # mask not allowed
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

        # First close position to gather cash, then enter new position
        for i, d in enumerate(available):
            if i not in selected_weights_index:
                self.order_target_percent(d, 0)

        for i, d in enumerate(available):
            if i in selected_weights_index:
                self.order_target_percent(d, target=weights[i])

        # in the end , print today's summary
        if self.p.debug:
            headers = [
                'No.', 'TICKER','RETURN','MARKET_RETURN','EXCESS_RETURN',
                'FORBID_LONG','FORBID_SHORT','FINAL_WEIGHT',
            ]
            data = list()

            for i, d in enumerate(available):
                data.append(
                    [ i, d._name, rets[i], market_ret, rets[i] - market_ret,
                      'TRUE' if i in not_allowed_longs else '-',
                      'TRUE' if i in not_allowed_shorts else '-',
                      weights[i],
                    ]
                )

            print(f"{self.datas[0].datetime.date(0)} STRATEGY STATS TABLE:")
            print(tabulate(data, headers=headers, tablefmt="github"))

def file_backtest():

    dataset = DataSet(DataSet.BINANCE_FUTURE_201708_202203)
    start = '2021-12-01'
    end = '2022-03-30'

    cerebro = bt.Cerebro(stdstats=False,
                         runonce=False,)

    dataset.configure_file_backtest(start,
                                    end,
                                    cerebro,
                                    CrossSectionalMR,
                                    optimize=False,
                                    n=30,
                                    pct=2,
                                    std=20,
                                    sma=20,
                                    vol_filter=False,
                                    debug=True,
                                    )

    result = dataset.run_backtest()

    cerebro.plot(iplot=False)[0][0]

    return

def file_backtests():

    dataset = DataSet(DataSet.BINANCE_SPOT_201708_202203)
    start = '2020-01-01'
    end = '2022-03-30'

    cerebro = bt.Cerebro(stdstats=False,
                         runonce=False,
                         )

    dataset.configure_file_backtest(start,
                                    end,
                                    cerebro,
                                    CrossSectionalMR,
                                    optimize=True,
                                    n=30,
                                    pct=range(1,3),
                                    std=20,
                                    sma=20,
                                    vol_filter=False,
                                    debug=False,
                                    )

    results = dataset.run_backtest()

    df = pd.DataFrame(results)
    df.to_csv('MR-Parameter-Opt.csv')

def file_backtests_seqrun():

    dataset = DataSet(DataSet.BINANCE_SPOT_201708_202203)
    start = '2020-01-01'
    end = '2022-03-30'

    results = []
    #n = range(5, 51, 5)
    n = [30]
    #pct = range(1, 4)
    pct = range(1, 3)
    #sma = range(10, 36)
    sma = [20]
    import itertools
    params_set = list(itertools.product(n, pct, sma))

    for i, params in enumerate(params_set):
        cerebro = bt.Cerebro(stdstats=False,
                             runonce=False,
                             )
        dataset.configure_file_backtest(start,
                                        end,
                                        cerebro,
                                        CrossSectionalMR,
                                        optimize=False,
                                        n=params[0],
                                        pct=params[1],
                                        std=20,
                                        sma=params[2],
                                        vol_filter=False,
                                        debug=True,
                                        )
        result = dataset.run_backtest()
        results.append(result)

    df = pd.DataFrame(results)
    df.to_csv('MR-Parameter-Opt.csv')

def livedata_backtest():

    dataset = DataSet(DataSet.BINANCE_SPOT_201708_202203)
    start = '2020-01-01'
    end = '2022-03-30'

    cerebro = bt.Cerebro(stdstats=False,
                         runonce=False,
                         )

    dataset.configure_livedata_backtest('spot',
                                        start,
                                        end,
                                        cerebro,
                                        CrossSectionalMR,
                                        optimize=False,
                                        n=30,
                                        pct=2,
                                        std=20,
                                        sma=20,
                                        vol_filter=False,
                                        debug=True,
                                        )

    result = dataset.run_backtest()

    cerebro.plot(iplot=False)[0][0]

    return

def run_live_trading():

    # absolute dir the script is in
    script_dir = os.path.dirname(__file__)
    abs_file_path = os.path.join(script_dir, '../config/params-production-future.json')
    with open(abs_file_path, 'r') as f:
        params = json.load(f)

    # get emailer
    mailer = AlertEmailer.getInstance()
    mailer.set_parameter(params["email"]["host"],
                         params["email"]["user"],
                         params["email"]["pass"],
                         params["email"]["port"])
    mailer.set_sender_receiver(params["email"]["sender"],
                               params["email"]["receiver"])

    mailer.send_email_alert("IMPORTANT:THIS IS A LIVE TRADING SESSION!!!")

    # Create our store
    config = {'apiKey': params["binance"]["apikey"],
              'secret': params["binance"]["secret"],
              'enableRateLimit': True,
              'options': {
                  'defaultType': 'future',
              },
              'nonce': lambda: str(int(time.time() * 1000)),
              }

    store = CCXTStore(exchange='binance', currency='USDT', config=config, retries=5, debug=False, sandbox=False)

    cerebro = bt.Cerebro(stdstats=False,
                         quicknotify=True,
                         exactbars=True,
                         )

    # read ticker file
    tickers_file = 'tickers.csv'
    data_path = '../datas/binance-future-20220330'
    tickers = pd.read_csv(f"{data_path}/{tickers_file}", header=None)[1].to_list()
    print(f"TOTAL {len(tickers)} CRYPTOS ON BINANCE FUTURE BEFORE 2022-03-30")

    # Make sure BTC is data0, as data0 need to contain the earliest bar,
    # otherwise backtrader might messup the timeframe
    tickers.remove('BTC')
    tickers.insert(0, 'BTC')

    # TODO: DATA0 Must have the earilest start datetime
    fromdate = datetime.strptime('2022-03-01', '%Y-%m-%d')

    for ticker in tickers:
        data = store.getdata(dataname=f"{ticker}/USDT", name=ticker,
                             timeframe=bt.TimeFrame.Days,
                             fromdate=fromdate,
                             #todate=todate,
                             compression=1,
                             ohlcv_limit=10000,
                             drop_newest=True)  # , historical=True)
        cerebro.adddata(data)
        data.plotinfo.plot = False

    # Get the broker and pass any kwargs if needed.
    # ----------------------------------------------
    # Broker mappings have been added since some exchanges expect different values
    # to the defaults. Case in point, Kraken vs Bitmex. NOTE: Broker mappings are not
    # required if the broker uses the same values as the defaults in CCXTBroker.
    broker_mapping = {
        'order_types': {
            bt.Order.Market: 'market',
            bt.Order.Limit: 'limit',
            bt.Order.Stop: 'stop-loss',  # stop-loss for kraken, stop for bitmex
            bt.Order.StopLimit: 'stop limit'
        },
        'mappings': {
            'closed_order': {
                'key': 'status',
                'value': 'closed'
            },
            'canceled_order': {
                'key': 'status',
                'value': 'canceled'
            }
        }
    }

    broker = store.getbroker(broker_mapping=broker_mapping)
    cerebro.setbroker(broker)
    cerebro.broker.addcommissioninfo(BinancePerpetualFutureCommInfo())

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

    cerebro.addstrategy(CrossSectionalMR,
                        n=30,
                        pct=2,
                        std=20,
                        sma=20,
                        vol_filter=False,
                        debug=True,
                        live_trading=True,
                        )

    stratrun = cerebro.run()

    pyfoliozer = stratrun[0].analyzers.getbyname('pyfolio')
    returns, positions, transactions, gross_lev = pyfoliozer.get_pf_items()
    returns.to_csv("returns.csv")
    positions.to_csv("positions.csv")
    transactions.to_csv("transactions.csv")

    result = DataSet.extract_result(stratrun)

    cerebro.plot(iplot=False)[0][0]

if __name__ == "__main__":

    file_backtest()
    #file_backtests()
    #file_backtests_seqrun()
    #livedata_backtest()
    #run_live_trading()