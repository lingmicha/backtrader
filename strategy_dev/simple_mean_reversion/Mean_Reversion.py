import backtrader as bt
import numpy as np
import pandas as pd
import os
import json
import time
from ccxtbt import CCXTStore
from datetime import datetime
from functools import reduce


class BinanceComissionInfo(bt.CommissionInfo):
    params = (
        ("commission", 0.075),
        ("mult", 1.0),
        ("margin", None),
        ("commtype", bt.CommInfoBase.COMM_PERC),
        ("stocklike", True),
        ("percabs", False),
        ("interest", 0.0),
        ("interest_long", False),
        ("leverage", 1.0),
        ("automargin", False),
    )

    def getsize(self, price, cash):
        """Returns fractional size for cash operation @price"""
        return self.p.leverage * (cash / price)


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

        print("This is a new strat")

    def prenext(self):
        self.next()

    def nextstart(self):
        self.next()

    def next(self):

        # only look at data that existed last week
        available = list(filter(lambda d: len(d) > self.p.sma + 2, self.datas))

        if len(available):
            print( f"{available[0].datetime.datetime(0)}: AVAILABLE TICKERS: {len(available)}")
        else:
            return

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
        regimes = np.zeros(len(available))

        for i, d in enumerate(available):
            rets[i] = self.inds[d]['pct'][0]
            stds[i] = self.inds[d]['std'][0]
            smas[i] = self.inds[d]['sma'][0]
            regimes[i] = d.close[0] - self.inds[d]['sma'][0]

        market_ret = np.mean(rets)
        weights = -(rets - market_ret)
        print(f"{self.data.datetime.datetime(0)} MARKET RETURN:{market_ret}")

        # for i, d in enumerate(available):
        #     # weights[i] = weights[i] if np.sign(weights[i]) == np.sign( d.close[0]  - smas[i]) \
        #     #                 else 0
        #     if weights[i] > 0:
        #         if d.close[0] < smas[i]:
        #             weight = 0
        #             if self.p.debug:
        #                 print(f"{d.datetime.datetime(0)} CANNOT GO LONG WITH {d._name}")
        #         else:
        #             weight = weights[i]
        #     else:
        #         if d.close[0] > smas[i]:
        #             weight = 0
        #             if self.p.debug:
        #                 print(f"{d.datetime.datetime(0)} CANNOT GO SHORT WITH {d._name}")
        #         else:
        #             weight = weights[i]
        #
        #     weights[i] = weight

        max_weights_index = max_n(np.abs(weights), self.params.n)
        low_volality_index = min_n(stds, self.params.n)

        if self.p.debug:
            msg = list()
            msg.append( f"TOP {self.params.n} [TICKER|RETURN]: ")
            # print selected top params.n and their returns
            for i, d in enumerate(available):
                if i in max_weights_index:
                    msg.append( f"[{d._name}|{rets[i]:.3f}] " )
            print( msg )

        if self.p.vol_filter:
            selected_weights_index = np.intersect1d(max_weights_index,
                                                    low_volality_index)
        else:
            selected_weights_index = max_weights_index

        not_allowed_shorts = reduce(np.intersect1d, (selected_weights_index, np.nonzero(regimes > 0), np.nonzero( weights < 0 ) ))
        not_allowed_longs = reduce(np.intersect1d, (selected_weights_index, np.nonzero(regimes < 0), np.nonzero( weights > 0 ) ))

        if self.p.debug:
            if len(not_allowed_shorts):
                print( f"EXCLUDE SHORTS: {[ available[x]._name for x in not_allowed_shorts ]} ")
            if len(not_allowed_longs):
                print( f"EXCLUDE LONGS: {[ available[x]._name for x in not_allowed_longs ]} " )

        selected_weights_index = selected_weights_index[~np.isin( selected_weights_index, not_allowed_shorts )]
        selected_weights_index = selected_weights_index[~np.isin( selected_weights_index, not_allowed_longs)]
        print(f"{self.data.datetime.datetime(0)}: POSITIONS TAKEN:{len(selected_weights_index)} ")

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


    def notify_trade(self, trade):
        """Execute after each trade
        Calcuate Gross and Net Profit/loss"""

        # trade closed
        if trade.isclosed:
            print(
                f"{trade.data.datetime.datetime(0)} OPERATIONAL PROFIT, GROSS: {trade.data._name}, {trade.pnl:.2f}, Net: {trade.pnlcomm:.2f}"
            )

def spot_backtest_livedata():

    cerebro = bt.Cerebro(stdstats=False)
    cerebro.broker.set_coc(True)

    # read ticker file
    tickers_file = 'tickers.csv'
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
    cerebro.broker.addcommissioninfo(BinanceComissionInfo())

    cerebro.addobserver(bt.observers.Value)
    cerebro.addanalyzer(bt.analyzers.TimeReturn, timeframe=bt.TimeFrame.NoTimeFrame, _name='alltimereturn')
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, riskfreerate=0.0)
    cerebro.addanalyzer(bt.analyzers.Returns)
    cerebro.addanalyzer(bt.analyzers.DrawDown)

    cerebro.addstrategy(CrossSectionalMR,
                        n=35,
                        pct=1,
                        std=20,
                        sma=28,
                        vol_filter=False,
                        debug=True,
                        )

    stratrun = cerebro.run()

    print(f"Return: {list(stratrun[0].analyzers.alltimereturn.get_analysis().values())[0]:.3f}")
    print(f"Sharpe: {stratrun[0].analyzers.sharperatio.get_analysis()['sharperatio']:.3f}")
    print(f"Norm. Annual Return: {stratrun[0].analyzers.returns.get_analysis()['rnorm100']:.2f}%")
    print(f"Max Drawdown: {stratrun[0].analyzers.drawdown.get_analysis()['max']['drawdown']:.2f}%")

    cerebro.plot()[0][0]

def spot_backtest_filedata():


    cerebro = bt.Cerebro(stdstats=False,
                         runonce=False,
                         )
    cerebro.broker.set_coc(True)

    # read ticker file
    data_path = '../datas/binance-spot-20220330-BEARMARKET'
    #data_path = '../datas/binance-spot-BEARMARKET'
    #tickers_file = 'tickers_20171201.csv'
    tickers_file = 'tickers.csv'


    tickers = pd.read_csv(f"{data_path}/{tickers_file}", header=None)[1].to_list()
    print(f"TOTAL {len(tickers)} CRYPTOS ON BINANCE BEFORE 2017-12-01")

    # make sure BTC is data0
    # tickers.remove('BTC')
    # tickers.insert(0,'BTC')

    # TODO: DATA0 Must have the earilest start datetime
    fromdate = datetime.strptime('2017-12-01', '%Y-%m-%d')
    todate = datetime.strptime('2020-03-30', '%Y-%m-%d')

    for ticker in tickers:
        df = pd.read_csv(f"{data_path}/{ticker}.csv",
                         parse_dates=True,
                         index_col=0)
        if len(df) > 100:  # data must be long enough to compute 100 day SMA
            cerebro.adddata(bt.feeds.PandasData(dataname=df,
                                                name=ticker,
                                                fromdate=fromdate,
                                                todate=todate,
                                                plot=False))

    cerebro.broker.setcash(10000)
    #cerebro.broker.addcommissioninfo(BinanceComissionInfo())

    cerebro.addobserver(bt.observers.Value)
    cerebro.addanalyzer(bt.analyzers.TimeReturn, timeframe=bt.TimeFrame.NoTimeFrame, _name='alltimereturn')
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, riskfreerate=0.0)
    cerebro.addanalyzer(bt.analyzers.Returns)
    cerebro.addanalyzer(bt.analyzers.DrawDown)

    cerebro.addstrategy(CrossSectionalMR,
                        n=30,
                        pct=2,
                        std=20,
                        sma=20,
                        vol_filter=False,
                        debug=True,
                        )

    stratrun = cerebro.run()

    result={}
    result["n"] = stratrun[0].p.n
    result["pct"] = stratrun[0].p.pct
    result["std"] = stratrun[0].p.std
    result["sma"] = stratrun[0].p.sma
    result["vol_filter"] = stratrun[0].p.vol_filter
    result["return"] = list(stratrun[0].analyzers.alltimereturn.get_analysis().values())[0]
    result["sharpe"] = stratrun[0].analyzers.sharperatio.get_analysis()['sharperatio']
    result["annual_return"] = stratrun[0].analyzers.returns.get_analysis()['rnorm100']
    result["max_drawdown"] = stratrun[0].analyzers.drawdown.get_analysis()['max']['drawdown']


    print(f"Run-Reulst: ",
            f"n={result['n']}, "
            f"pct={result['pct']}, "
            f"std={result['std']}, "
            f"sma={result['sma']}, "
            f"vol_filter={result['vol_filter']}, "
            f"Return: {result['return']:.3f} ",
            f"Sharpe: {result['sharpe']:.3f} ",
            f"Norm. Annual Return: {result['annual_return']:.2f}% ",
            f"Max Drawdown: {result['max_drawdown']:.2f}% ",
          )

    cerebro.plot(iplot=False)[0][0]


    return

def spot_opt_backtest():

    cerebro = bt.Cerebro(stdstats=False,
                         optreturn=True,
                         )
    cerebro.broker.set_coc(True)

    # read ticker file
    tickers_file = 'tickers.csv'
    tickers = pd.read_csv(f"data/{tickers_file}", header=None)[1].to_list()
    print(f"TOTAL {len(tickers)} CRYPTOS ON BINANCE BEFORE 2020-01-01")

    # make sure BTC is data0
    tickers.remove('BTC')
    tickers.insert(0,'BTC')

    # TODO: DATA0 Must have the earilest start datetime
    fromdate = datetime.strptime('2020-01-01', '%Y-%m-%d')
    todate = datetime.strptime('2022-03-30', '%Y-%m-%d')

    for ticker in tickers:
        df = pd.read_csv(f"data/{ticker}.csv",
                         parse_dates=True,
                         index_col=0)
        if len(df) > 100:  # data must be long enough to compute 100 day SMA
            cerebro.adddata(bt.feeds.PandasData(dataname=df,
                                                name=ticker,
                                                fromdate=fromdate,
                                                todate=todate,
                                                plot=False))

    cerebro.broker.setcash(10000)
    #cerebro.broker.addcommissioninfo(BinanceComissionInfo())

    cerebro.addobserver(bt.observers.Value)
    cerebro.addanalyzer(bt.analyzers.TimeReturn, timeframe=bt.TimeFrame.NoTimeFrame, _name='alltimereturn')
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, riskfreerate=0.0)
    cerebro.addanalyzer(bt.analyzers.Returns)
    cerebro.addanalyzer(bt.analyzers.DrawDown)

    cerebro.optstrategy(CrossSectionalMR,
                        n=range(5,51,5),
                        pct=range(1,4),
                        std=20,
                        sma=range(10,36),
                        vol_filter=False,
                        debug=True,
                        )

    stratruns = cerebro.run()

    results = []
    for i, stratrun in enumerate(stratruns):
        result = {}
        result["run_no"] = i
        result["n"] = stratrun[0].p.n
        result["pct"] = stratrun[0].p.pct
        result["std"] = stratrun[0].p.std
        result["sma"] = stratrun[0].p.sma
        result["vol_filter"] = stratrun[0].p.vol_filter
        result["return"] = list(stratrun[0].analyzers.alltimereturn.get_analysis().values())[0]
        result["sharpe"] = stratrun[0].analyzers.sharperatio.get_analysis()['sharperatio']
        result["annual_return"] = stratrun[0].analyzers.returns.get_analysis()['rnorm100']
        result["max_drawdown"] = stratrun[0].analyzers.drawdown.get_analysis()['max']['drawdown']

        results.append(result)

        print(f"Run-{i}th: ",
                f"n={result['n']}, "
                f"pct={result['pct']}, "
                f"std={result['std']}, "
                f"sma={result['sma']}, "
                f"vol_filter={result['vol_filter']}, "
                f"Return: {result['return']:.3f} ",
                f"Sharpe: {result['sharpe']:.3f} ",
                f"Norm. Annual Return: {result['annual_return']:.2f}% ",
                f"Max Drawdown: {result['max_drawdown']:.2f}% ",
              )

    df = pd.DataFrame(results)
    df.set_index('run_no', inplace=True)
    df.to_csv('MR-Parameter-Opt.csv')


def spot_scenarios_backtest():

    # read ticker file
    tickers_file = 'tickers.csv'
    data_path = '../datas/binance-spot-cg-20220330'
    tickers = pd.read_csv(f"{data_path}/{tickers_file}", header=None)[1].to_list()
    print(f"TOTAL {len(tickers)} CRYPTOS ON BINANCE BEFORE 2022-03-30")

    # TODO: DATA0 Must have the earilest start datetime
    fromdate = datetime.strptime('2017-12-01', '%Y-%m-%d')
    todate = datetime.strptime('2022-03-30', '%Y-%m-%d')


    results = []
    n = range(5, 51, 5)
    pct = range(1, 4)
    sma = range(10, 36)
    import itertools
    params_set = list(itertools.product(n, pct, sma))

    for i, params in enumerate(params_set):

        cerebro = bt.Cerebro(stdstats=False,
                             # maxcpus=1,
                             # optreturn=True,
                             # exactbars=True,
                             preload=False,
                             runonce=False,
                             )
        cerebro.broker.set_coc(True)

        for ticker in tickers:
            df = pd.read_csv(f"{data_path}/{ticker}.csv",
                             parse_dates=True,
                             index_col=0)
            if len(df) > 100:  # data must be long enough to compute 100 day SMA
                cerebro.adddata(bt.feeds.PandasData(dataname=df,
                                                    name=ticker,
                                                    fromdate=fromdate,
                                                    todate=todate,
                                                    plot=False))

        cerebro.broker.setcash(10000)
        #cerebro.broker.addcommissioninfo(BinanceComissionInfo())

        cerebro.addobserver(bt.observers.Value)
        cerebro.addanalyzer(bt.analyzers.TimeReturn, timeframe=bt.TimeFrame.NoTimeFrame, _name='alltimereturn')
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, riskfreerate=0.0)
        cerebro.addanalyzer(bt.analyzers.Returns)
        cerebro.addanalyzer(bt.analyzers.DrawDown)

        cerebro.addstrategy(CrossSectionalMR,
                            n=params[0],
                            pct=params[1],
                            std=20,
                            sma=params[2],
                            vol_filter=False,
                            debug=True,
                            )

        stratrun = cerebro.run()

        result = {}
        result["run_no"] = i
        result["n"] = stratrun[0].p.n
        result["pct"] = stratrun[0].p.pct
        result["std"] = stratrun[0].p.std
        result["sma"] = stratrun[0].p.sma
        result["vol_filter"] = stratrun[0].p.vol_filter
        result["return"] = list(stratrun[0].analyzers.alltimereturn.get_analysis().values())[0]
        result["sharpe"] = stratrun[0].analyzers.sharperatio.get_analysis()['sharperatio']
        result["annual_return"] = stratrun[0].analyzers.returns.get_analysis()['rnorm100']
        result["max_drawdown"] = stratrun[0].analyzers.drawdown.get_analysis()['max']['drawdown']

        results.append(result)

        print(f"Run-{i}th: ",
                f"n={result['n']}, "
                f"pct={result['pct']}, "
                f"std={result['std']}, "
                f"sma={result['sma']}, "
                f"vol_filter={result['vol_filter']}, "
                f"Return: {result['return']:.3f} ",
                f"Sharpe: {result['sharpe']:.3f} ",
                f"Norm. Annual Return: {result['annual_return']:.2f}% ",
                f"Max Drawdown: {result['max_drawdown']:.2f}% ",
              )

    df = pd.DataFrame(results)
    df.set_index('run_no', inplace=True)
    df.to_csv('MR-Parameter-Opt.csv')



if __name__ == "__main__":
    #spot_backtest_livedata()
    spot_backtest_filedata()
    #spot_opt_backtest()
    #spot_scenarios_backtest()