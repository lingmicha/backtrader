import json
from datetime import datetime
import pandas as pd
import backtrader as bt
from ccxtbt import CCXTStore
import time

from strategy_dev import BinancePerpetualFutureCommInfo

class DataSet:
    '''
        NOTES ON DATASET:

        BULL MARKET CASE: 2020.3~2022.12
        BEAR MARKET CASE: 2017.12~2020.3

        (1) BINANCE_SPOT_201708_202203:
            spot market usdt quote pairs, binance history dates back to 201708
        (2) BINANCE_SPOT_BTC_QUOTE_201708_202203:
            spot market btc quote paris, only close price available,
            price is btc quote price converted into usdt price.
            since usdt becomes popular after 2018, so this gives us more coins during
            bear market 2017.12~2020.3
        (3) BINANCE_FUTURE_202001_202203:
            a subset of BINANCE_FUTURE_201708_202203, consider remove in future
        (4) BINANCE_FUTURE_201708_202203:
            binance future market usdt quote pairs
    '''
    (BINANCE_SPOT_201708_202203,
     BINANCE_SPOT_BTC_QUOTE_201708_202203,
     BINANCE_FUTURE_202001_202203,
     BINANCE_FUTURE_201708_202203,
     ) = range(1,5)

    DATA_PATH = '../datas'
    CONFIG_PATH = '../config'
    TICKERS_FILE = 'tickers.csv'

    def __init__(self, dataset):

        if dataset == DataSet.BINANCE_SPOT_201708_202203:
            self.data_path = f"{DataSet.DATA_PATH}/binance-spot-20220330"
        elif dataset == DataSet.BINANCE_SPOT_BTC_QUOTE_201708_202203:
            self.data_path = f"{DataSet.DATA_PATH}/binance-spot-20220330-BEARMARKET"
        elif dataset == DataSet.BINANCE_FUTURE_201708_202203:
            self.data_path = f"{DataSet.DATA_PATH}/binance-future-20220330"
        elif dataset == DataSet.BINANCE_FUTURE_202001_202203:
            self.data_path = f"{DataSet.DATA_PATH}/binance-future-20200101-20220330"
        else:
            raise Exception(f"dataset not recognized!!!")

        self.tickers = pd.read_csv(f"{self.data_path}/{DataSet.TICKERS_FILE}", header=None)[1].to_list()
        # Make sure BTC is data0, as data0 need to contain the earliest bar,
        # otherwise backtrader might messup the timeframe
        self.tickers.remove('BTC')
        self.tickers.insert(0,'BTC')

        print(f"TOTAL {len(self.tickers)} CRYPTO TICKERS ON THIS DATASET")

    def get_tickers(self):
        return self.tickers

    def configure_file_backtest(self, start, end, cerebro, strategy, optimize=False, **kwargs):

        self.cerebro = cerebro
        self.optimize = optimize

        # TODO: DATA0 Must have the earilest start datetime
        fromdate = datetime.strptime(start, '%Y-%m-%d')
        todate = datetime.strptime(end, '%Y-%m-%d')

        if self.tickers is not None and len(self.tickers) > 0:
            for ticker in self.tickers:
                df = pd.read_csv(f"{self.data_path}/{ticker}.csv",
                                 parse_dates=True,
                                 index_col=0)

                if len(df) > 20:  # at least 20 day's bar
                    cerebro.adddata(bt.feeds.PandasData(dataname=df,
                                                        name=ticker,
                                                        fromdate=fromdate,
                                                        todate=todate,
                                                        plot=False))

        cerebro.broker.setcash(10000)
        cerebro.broker.set_coc(True)
        cerebro.broker.addcommissioninfo(BinancePerpetualFutureCommInfo())

        if optimize:
            cerebro.optstrategy(strategy,
                                **kwargs,)
        else:
            cerebro.addstrategy(strategy,
                                **kwargs,)

        cerebro.addobserver(bt.observers.Value)
        cerebro.addanalyzer(bt.analyzers.TimeReturn, timeframe=bt.TimeFrame.NoTimeFrame, _name='alltimereturn')
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, riskfreerate=0.0)
        cerebro.addanalyzer(bt.analyzers.Returns)
        cerebro.addanalyzer(bt.analyzers.DrawDown)

    def configure_livedata_backtest(self, market, start, end, cerebro, strategy, optimize=False, **kwargs):

        self.cerebro = cerebro
        self.optimize = optimize

        # absolute dir the script is in
        config_file = f"{DataSet.CONFIG_PATH}/params-production-{market}.json"
        with open(config_file, 'r') as f:
            params = json.load(f)

        # Create our store
        config = {'apiKey': params["binance"]["apikey"],
                  'secret': params["binance"]["secret"],
                  'enableRateLimit': True,
                  'options': {
                      'defaultType': market,
                  },
                  'nonce': lambda: str(int(time.time() * 1000)),
                  }

        store = CCXTStore(exchange='binance', currency='USDT', config=config, retries=5, debug=False, sandbox=False)

        # TODO: DATA0 Must have the earilest start datetime
        fromdate = datetime.strptime(start, '%Y-%m-%d')
        todate = datetime.strptime(end, '%Y-%m-%d')

        if self.tickers is not None and len(self.tickers) > 0:
            for ticker in self.tickers:
                data = store.getdata(dataname=f"{ticker}/USDT", name=ticker,
                                     timeframe=bt.TimeFrame.Days,
                                     fromdate=fromdate,
                                     todate=todate,
                                     compression=1,
                                     ohlcv_limit=10000,
                                     drop_newest=True,
                                     historical=True)
                cerebro.adddata(data)
                data.plotinfo.plot = False

        cerebro.broker.setcash(10000)
        cerebro.broker.set_coc(True)
        cerebro.broker.addcommissioninfo(BinancePerpetualFutureCommInfo())

        if optimize:
            cerebro.optstrategy(strategy,
                                **kwargs,)
        else:
            cerebro.addstrategy(strategy,
                                **kwargs,)

        cerebro.addobserver(bt.observers.Value)
        cerebro.addanalyzer(bt.analyzers.TimeReturn, timeframe=bt.TimeFrame.NoTimeFrame, _name='alltimereturn')
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, riskfreerate=0.0)
        cerebro.addanalyzer(bt.analyzers.Returns)
        cerebro.addanalyzer(bt.analyzers.DrawDown)


    def run_backtest(self):
        if not self.optimize:
            stratrun = self.cerebro.run()
            result = DataSet.extract_result(stratrun)
            return result

        stratruns = self.cerebro.run()
        results = []
        for stratrun in stratruns:
            result = DataSet.extract_result(stratrun)
            results.append(result)
        return results

    @staticmethod
    def extract_result(stratrun):

        result = {}
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

        return result

