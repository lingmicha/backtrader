import backtrader as bt
import numpy as np
import scipy.stats as stats
from setup_backtest import DataSet
from datetime import datetime
import signal
import time
from tabulate import tabulate
import os
import json
from ccxtbt import CCXTStore
import pandas as pd


from strategy_dev import AlertEmailer, BinancePerpetualFutureCommInfo


class ZScore(bt.ind.BaseApplyN):
    lines = ('zscore',)
    params = (('period', 20),
              ('func', lambda x: stats.zscore(x)[-1]),
              )


def min_n(array, n):
    return np.argpartition(array, n)[:n]


def max_n(array, n):
    return np.argpartition(array, -n)[-n:]


class LongShortReturn(bt.Strategy):
    params = (
        ('pct', 2),
        ('std', 20),
        ('live_trading', False),
    )

    MIN_ALLOWED_TICKERS = 60

    def __init__(self):

        self.live_data = False

        if self.p.live_trading:
            signal.signal(signal.SIGINT, self.sigstop)
            self.resume()

        self.inds = {}
        for d in self.datas:
            self.inds[d] = {}
            self.inds[d]["distance"] = ZScore(d.close, period=self.p.std)
            self.inds[d]["pct"] = bt.indicators.PercentChange(d.close, period=self.p.pct)

    def prenext(self):
        self.next()

    def nextstart(self):
        self.next()

    def next(self):

        if self.p.live_trading and (not self.live_data):
            return  # prevent live trading with delayed data
        #else:
            #time.sleep(30) # sleep 30s to avoid over pulling

        # only look at data that existed last week
        available = list(filter(lambda d: len(d) > self.p.std + 2, self.datas))

        if len(available) > LongShortReturn.MIN_ALLOWED_TICKERS:
            print(f"{self.datetime.date(0)}: AVAILABLE TICKERS: {len(available)}")
        else:
            print(f"NOT ENOUGH TICKERS FOR THE STRATEGY, CURRENTLY {len(available)} TICKERS")
            return

        data = list()
        total = 0
        # print T+1 return
        for i, d in enumerate(available):
            pos = self.getposition(d)
            price = pos.price
            size = pos.size
            cur_price = d.close[0]
            pre_price = d.close[-1]
            pnl = (cur_price - pre_price) * size
            total += pnl
            if( size != 0 ):
                data.append([d._name, size, price, cur_price, pnl, total])

        headers = ['TICKER', 'Amount', 'Entry', 'Exit', 'PnL','Total' ]
        print(f"DAILY STRATEGY CALCULATION TABLE({self.datetime.date(0)}) :")
        print( tabulate(data, headers=headers, tablefmt="github") )
        

        self.manual_update_balance()

        rets = np.zeros(len(available))
        distances = np.zeros(len(available))
        weights = np.zeros(len(available))

        for i, d in enumerate(available):
            rets[i] = self.inds[d]['pct'][0]
            distances[i] = self.inds[d]['distance'][0]

        ret_scores = stats.zscore(rets)
        scores = (ret_scores - 2 * distances)

        # calculate top20/bottom20 percentile to short/long
        numbers_per_basket = int(np.floor(len(available) * 0.2))
        longs_index = min_n(scores, numbers_per_basket)
        shorts_index = max_n(scores, numbers_per_basket)
        selects = np.union1d(longs_index, shorts_index)

        # calculate weight: factor weighted
        sum_scores = np.sum(np.abs(scores[selects]))
        weights[selects] = [abs(scores[i]) / sum_scores for i in selects]

        # close position first
        for i, d in enumerate(available):
            if i not in longs_index and i not in shorts_index:
                self.close(d)

        # if stats.skew(ret_scores) < 0 :
        #     return # no trading today

        # rebalance position
        for i, d in enumerate(available):
            if i in longs_index:
                self.order_target_percent(d, target=0.5 / numbers_per_basket)
                #self.order_target_percent(d, target= weights[i] )

            if i in shorts_index:
                self.order_target_percent(d, target=-0.5 / numbers_per_basket)
                #self.order_target_percent(d, target= - weights[i] )

        # print out all scores and mark selected
        print(f"VARIANCE FOR A-2B/A/B: {stats.tstd(scores)}, {stats.tstd(rets)}, {stats.tstd(distances)}")
        print(f"SKEW FOR A-2B/A/B: {stats.skew(scores)},{stats.skew(ret_scores)}, {stats.skew(distances)}, ")
        print(f"KURTOSIS FOR A-2B/A/B: {stats.kurtosis(scores)}, {stats.kurtosis(ret_scores)}, {stats.kurtosis(distances)}")

        data = list()
        for i, d in enumerate(available):
            data.append( [ d._name, scores[i], ret_scores[i], distances[i],
                           'L' if i in longs_index else 'S' if i in shorts_index else '-'] )
        headers = ['TICKER', 'A-2B', 'A(Return)', 'B(Distance)', 'POSITION', ]
        print(f"DAILY STRATEGY CALCULATION TABLE({self.datetime.date(0)}) :")
        print( tabulate(data, headers=headers, tablefmt="github") )

    def manual_update_balance(self):
        ''' FOR LIVE TRADING,
         the broker would fetch account balance every bar,
         result in too many request per minute,
         so we manually update after order and each day.
         This is not the best way, but it is more economical
        '''
        if not self.p.live_trading:
            return

        self.broker.get_balance()

    def notify_data(self, data, status, *args, **kwargs):
        dn = data._name
        dt = datetime.now()
        msg = 'Data Status: {}, Order Status: {}'.format(data._getstatusname(status), status)
        print(f"{dt}, {dn}, {msg}")

        if data._getstatusname(status) == 'LIVE':
            self.live_data = True
        else:
            self.live_data = False

    def resume(self):

        if self.p.live_trading:
            self.broker.sync_exchange_positions(self.datas)
        # else:

    def sigstop(self, a, b):
        print('STOPPING BACKTRADER......')

        # close all position
        # print('CLOSING ALL POSITIONS......')
        # for d in self.datas:
        #     if self.getposition(d).size !=0:
        #         self.close(d)  # LOOSE ORDER

        time.sleep(5)
        self.env.runstop()


def file_backtest():
    dataset = DataSet(DataSet.BINANCE_FUTURE_201708_202203)
    start = '2021-01-01'
    end = '2022-03-30'

    cerebro = bt.Cerebro(stdstats=False,
                         runonce=False, )

    dataset.configure_file_data_backtest(start,
                                         end,
                                         cerebro,
                                         LongShortReturn,
                                         optimize=False,
                                         pct=1,
                                         std=20,
                                         )

    result = dataset.run_backtest()

    cerebro.plot(iplot=False)[0][0]

    return


def live_backtest():
    dataset = DataSet(DataSet.BINANCE_FUTURE_201708_202203)
    start = '2021-12-01'
    end = '2022-05-02'

    cerebro = bt.Cerebro(stdstats=False,
                         runonce=False,
                         )

    dataset.configure_live_data_backtest('future',
                                         start,
                                         end,
                                         cerebro,
                                         LongShortReturn,
                                         optimize=False,
                                         pct=2,
                                         std=20,
                                         )

    result = dataset.run_backtest()

    cerebro.plot(iplot=False)[0][0]

    return

def live_trading():

    # absolute dir the script is in
    script_dir = os.path.dirname(__file__)
    abs_file_path = os.path.join(script_dir, '../config/params-production-future-hou.json')
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
    fromdate = datetime.strptime('2022-04-01', '%Y-%m-%d')

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

    cerebro.addstrategy(LongShortReturn,
                        pct=2,
                        std=20,
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
    # live_backtest()
    # live_trading()