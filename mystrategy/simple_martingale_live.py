from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import backtrader as bt
import time

from ccxtbt import CCXTStore
import os
import json
from datetime import datetime, timedelta

from simple_martingale import Martingale
import atexit

import logging
log = logging.getLogger("strategy")


class MartingaleLive(Martingale):
    def __init__(self):

        self.live_data = False  #only live data to start trading
        self.broker.get_balance()
        super(MartingaleLive, self).__init__()

    def next(self):
        self.broker.get_balance()
        if self.live_data:
            super().next()

    def notify_data(self, data, status, *args, **kwargs):
        dn = data._name
        dt = datetime.now()
        msg = 'Data Status: {}, Order Status: {}'.format(data._getstatusname(status), status)
        log.info(f"{dt}, {dn}, {msg}")
        if data._getstatusname(status) == 'LIVE':
            self.live_data = True
        else:
            self.live_data = False



def OnExitApp(cerebro):
    log.info("Exit Live Trading Sessiong, Do proper cleanup...")

    st0 = cerebro.runningstrats[0]
    log.info(
        (
            f"Sharp Ratio: {str(st0.analyzers.dailysharp.get_analysis())}, "
            f"All Time Return: {str(st0.analyzers.alltimereturn.get_analysis())}"
        )
    )

def run_strategy():

    # absolute dir the script is in
    script_dir = os.path.dirname(__file__)
    abs_file_path = os.path.join(script_dir, 'params-sandbox.json')
    with open(abs_file_path, 'r') as f:
        params = json.load(f)

    cerebro = bt.Cerebro(quicknotify=True)

    # Create our store
    config = {'apiKey': params["binance"]["apikey"],
              'secret': params["binance"]["secret"],
              'enableRateLimit': True,
              'nonce': lambda: str(int(time.time() * 1000)),
              }

    store = CCXTStore(exchange='binance', currency='USDT', config=config, retries=5, debug=True, sandbox=True)

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
    # cerebro.broker.setcash(10000.0)

    # Add the strategy
    cerebro.addstrategy(MartingaleLive)

    # Get our data
    # Drop newest will prevent us from loading partial data from incomplete candles

    hist_start_date = datetime.utcnow() - timedelta(minutes=5)
    data = store.getdata(dataname='BNB/USDT', name="BNBUSDT",
                         timeframe=bt.TimeFrame.Minutes, fromdate=hist_start_date,
                         compression=1, ohlcv_limit=10000, drop_newest=True)  # , historical=True)

    # Add the feed
    cerebro.adddata(data)
    #cerebro.resampledata(data, timeframe=bt.TimeFrame.Days)

    # Add a Commission and Support Fractional Size
    # cerebro.broker.addcommissioninfo(BinanceComissionInfo())

    # Add Oberserver
    # cerebro.addobserver(bt.observers.DrawDown)
    # cerebro.addobserver(bt.observers.DrawDown_Old)

    # Add Analyzer

    # Add Daily SharpeRatio
    cerebro.addanalyzer(
        bt.analyzers.SharpeRatio,
        timeframe=bt.TimeFrame.Days,
        riskfreerate=0,
        _name='dailysharp'
    )

    cerebro.addanalyzer(
        bt.analyzers.TimeReturn,
        timeframe=bt.TimeFrame.NoTimeFrame,
        _name='alltimereturn'
    )

    # add exit handler
    atexit.register(OnExitApp, cerebro=cerebro)

    # Run Here
    cerebro.run()


if __name__ == '__main__':

    # config in main, and getLogger in each module
    import logging.config
    logging.config.fileConfig(fname='../config/log.conf')

    run_strategy()

