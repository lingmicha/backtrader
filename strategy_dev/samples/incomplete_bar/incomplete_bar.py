import json

import backtrader as bt
import pandas as pd
from datetime import datetime, timedelta
import time
from ccxtbt import CCXTStore


class LiveStrategy(bt.Strategy):
    '''
    Test a incomplete bar in real-time feed
    Can a incomplete bar for day's bar used as a minute bar??

    Retrieve a minute bar and form a hour bar
    A minute bar would trigger next multiple times and only one hour bar would deliver
    '''

    def __init__(self):
        self.add_timer(
            when=bt.timer.SESSION_START,
            # offset=self.p.offset,
            repeat=timedelta(minutes=5),
            # weekdays=self.p.weekdays,
        )

    def next(self):
        print('strategy next with utc:{}, data0:{}, data1:{}'.
              format(datetime.utcnow(),
                     self.datas[0].datetime.datetime(0),
                     self.datas[1].datetime.datetime(0)))

    def notify_timer(self, timer, when, *args, **kwargs):
        print('strategy notify_timer with tid {}, when {}'.
              format(timer.p.tid, when, ))

    def notify_data(self, data, status, *args, **kwargs):
        dn = data._name
        dt = datetime.now()
        msg = 'Data Status: {}, Order Status: {}'.format(data._getstatusname(status), status)
        print(f"{dt}, {dn}, {msg}")

        if data._getstatusname(status) == 'LIVE':
            self.live_data = True
        else:
            self.live_data = False


if __name__ == "__main__":
    # absolute dir the script is in
    config_path = '../../config'
    market = 'future'
    config_file = f"{config_path}/params-production-{market}.json"
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
    fromdate = datetime.strptime('2022-04-22', '%Y-%m-%d')
    todate = datetime.strptime('2022-03-29', '%Y-%m-%d')

    cerebro = bt.Cerebro(exactbars=True)
    cerebro.broker.setcash(10000)
    cerebro.broker.set_coc(True)

    ticker = 'BTC'
    data = store.getdata(dataname=f"{ticker}/USDT", name=ticker,
                         timeframe=bt.TimeFrame.Minutes,
                         fromdate=fromdate,
                         # todate=todate,
                         compression=1,
                         ohlcv_limit=10,
                         drop_newest=True,
                         historical=False,
                         qcheck=30,)
    cerebro.adddata(data)
    cerebro.resampledata(data,
                         #rightedge=False,
                         #boundoff=1,
                         timeframe=bt.TimeFrame.Minutes, compression=5)
    data.plotinfo.plot = False

    cerebro.addstrategy(LiveStrategy)
    cerebro.run()
