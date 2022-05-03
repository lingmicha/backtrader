import json
import os
import time
from datetime import datetime, timedelta

import backtrader as bt
from backtrader import Order

from ccxtbt import CCXTStore

from strategy_dev import BinancePerpetualFutureCommInfo


class TestStrategy(bt.Strategy):

    def __init__(self):

        self.bought = False
        # To keep track of pending orders and buy price/commission
        self.order = None

    def next(self):


        if self.live_data and not self.bought:

            self.broker.get_balance()
            cash = self.broker.getcash()
            value = self.broker.getvalue(datas=self.datas)
            positions = self.broker.getposition(data=self.data0)
            print(
                f" Current Cash: {cash} ",
                f" Current Value:{value} "
                #f" Position Value:{(pos.size, pos.price) for pos in positions}"
            )

            # Buy
            self.order = self.buy(size=2) # below min size


            if self.order != None:
                self.bought = True

        for data in self.datas:
            print('{} - {} | O: {} H: {} L: {} C: {} V:{}'.format(data.datetime.datetime(),
                                                                  data._name, data.open[0], data.high[0], data.low[0],
                                                                  data.close[0], data.volume[0]))

    def notify_data(self, data, status, *args, **kwargs):
        dn = data._name
        dt = datetime.now()
        msg = 'Data Status: {}, Order Status: {}'.format(data._getstatusname(status), status)
        print(dt, dn, msg)
        if data._getstatusname(status) == 'LIVE':
            self.live_data = True
        else:
            self.live_data = False

    def notify_order(self, order):
        """Execute when buy or sell is triggered
        Notify if order was accepted or rejected
        """
        if order.alive():
            if self.p.debug:
                print(f"Order is alive: {self.datas[0].datetime.datetime(0)}")
                # submitted, accepted, partial, created
                # Returns if the order is in a status in which it can still be executed
                return

        order_side = "Buy" if order.isbuy() else "Sell"
        if order.status == order.Completed:
            print(
                (
                    f"{order_side} Order Completed - {self.datas[0].datetime.datetime(0)} "
                    f"Size: {order.executed.size} "
                    f"@Price: {order.executed.price} "
                    f"Value: {order.executed.value:.2f} "
                    f"Comm: {order.executed.comm:.6f} "
                )
            )

        elif order.status in {order.Canceled, order.Margin, order.Rejected}:
            print(
                (
                    f"{order_side} Order Canceled/Margin/Rejected"
                    f"Size: {order.created.size} "
                    f"@Price: {order.created.price} "
                    f"Value: {order.created.value:.2f} "
                    f"N Value: {self.N} "
                    f"Remaining Cash: {self.broker.getcash()}"
                )
            )

        self.order = None  # indicate no order pending

    def start(self):
        # cash = self.broker.getcash()
        # value = self.broker.getvalue(datas=self.datas)
        #
        # print(
        #     f" Initial Cash: {cash} ",
        #     f" Initial Value:{value} "
        # )
        pass

    def stop(self):
        cash = self.broker.getcash()
        value = self.broker.getvalue(datas=self.datas)

        print(
            f" Ending Cash: {cash} ",
            f" Ending Value:{value} "
        )



def send_ill_orders():

    # absolute dir the script is in
    script_dir = os.path.dirname(__file__)
    abs_file_path = os.path.join(script_dir, '../../config/params-production-future.json')
    with open(abs_file_path, 'r') as f:
        params = json.load(f)

    cerebro = bt.Cerebro(quicknotify=True)

    # Add the strategy
    cerebro.addstrategy(TestStrategy)

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


    # Get our data
    # Drop newest will prevent us from loading partial data from incomplete candles
    hist_start_date = datetime.utcnow() - timedelta(minutes=50)
    data = store.getdata(dataname='PEOPLE/USDT', name="PEOPLEUSDT",
                         timeframe=bt.TimeFrame.Minutes, fromdate=hist_start_date,
                         compression=1, ohlcv_limit=50, drop_newest=True)  # , historical=True)

    # Add the feed
    cerebro.adddata(data)

    # Run the strategy
    cerebro.run()


if __name__ == "__main__":
    send_ill_orders()