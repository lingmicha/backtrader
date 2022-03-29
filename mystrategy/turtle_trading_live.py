from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import argparse

import backtrader as bt
import time
from ccxtbt import CCXTStore
import os
import json
from datetime import datetime, timedelta
from turtle_trading import BinanceComissionInfo,Turtle
from send_email import AlertEmailer
import signal

class TurtleLive(Turtle):
    def __init__(self):

        self.live_data = False  #only live data to start trading
        self.broker.get_balance()
        signal.signal(signal.SIGINT, self.sigstop)

        super(TurtleLive, self).__init__()

    def next(self):
        if self.live_data:
            super().next()

    def notify_data(self, data, status, *args, **kwargs):
        dn = data._name
        dt = datetime.now()
        msg = 'Data Status: {}, Order Status: {}'.format(data._getstatusname(status), status)
        print(f"{dt}, {dn}, {msg}")

        if data._getstatusname(status) == 'LIVE':
            self.live_data = True
        else:
            self.live_data = False

    def notify_order(self, order):

        super().notify_order(order)

        order_side = "Buy" if order.isbuy() else "Sell"
        if order.status == order.Completed:
            msg =(
                    f"{order_side} Order Completed - {self.datas[0].datetime.datetime(0)} "
                    f"Name: {order.data._name} "
                    f"Size: {order.executed.size} "
                    f"@Price: {order.executed.price} "
                    f"N Value: {self.N} "
                    f"Remaining Cash: {self.broker.getcash()} "
                    f"Stop Loss: {self.stop_loss} "
                    f"Position Size: {self.position.size} "
                    f"Position Price: {self.position.price} "
            )

        elif order.status in {order.Canceled, order.Margin, order.Rejected}:
            msg =(
                    f"{order_side} Order Canceled/Margin/Rejected"
                    f"Name: {order.data._name} "
                    f"Size: {order.executed.size} "
                    f"@Price: {order.executed.price} "
                    f"N Value: {self.N} "
                    f"Remaining Cash: {self.broker.getcash()} "
                    f"Stop Loss: {self.stop_loss} "
                    f"Position Size: {self.position.size} "
                    f"Position Price: {self.position.price} "
                )

        AlertEmailer.getInstance().send_email_alert(msg)

    def sigstop(self, a ,b):
        print('Stopping Backtrader')

        # print out position info and parameter info
        msg = (
            f"Ending Program: "
            f"N Value: {self.N} "
            f"Remaining Cash: {self.broker.getcash()} "
            f"Stop Loss: {self.stop_loss} "
            f"Position Size: {self.position.size} "
            f"Position Price: {self.position.price} "
        )

        AlertEmailer.getInstance().send_email_alert(msg)

        self.env.runstop()

def run_strategy():

    # absolute dir the script is in
    script_dir = os.path.dirname(__file__)
    abs_file_path = os.path.join(script_dir, 'params-sandbox-future.json')
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

    mailer.send_email_alert("this is a test")

    cerebro = bt.Cerebro(quicknotify=True,
                         exactbars=True)

    # Create our store
    config = {'apiKey': params["binance"]["apikey"],
              'secret': params["binance"]["secret"],
              'enableRateLimit': True,
              'options': {
                  'defaultType': 'future',
              },
              'nonce': lambda: str(int(time.time() * 1000)),
              }

    store = CCXTStore(exchange='binance', currency='USDT', config=config, retries=5, debug=False, sandbox=True)

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
    cerebro.addstrategy(TurtleLive)

    # Get our data
    # Drop newest will prevent us from loading partial data from incomplete candles

    hist_start_date = datetime.utcnow() - timedelta(days=21)
    data = store.getdata(dataname='PEOPLE/USDT', name="PEOPLEUSDT",
                         timeframe=bt.TimeFrame.Minutes, fromdate=hist_start_date,
                         compression=1, ohlcv_limit=10000, drop_newest=True)  # , historical=True)

    # Add the feed
    cerebro.adddata(data)
    cerebro.resampledata(data, timeframe=bt.TimeFrame.Days)

    # Add a Commission and Support Fractional Size
    # cerebro.broker.addcommissioninfo(BinanceComissionInfo())

    # Add Oberserver
    # cerebro.addobserver(bt.observers.DrawDown)
    # cerebro.addobserver(bt.observers.DrawDown_Old)

    # Add Analyzer

    # # Add Daily SharpeRatio
    # cerebro.addanalyzer(
    #     bt.analyzers.SharpeRatio,
    #     timeframe=bt.TimeFrame.Days,
    #     riskfreerate=0,
    #     _name='dailysharp'
    # )
    #
    # cerebro.addanalyzer(
    #     bt.analyzers.TimeReturn,
    #     timeframe=bt.TimeFrame.NoTimeFrame,
    #     _name='alltimereturn'
    # )

    # Run Here
    results = cerebro.run()

    # st0 = results[0]
    # log.info(
    #     (
    #         f"Sharp Ratio: {str(st0.analyzers.dailysharp.get_analysis())}, "
    #         f"All Time Return: {str(st0.analyzers.alltimereturn.get_analysis())}"
    #     )
    # )

if __name__ == '__main__':

    run_strategy()

