import signal
import time

import backtrader as bt
from tabulate import tabulate
from datetime import datetime

class CryptoStrategy(bt.Strategy):
    '''
        subclass bt.Strategy to implement crypto trading specifics
    '''

    params = (
        ('debug', False),
        ('live_trading', False),
    )

    def __init__(self):
        self.closed_trades = list()
        self.closed_orders = list()
        self.live_data = False

        if self.p.live_trading:
            signal.signal(signal.SIGINT, self.sigstop)
            self.resume()
            self.print_position_table()

    def print_daily_pnl_table(self):

        data = list()
        headers = ['DATE',
                   'TICKER',
                   'ENTRY_PRICE',
                   'AMOUNT',
                   'ENTRY_VALUE',
                   'T-1_VALUE',
                   'T-0_VALUE',
                   'DAILY_PNL',
                   ]

        for i, d in enumerate(self.datas):
            # compare the close[0] and close[-1] price, calculate
            if len(d) < 2:
                continue

            size = self.getposition(d).size
            price = self.getposition(d).price

            if size == 0:
                continue

            data.append(
                [d.datetime.date(0),
                 d._name,
                 price,
                 size,
                 price * size,
                 size * d.close[-1],
                 size * d.close[0],
                 size * d.close[0] - size * d.close[-1]
                 ])

        print(f"{self.datas[0].datetime.datetime(0)} DAILY PNL TABLE:")
        print( tabulate(data, headers=headers, tablefmt="github") )

    def print_position_table(self):

        data = list()
        headers = ['DATE',
                   'TICKER',
                   'ENTRY_PRICE',
                   'AMOUNT',
                   'ENTRY_VALUE',
                   ]

        for i, d in enumerate(self.datas):
            # compare the close[0] and close[-1] price, calculate
            if len(d) == 0:
                dt = datetime.utcnow()
            else:
                dt = d.datetime.date(0)

            size = self.getposition(d).size
            price = self.getposition(d).price

            if size == 0:
                continue

            data.append(
                [dt,
                 d._name,
                 price,
                 size,
                 price * size,
                 ])

        print(f"{dt} POSITION TABLE:")
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

    def print_closed_trade(self):
        # this function is used in conjucation with notify_trade
        if self.p.debug and len(self.closed_trades)>0:
            print(f"{self.datas[0].datetime.datetime(0)} T-1 TRADE TABLE:")
            print(tabulate(self.closed_trades, headers='keys', tablefmt='github'))
            self.closed_trades.clear()

    def print_closed_order(self):
        if self.p.debug and len(self.closed_orders)>0:
            print(f"{self.datas[0].datetime.datetime(0)} T-1 ORDER TABLE:")
            print(tabulate(self.closed_orders, headers='keys', tablefmt='github'))
            self.closed_orders.clear()

    def print_period_stats(self):
        #self.print_daily_pnl_table()
        self.print_closed_order()
        #self.print_closed_trade()

    def notify_trade(self, trade):
        """Execute after each trade
        Calcuate Gross and Net Profit/loss"""

        # trade closed
        if trade.isclosed and self.p.debug:
            dt = datetime.utcnow() \
                if self.cerebro.p.quicknotify \
                else trade.data.datetime.datetime(0)
            t = {
                'DATETIME': dt,
                'TICKER': trade.data._name,
                'GROSS_PNL': trade.pnl,
                'NET_PNL': trade.pnlcomm,
            }
            self.closed_trades.append(t)

    def notify_order(self, order):
        """Execute when buy or sell is triggered
        Notify if order was accepted or rejected
        """

        if order.alive():
            # if self.p.debug:
            #     print(f"{order.p.data._name} ORDER IS ALIVE: {self.datas[0].datetime.datetime(0)}")
            # submitted, accepted, partial, created
            # Returns if the order is in a status in which it can still be executed
            return

        order_side = "Buy" if order.isbuy() else "Sell"
        if order.status == order.Completed:
            dt = datetime.utcnow() \
                if self.cerebro.p.quicknotify \
                else order.p.data.datetime.datetime(0)
            o = {
                'DATETIME': dt,
                'TICKER': order.p.data._name,
                'STATUS': 'COMPLETED',
                'AMOUNT': order.executed.size,
                'PRICE': order.executed.price,
                'COMM': order.executed.comm,
            }
            self.closed_orders.append(o)

        elif order.status in {order.Canceled, order.Margin, order.Rejected}:
            dt = datetime.utcnow() \
                if self.cerebro.p.quicknotify \
                else order.p.data.datetime.datetime(0)
            o = {
                'DATETIME': dt,
                'TICKER': order.p.data._name,
                'STATUS': 'Margin' if order.status == 7 else 'Rejected',
                'AMOUNT': order.created.size,
                'PRICE': order.created.price,
                'COMM': order.created.comm,
            }
            self.closed_orders.append(o)

        self.manual_update_balance()

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
        #else:

    def sigstop(self, a, b):
        print('STOPPING BACKTRADER......')

        # close all position
        # print('CLOSING ALL POSITIONS......')
        # for d in self.datas:
        #     if self.getposition(d).size !=0:
        #         self.close(d)  # LOOSE ORDER

        time.sleep(5)
        self.env.runstop()
