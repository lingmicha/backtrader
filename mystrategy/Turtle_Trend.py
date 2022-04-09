import json
import os

import backtrader as bt
import pandas as pd
from datetime import datetime
import time
from ccxtbt import CCXTStore
from send_email import AlertEmailer
import signal

'''
    Revised Turle Trading Algorithm

    Not in Market (Long):
        (1) Market Regime Filter: BTC-90SMA ( Don't Open New Position )
        (2) Enter Signal: 20day Highest
            - Position Sizing = 1% Portfolio Value/ ATR(20)
            - Stop Loss = Executed Price - 2N
            - Next Add = Executed Price + 0.5N
    In Market (Long):
        (1) Check Stop Loss -> met, close
        (2) Check Exit: 10day Lowest -> met, close
        (3) Check Add:
            - Previous Executed Price + 0.5N
            - 1 Unit = 1% Portfolio Value/ ATR(20)
            - Max: 4 Unit
            - Stop Loss:  Foreach Position, Executed Price - 2N
            - Update Next Add: Executed Price + 0.5N
'''


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


class TurtleUnit:

    def __init__(self, data, strategy):

        self.data = data
        self.strategy = strategy

        self.direction = None
        self.target_price = None
        self.target_size = None
        self.N = None

        self.executed_price = None
        self.executed_size = None
        self.stop = None
        self.next = None

        self.open_order = None
        self.close_order = None

    def __str__(self):
        msg = []
        msg += [f"TURTLE POSITION: {self.data._name}, "]
        msg += [f"DIRECTION: {self.direction}, "]
        msg += [f"PRICE: {self.executed_price}, "]
        msg += [f"SIZE: {self.executed_size}, "]
        msg += [f"N: {self.N}, "]
        msg += [f"STOP: {self.stop}, "]
        msg += [f"NEXT: {self.next}"]
        return "".join(msg)

    def calc_size(self, target_price, N):

        value = self.strategy.broker.get_value()
        cash = self.strategy.broker.get_cash()

        target_size = value * 0.01 / N

        # TODO: More Sophisticated Check
        if cash - abs(target_price * target_size) <= 20:
            print(f"CASH NOT ENOUGH TO OPEN ORDER: {self.data._name}, "
                  f"TARGET_PRICE: {target_price:2f}, "
                  f"TARGET_SIZE: {target_size:2f}, "
                  f"Cash: {cash:2f}")
            return 0

        return target_size

    def open(self, direction, target_price, N):

        target_size = self.calc_size(target_price, N)

        if target_size == 0:
            return False

        self.direction = direction
        self.target_size = target_size
        self.target_price = target_price
        self.N = N

        if self.direction == 'long':
            self.open_order = self.strategy.buy(self.data, size=self.target_size)
        else:
            self.open_order = self.strategy.sell(self.data, size=self.target_size)

        return True

    def close(self):
        self.close_order = self.strategy.close(self.data, size=self.executed_size)
        return True

    def stop_loss(self):
        if self.direction == 'long' and (self.stop >= self.data.low):
            self.close_order = self.strategy.close(self.data, size=self.executed_size)
            return True
        if self.direction == 'short' and (self.stop <= self.data.high):
            self.close_order = self.strategy.close(self.data, size=self.executed_size)
            return True
        return False

    def process_open(self, open_order):
        self.executed_price = open_order.executed.price
        self.executed_size = abs(open_order.executed.size)
        if self.direction == 'long':
            self.stop = self.executed_price - 2 * self.N
            self.next = max(self.executed_price, self.target_price) + 0.5 * self.N
        else:
            self.stop = self.executed_price + 2 * self.N
            self.next = min(self.executed_price, self.target_price) - 0.5 * self.N
        # need to update other stops as well

    def match_open_order(self, open_order):
        if self.open_order == open_order:
            return True
        return False

    def match_close_order(self, close_order):
        if self.close_order == close_order:
            return True
        return False


class TurtleTrend(bt.Strategy):
    params = (
        # ('index_sma', 90),
        ('breakout_window', 20 ),  # 28800 min - 20day
        ('exit_window', 10 ),  # 14400 min - 10day
        ('atr', 20 ),  # 28800 min - 20day
        ('max_units', 4),
        ('debug', False),
        ('live_feed', False),
        ('live_trading', False),
    )

    def __init__(self):

        self.live_data = False
        if self.p.live_feed or self.p.live_trading:
            signal.signal(signal.SIGINT, self.sigstop)

        self.index = self.datas[0]
        # self.index_sma = bt.indicators.SimpleMovingAverage(self.index, period=self.p.index_sma)

        #self.cryptos = self.datas[1:]
        self.cryptos = self.datas[1::2]
        self.cryptos_daybar = self.datas[2::2]

        self.inds = {}
        for i, d in enumerate(self.cryptos):
            self.inds[d] = {}

            # self.inds[d]["breakout_long"] = bt.indicators.Highest(d.high, period=self.p.breakout_window)
            # self.inds[d]["exit_long"] = bt.indicators.Lowest(d.low, period=self.p.exit_window)
            #
            # self.inds[d]["breakout_short"] = bt.indicators.Lowest(d.low, period=self.p.breakout_window)
            # self.inds[d]["exit_short"] = bt.indicators.Highest(d.high, period=self.p.exit_window)
            #
            # self.inds[d]["atr"] = bt.indicators.ATR(d, period=self.p.atr)

            self.inds[d]["breakout_long"] = bt.indicators.Highest(self.cryptos_daybar[i].high, period=self.p.breakout_window, plot=False)
            self.inds[d]["exit_long"] = bt.indicators.Lowest(self.cryptos_daybar[i].low, period=self.p.exit_window, plot=False)

            self.inds[d]["breakout_short"] = bt.indicators.Lowest(self.cryptos_daybar[i].low, period=self.p.breakout_window, plot=False)
            self.inds[d]["exit_short"] = bt.indicators.Highest(self.cryptos_daybar[i].high, period=self.p.exit_window, plot=False)

            self.inds[d]["atr"] = bt.indicators.ATR(self.cryptos_daybar[i], period=self.p.atr, plot=False)

        self.sigs = {}
        for d in self.cryptos:
            self.sigs[d] = {}
            # TODO: DONOT FORGET A LINE COUPLER BETWEEN TWO TIMEFRAME COMPARE!!!
            # CEREBRO RUNONCE = FALSE
            # REF: https://www.backtrader.com/docu/mixing-timeframes/indicators-mixing-timeframes
            # REF: https://www.backtrader.com/blog/posts/2016-05-05-indicators-mixing-timeframes/indicators-mixing-timeframes/
            self.sigs[d]["breakout_long"] = d.high >= self.inds[d]["breakout_long"].highest()
            self.sigs[d]["exit_long"] = d.low <= self.inds[d]['exit_long'].lowest()

            self.sigs[d]["breakout_short"] = d.low <= self.inds[d]["breakout_short"].lowest()
            self.sigs[d]["exit_short"] = d.high >= self.inds[d]['exit_short'].highest()

        self.crypto_pos = {}
        for d in self.cryptos:
            self.crypto_pos[d] = []

        self.pending_orders = []

    def prenext(self):
        self.next()

    def nextstart(self):
        self.next()

    def next(self):

        if self.p.live_trading:
            # do live trading specifics
            self.broker.get_balance()  # otherwise cash and value not updated

        if self.p.debug and ( len(self) % 1440 == 0 ):
            # print every hour for miniute bar
            print(f"NEXT-DATA0: {self.data.datetime.datetime(0):%Y-%m-%d %H:%M:%S}, "
                  # f"HIGH: {self.data.high[0]}, "
                  # f"LOW: {self.data.low[0]}, "
                  # f"OPEN: {self.data.open[0]}, "
                  # f"CLOSE: {self.data.close[0]}, "
                  # f"RECENT HIGH: {self.inds[self.datas[1]]['breakout_long'][0]}, "
                  # f"RECENT LOW: {self.inds[self.datas[1]]['breakout_short'][0]}, "
                  f"STRATEGY-LEN: {len(self)}")

        if self.p.live_trading and (not self.live_data ):
            return # prevent live trading with delayed data

        # check len > 20day/28800min
        candidates = list(filter(lambda d: len(d) > 28800, self.cryptos))

        '''
            (1) Separate Pending/Long/Short/Zero Positions
            (2) For Pending:
                    - Order alive, no activity, just print
                For Long:
                    - Check Exit, if met, return
                    - Check Stop Loss, if met, return
                    - Check Add Position, if met, return
                For Short:
                    - Same as Long
                For Zero:
                    - Check Breakout, if met, return
        '''
        # REMOVE TICKER W/ PENDING ORDERS:
        pending = list(set([o.p.data for o in self.pending_orders]))
        if len(pending) > 0:
            print(f"TICKERS W/ PENDING ORDERS, NOT PROCESSING: {[d._name for d in pending]}")
        candidates = list(filter(lambda d: d not in pending, candidates))

        if self.p.debug and  ( len(self) % 1440 == 0 ):
            for i, d in enumerate(candidates):
                if len(self.crypto_pos[d]) > 0:
                    poss = '\n'.join( [ str(x) for x in self.crypto_pos[d] ])
                    print(f" { poss }")

        # SEPRATE LONG/SHORT/NO POSITIONS:
        long = list(filter(lambda x: self.getposition(x).size > 1e-8, candidates))
        short = list(filter(lambda x: self.getposition(x).size < -1e-8, candidates))
        zero = list(filter(lambda x: abs(self.getposition(x).size) <= 1e-8, candidates))

        # PROCESS LONG POSITION:
        for i, d in enumerate(long):
            poss = self.crypto_pos[d]

            # check exit
            if self.sigs[d]['exit_long']:
                for pos in poss:
                    if pos.close():
                        self.pending_orders.append(pos.close_order)
                continue

            # check stop loss
            is_stop_loss = False
            for pos in poss:
                if pos.stop_loss():
                    is_stop_loss = True
                    self.pending_orders.append(pos.close_order)
            if is_stop_loss:
                continue

            # check add position
            if len(poss) < self.p.max_units:
                pos = poss[-1]
                if pos.next <= d.high:
                    new_pos = TurtleUnit(d, self)
                    if new_pos.open('long', pos.next, self.inds[d]['atr'][0]):
                        self.crypto_pos[d].append(new_pos)
                        self.pending_orders.append(new_pos.open_order)
        # END LONG POSITION

        # PROCESS SHORT POSITION:
        for i, d in enumerate(short):
            poss = self.crypto_pos[d]

            # check exit
            if self.sigs[d]['exit_short']:
                for pos in poss:
                    if pos.close():
                        self.pending_orders.append(pos.close_order)
                continue

            # check stop loss
            is_stop_loss = False
            for pos in poss:
                if pos.stop_loss():
                    is_stop_loss = True
                    self.pending_orders.append(pos.close_order)
            if is_stop_loss:
                continue

            # check add position
            if len(poss) < self.p.max_units:
                pos = poss[-1]
                if pos.next >= d.low:
                    new_pos = TurtleUnit(d, self)
                    if new_pos.open('short', pos.next, self.inds[d]['atr'][0]):
                        self.crypto_pos[d].append(new_pos)
                        self.pending_orders.append(new_pos.open_order)
        # END SHORT POSITION

        # TODO: rank breakout by some sort~
        # TODO: could use momentum to sort breakout
        # PROCESS MARKET ENTRY:
        for i, d in enumerate(zero):

            # TODO: find out what this means??
            if self.sigs[d]['breakout_long'] and self.sigs[d]['breakout_short']:
                # long and short triggered,do nothing!!!
                print('BREAKOUT SIGNAL INDICATES LONG AND SHORT, SKIP SIGNALS...')
                continue

            if self.sigs[d]['breakout_long']:
                pos = TurtleUnit(d, self)
                if pos.open('long', d.high[0], self.inds[d]['atr'][0]):
                    self.crypto_pos[d].append(pos)
                    self.pending_orders.append(pos.open_order)

            if self.sigs[d]['breakout_short']:
                pos = TurtleUnit(d, self)
                if pos.open('short', d.low[0], self.inds[d]['atr'][0]):
                    self.crypto_pos[d].append(pos)
                    self.pending_orders.append(pos.open_order)

    def process_order(self, order):
        if order.alive():
            return

        data = order.p.data
        # sanity check
        if order not in self.pending_orders:
            print(f"LOOSE ORDER: {data._name}")
            return

        if order.status in {order.Canceled, order.Margin, order.Rejected}:

            poss = self.crypto_pos[data]

            # Match Open Order ( Only Open Order One At A Time)
            if poss[-1].match_open_order(order):
                self.pending_orders.remove(order)
                self.crypto_pos[data].remove(poss[-1])

            # Match Close Order
            for pos in poss:
                if pos.match_close_order(order):
                    self.pending_orders.remove(order)
                    self.pos = None
                    break

        if order.status == order.Completed:
            poss = self.crypto_pos[data]

            # Match Open Order
            if poss[-1].match_open_order(order):
                poss[-1].process_open(order)
                direction = poss[-1].direction
                # update all stop loss
                if direction == 'long':
                    for pos in poss[0:-1]:
                        pos.stop += 0.5 * pos.N
                else:
                    for pos in poss[0:-1]:
                        pos.stop -= 0.5 * pos.N
                self.pending_orders.remove(order)

            # Match Close Order
            for pos in poss:
                if pos.match_close_order(order):
                    self.pending_orders.remove(order)
                    self.crypto_pos[data].remove(pos)
                    break

    def notify_order(self, order):
        """Execute when buy or sell is triggered
        Notify if order was accepted or rejected
        """

        self.process_order(order)

        if order.alive():
            if self.p.debug:
                print(f"{order.p.data._name} ORDER IS ALIVE: {self.datas[0].datetime.datetime(0)}")
            # submitted, accepted, partial, created
            # Returns if the order is in a status in which it can still be executed
            return

        order_side = "Buy" if order.isbuy() else "Sell"
        if order.status == order.Completed:
            msg = []
            msg += [f"{order_side} Order Completed - {self.datas[0].datetime.datetime(0)}, "]
            msg += [f"Ticker: {order.p.data._name}, "]
            msg += [f"Size: {order.executed.size}, "]
            msg += [f"@Price: {order.executed.price}, "]
            msg += [f"Value: {order.executed.value:.2f}, "]
            msg += [f"Comm: {order.executed.comm:.6f} "]
            print( "".join(msg))
            if self.p.live_trading:
                AlertEmailer.getInstance().send_email_alert("\n".join(msg))

        elif order.status in {order.Canceled, order.Margin, order.Rejected}:
            msg = []
            msg += [f"{order_side} Order Canceled/Margin/Rejected"]
            msg += [f"Ticker: {order.p.data._name} "]
            msg += [f"Size: {order.created.size} "]
            msg += [f"@Price: {order.created.price} "]
            msg += [f"Value: {order.created.value:.2f} "]
            msg += [f"Remaining Cash: {self.broker.getcash()}"]
            print( "".join(msg) )
            if self.p.live_trading:
                AlertEmailer.getInstance().send_email_alert("\n".join(msg))

    def notify_trade(self, trade):
        """Execute after each trade
        Calcuate Gross and Net Profit/loss"""

        # trade closed
        if trade.isclosed:
            print(
                f"OPERATIONAL PROFIT, GROSS: {trade.pnl:.2f}, Net: {trade.pnlcomm:.2f}"
            )

    def notify_data(self, data, status, *args, **kwargs):
        dn = data._name
        dt = datetime.now()
        msg = 'Data Status: {}, Order Status: {}'.format(data._getstatusname(status), status)
        print(f"{dt}, {dn}, {msg}")

        if data._getstatusname(status) == 'LIVE':
            self.live_data = True
        else:
            self.live_data = False

    def sigstop(self, a ,b):
        print('STOPPING BACKTRADER......')

        # close all position
        print('CLOSING ALL POSITIONS......')
        outstanding = list(filter( lambda d: self.getposition(d).size !=0, self.cryptos))
        for i, d in enumerate(outstanding):
            self.close(d) # LOOSE ORDER
        print(f'CLOSING POSITIONS FOR {[ d._name for d in outstanding]}')

        if self.p.live_trading:
            AlertEmailer.getInstance().send_email_alert("PROGRAM END")

        self.env.runstop()


def run_filefeed_backtest():

    data_path = '../../algo-trading/My_Stuff/notebook/Turtle_Trend/data'
    tickers = pd.read_csv(data_path + '/tickers.csv', header=None)[1].tolist()

    cerebro = bt.Cerebro(stdstats=False, preload=False, runonce=False)

    # TODO: DATA0 Must have the earilest start datetime
    fromdate = datetime.strptime('2020-01-01', '%Y-%m-%d')
    todate = datetime.strptime('2022-03-30', '%Y-%m-%d')

    bitcoin = 'BNB'
    df = pd.read_csv(f"{data_path}/{bitcoin}.csv",
                     parse_dates=True,
                     index_col=0)
    cerebro.adddata(bt.feeds.PandasData(dataname=df,
                                             name=bitcoin,
                                             fromdate=fromdate,
                                             todate=todate,
                                             timeframe=bt.TimeFrame.Minutes,
                                             plot=False),
                    )

    for ticker in ['BNB']: #tickers:
        df = pd.read_csv(f"{data_path}/{ticker}.csv",
                         parse_dates=True,
                         index_col=0)
        data= bt.feeds.PandasData(dataname=df,
                                  name=ticker,
                                  fromdate=fromdate,
                                  todate=todate,
                                  timeframe=bt.TimeFrame.Minutes,
                                  plot=False)
        cerebro.adddata(data)
        cerebro.resampledata(data, name=f"_{ticker}", timeframe=bt.TimeFrame.Days)


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

    cerebro.broker.setcash(10000)
    cerebro.broker.addcommissioninfo(BinanceComissionInfo())
    cerebro.addstrategy(TurtleTrend,
                        debug=True)

    results = cerebro.run()

    pyfoliozer = results[0].analyzers.getbyname('pyfolio')
    returns, positions, transactions, gross_lev = pyfoliozer.get_pf_items()
    returns.to_csv("returns.csv")
    positions.to_csv("positions.csv")
    transactions.to_csv("transactions.csv")

    print(f"Return: {list(results[0].analyzers.alltimereturn.get_analysis().values())[0]:.3f}")
    print(f"Sharpe: {results[0].analyzers.sharperatio.get_analysis()['sharperatio']:.3f}")
    print(f"Norm. Annual Return: {results[0].analyzers.returns.get_analysis()['rnorm100']:.2f}%")
    print(f"Max Drawdown: {results[0].analyzers.drawdown.get_analysis()['max']['drawdown']:.2f}%")

    cerebro.plot(iplot=False)[0][0]
    #cerebro.plot()


def run_livefeed_backtest():

    # absolute dir the script is in
    script_dir = os.path.dirname(__file__)
    abs_file_path = os.path.join(script_dir, 'params-production.json')
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

    mailer.send_email_alert("THIS IS A START OF LIVE DATA BACKTEST")

    cerebro = bt.Cerebro(exactbars=True, runonce=False, preload=False)

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

    data_path = 'data'
    tickers = pd.read_csv(data_path + '/tickers.csv', header=None)[1].tolist()

    # TODO: DATA0 Must have the earilest start datetime
    fromdate = datetime.strptime('2020-01-01', '%Y-%m-%d')
    todate = datetime.strptime('2022-03-30', '%Y-%m-%d')

    bitcoin = 'BTC'
    data = store.getdata(dataname=f"{bitcoin}/USDT", name=bitcoin,
                         timeframe=bt.TimeFrame.Minutes,
                         fromdate=fromdate,
                         todate=todate,
                         compression=1,
                         ohlcv_limit=10000,
                         drop_newest=True)  # , historical=True)
    cerebro.adddata(data)

    for ticker in tickers:
        data = store.getdata(dataname=f"{ticker}/USDT", name=ticker,
                             timeframe=bt.TimeFrame.Minutes,
                             fromdate=fromdate,
                             todate=todate,
                             compression=1,
                             ohlcv_limit=10000,
                             drop_newest=True)  # , historical=True)
        cerebro.adddata(data)
        cerebro.resampledata(data, name=f"_{ticker}", timeframe=bt.TimeFrame.Days)

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

    # TODO : Use Default Broker
    cerebro.broker.setcash(10000.0)
    cerebro.broker.addcommissioninfo(BinanceComissionInfo())
    # Add the strategy
    cerebro.addstrategy(TurtleTrend,
                        debug=True,
                        live_feed=True,
                        live_trading=False)
    # Run Here
    results = cerebro.run()

    pyfoliozer = results[0].analyzers.getbyname('pyfolio')
    returns, positions, transactions, gross_lev = pyfoliozer.get_pf_items()
    returns.to_csv("returns.csv")
    positions.to_csv("positions.csv")
    transactions.to_csv("transactions.csv")

    print(f"Return: {list(results[0].analyzers.alltimereturn.get_analysis().values())[0]:.3f}")
    print(f"Sharpe: {results[0].analyzers.sharperatio.get_analysis()['sharperatio']:.3f}")
    print(f"Norm. Annual Return: {results[0].analyzers.returns.get_analysis()['rnorm100']:.2f}%")
    print(f"Max Drawdown: {results[0].analyzers.drawdown.get_analysis()['max']['drawdown']:.2f}%")

    cerebro.plot(iplot=False)[0][0]
    # cerebro.plot()


def run_live_trading():

    # absolute dir the script is in
    script_dir = os.path.dirname(__file__)
    abs_file_path = os.path.join(script_dir, 'params-production.json')
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

    #============LOAD DATAFEEDS=========================
    data_path = 'data'
    tickers = pd.read_csv(data_path + '/tickers.csv', header=None)[1].tolist()

    # TODO: DATA0 Must have the earilest start datetime
    fromdate = datetime.strptime('2022-03-01', '%Y-%m-%d')
    todate = datetime.strptime('2022-03-30', '%Y-%m-%d')

    bitcoin = 'BTC'
    data = store.getdata(dataname=f"{bitcoin}/USDT", name=bitcoin,
                         timeframe=bt.TimeFrame.Minutes,
                         fromdate=fromdate,
                         #todate=todate,
                         compression=1,
                         ohlcv_limit=10000,
                         drop_newest=True)  # , historical=True)
    cerebro.adddata(data)

    for ticker in tickers:
        data = store.getdata(dataname=f"{ticker}/USDT", name=ticker,
                             timeframe=bt.TimeFrame.Minutes,
                             fromdate=fromdate,
                             #todate=todate,
                             compression=1,
                             ohlcv_limit=10000,
                             drop_newest=True)  # , historical=True)
        cerebro.adddata(data)
        cerebro.resampledata(data, name=f"_{ticker}", timeframe=bt.TimeFrame.Days)

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

    # Add the strategy
    cerebro.addstrategy(TurtleTrend,
                        debug=True,
                        live_feed=True,
                        live_trading=True)
    # Run Here
    results = cerebro.run()

    pyfoliozer = results[0].analyzers.getbyname('pyfolio')
    returns, positions, transactions, gross_lev = pyfoliozer.get_pf_items()
    returns.to_csv("returns.csv")
    positions.to_csv("positions.csv")
    transactions.to_csv("transactions.csv")

    print(f"Return: {list(results[0].analyzers.alltimereturn.get_analysis().values())[0]:.3f}")
    print(f"Sharpe: {results[0].analyzers.sharperatio.get_analysis()['sharperatio']:.3f}")
    print(f"Norm. Annual Return: {results[0].analyzers.returns.get_analysis()['rnorm100']:.2f}%")
    print(f"Max Drawdown: {results[0].analyzers.drawdown.get_analysis()['max']['drawdown']:.2f}%")

    cerebro.plot(iplot=False)[0][0]
    # cerebro.plot()


if __name__ == "__main__":

    '''TODO: LIVE TRADING CHECKLIST:
        (1) params: set on days
        (2) strategy: in next(), check len(self)， so that prenext/nextstart won't error
        (3) datafeeed: fromdate, enough time for backfill data
        (4) p.live_feed = True p.live_trading = True
        (5) check ticker.csv, which ticker to include
        (6) data should feed in min, and resample a day data
        (7) use line coupler when comparing trading signals
    '''
    run_filefeed_backtest()
    #run_livefeed_backtest()
    #run_live_trading()