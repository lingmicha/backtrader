import json
import os
import random

import backtrader as bt
import pandas as pd
from datetime import datetime
import time
from ccxtbt import CCXTStore
from strategy_dev import AlertEmailer
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

    BETSIZE = 0.01

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

        target_size = value * TurtleUnit.BETSIZE / N

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


class TurtleTrendDaily(bt.Strategy):
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

        self.cryptos = self.datas[1:]

        self.inds = {}
        for i, d in enumerate(self.cryptos):
            self.inds[d] = {}

            self.inds[d]["breakout_long"] = bt.indicators.Highest(d.high, period=self.p.breakout_window, plot=False)
            self.inds[d]["exit_long"] = bt.indicators.Lowest(d.low, period=self.p.exit_window, plot=False)

            self.inds[d]["breakout_short"] = bt.indicators.Lowest(d.low, period=self.p.breakout_window, plot=False)
            self.inds[d]["exit_short"] = bt.indicators.Highest(d.high, period=self.p.exit_window, plot=False)

            self.inds[d]["atr"] = bt.indicators.ATR(d, period=self.p.atr, plot=False)

        self.sigs = {}
        for d in self.cryptos:
            self.sigs[d] = {}
            self.sigs[d]["breakout_long"] = d.high >= self.inds[d]["breakout_long"]
            self.sigs[d]["exit_long"] = d.low <= self.inds[d]['exit_long']

            self.sigs[d]["breakout_short"] = d.low <= self.inds[d]["breakout_short"]
            self.sigs[d]["exit_short"] = d.high >= self.inds[d]['exit_short']

        self.crypto_pos = {}
        for d in self.cryptos:
            self.crypto_pos[d] = []

        self.pending_orders = []

    def prenext(self):
        self.next()

    def nextstart(self):
        self.next()

    def next(self):

        # Print Every Day Bar
        if self.p.debug:
            print(f"NEXT-DATA0: {self.data.datetime.datetime(0):%Y-%m-%d}, "
                  f"STRATEGY-LEN: {len(self)}")

        if self.p.live_trading and (not self.live_data):
            return  # prevent live trading with delayed data

        self.manual_update_balance()

        # check len > 20day/28800min
        candidates = list(filter(lambda d: len(d) > 20, self.cryptos))

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

        if self.p.debug :
            for i, d in enumerate(candidates):
                if len(self.crypto_pos[d]) > 0:
                    poss = '\n'.join( [ str(x) for x in self.crypto_pos[d] ])
                    print(f" { poss }")

        # SEPRATE LONG/SHORT/NO POSITIONS:
        long = list(filter(lambda x: self.getposition(x).size > 1e-8, candidates))
        short = list(filter(lambda x: self.getposition(x).size < -1e-8, candidates))
        zero = list(filter(lambda x: abs(self.getposition(x).size) <= 1e-8, candidates))

      # START TURTLE
        self.process_long_position(long)
        self.process_short_position(short)
        self.process_enter_market(zero)
        # END TURTLE

    def process_long_position(self, long_position):
        # PROCESS LONG POSITION:
        for i, d in enumerate(long_position):
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

    def process_short_position(self, short_position):
        # PROCESS SHORT POSITION:
        for i, d in enumerate(short_position):
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

    def process_enter_market(self, zero_position):
        # TODO: rank breakout by some sort~
        # TODO: could use momentum to sort breakout
        # PROCESS MARKET ENTRY:
        for i, d in enumerate(zero_position):

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

    def manual_update_balance(self, force=False):
        ''' FOR LIVE TRADING,
         the broker would fetch account balance every bar,
         result in too many request per minute,
         so we mannually update after order and each day.
         This is not the best way, but it is more economical
        '''
        if not self.p.live_trading:
            return

        if force:
            self.broker.get_balance()

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

        self.manual_update_balance(force=True)


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

    data_path = '../../../algo-trading/My_Stuff/notebook/Turtle_Trend/data'
    tickers = pd.read_csv(data_path + '/tickers.csv', header=None)[1].tolist()

    cerebro = bt.Cerebro(stdstats=False, preload=False, runonce=False)

    # TODO: DATA0 Must have the earilest start datetime
    fromdate = datetime.strptime('2020-01-01', '%Y-%m-%d')
    todate = datetime.strptime('2022-03-30', '%Y-%m-%d')

    bitcoin = 'AVAX'
    df = pd.read_csv(f"{data_path}/{bitcoin}.csv",
                     parse_dates=True,
                     index_col=0)
    cerebro.resampledata(bt.feeds.PandasData(dataname=df,
                                             name=bitcoin,
                                             fromdate=fromdate,
                                             todate=todate,
                                             timeframe=bt.TimeFrame.Minutes,
                                             plot=False),
                         timeframe=bt.TimeFrame.Days
                    )

    for ticker in ['AVAX']: #tickers:
        df = pd.read_csv(f"{data_path}/{ticker}.csv",
                         parse_dates=True,
                         index_col=0)
        data= bt.feeds.PandasData(dataname=df,
                                  name=ticker,
                                  fromdate=fromdate,
                                  todate=todate,
                                  timeframe=bt.TimeFrame.Minutes,
                                  plot=False)
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
    cerebro.addstrategy(TurtleTrendDaily,
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

    cerebro.plot(iplot=False)
    #cerebro.plot()


def run_livefeed_backtest():

    # absolute dir the script is in
    script_dir = os.path.dirname(__file__)
    abs_file_path = os.path.join(script_dir, '../config/params-production-spot.json')
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

    cerebro = bt.Cerebro(stdstats=False, runonce=False, preload=False)

    # Create our store
    config = {'apiKey': params["binance"]["apikey"],
              'secret': params["binance"]["secret"],
              'enableRateLimit': True,
              # 'options': {
              #    'defaultType': 'future',
              # },
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
                         timeframe=bt.TimeFrame.Days,
                         fromdate=fromdate,
                         todate=todate,
                         compression=1,
                         ohlcv_limit=10000,
                         drop_newest=True)  # , historical=True)
    cerebro.adddata(data)
    data.plotinfo.plot = False

    for ticker in tickers:
        data = store.getdata(dataname=f"{ticker}/USDT", name=ticker,
                             timeframe=bt.TimeFrame.Days,
                             fromdate=fromdate,
                             todate=todate,
                             compression=1,
                             ohlcv_limit=10000,
                             drop_newest=True)  # , historical=True)
        cerebro.adddata(data)
        data.plotinfo.plot=False

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
    cerebro.addstrategy(TurtleTrendDaily,
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

    cerebro.plot()


def run_random_three():
    '''
    this experiment selects 3 random coins
    whose ICO date before 2020-01-01
    and runs the turtle trend algorithm
    between 2020-01-01 to 2022-03-30 on them.

    the result outputs each combination
    versus return curve for analysis
    '''

    tickers_file = 'tickers_20200101.csv'
    tickers = pd.read_csv(f"data/{tickers_file}", header=None)[1].to_list()
    print(f"TOTAL {len(tickers)} CRYPTOS ON BINANCE BEFORE 2020-01-01")

    combos = set()
    while True:
        combos.add( tuple(random.sample(tickers,3)) )
        if len(combos) == 3:
            break
    print(f"TOTAL NUMBER OF DIFFERENT CRYPTO COMBINATION: {len(combos)}")

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
    todate = datetime.strptime('2022-03-30', '%Y-%m-%d')


    results = []
    for combo in combos:
        # RUN EACH COMBO A BACKTEST

        cerebro = bt.Cerebro(stdstats=False, runonce=False, preload=False)

        bitcoin = 'BTC'
        data = store.getdata(dataname=f"{bitcoin}/USDT", name=bitcoin,
                             timeframe=bt.TimeFrame.Days,
                             fromdate=fromdate,
                             todate=todate,
                             compression=1,
                             ohlcv_limit=10000,
                             drop_newest=True)  # , historical=True)
        cerebro.adddata(data)
        data.plotinfo.plot = False

        for ticker in combo:

            data = store.getdata(dataname=f"{ticker}/USDT", name=ticker,
                                 timeframe=bt.TimeFrame.Days,
                                 fromdate=fromdate,
                                 todate=todate,
                                 compression=1,
                                 ohlcv_limit=10000,
                                 drop_newest=True)  # , historical=True)
            cerebro.adddata(data)
            data.plotinfo.plot=False

        cerebro.addobserver(bt.observers.Value)
        cerebro.addanalyzer(bt.analyzers.TimeReturn, timeframe=bt.TimeFrame.NoTimeFrame, _name='alltimereturn')
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, riskfreerate=0.0)
        cerebro.addanalyzer(bt.analyzers.Returns)
        cerebro.addanalyzer(bt.analyzers.DrawDown)

        # TODO : Use Default Broker
        cerebro.broker.setcash(10000.0)
        cerebro.broker.addcommissioninfo(BinanceComissionInfo())
        # Add the strategy
        cerebro.addstrategy(TurtleTrendDaily,
                            debug=False,
                            live_feed=True,
                            live_trading=False)
        # Run Here
        res = cerebro.run()

        result = {}
        result["Portfolio"] = sorted(combo)
        result["Return"] = list(res[0].analyzers.alltimereturn.get_analysis().values())[0]
        result["Sharpe"] = res[0].analyzers.sharperatio.get_analysis()['sharperatio']
        result["Annual_Return"] = res[0].analyzers.returns.get_analysis()['rnorm100']
        result["Max_Drawdown"] = res[0].analyzers.drawdown.get_analysis()['max']['drawdown']

        results.append(result)
        print(f"RESULT FOR {list(combo)}:")
        print(f"Return: {result['Return']}")
        print(f"Sharpe: {result['Sharpe']}")
        print(f"Annual_Return: {result['Annual_Return']}")
        print(f"Max_Drawdown: {result['Max_Drawdown']}")


    df = pd.DataFrame(results)
    df.set_index('Portfolio', inplace=True)
    df.to_csv('Random_Experiment.csv')



def run_live_trading():

    # absolute dir the script is in
    script_dir = os.path.dirname(__file__)
    abs_file_path = os.path.join(script_dir, '../config/params-production.json')
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
                         timeframe=bt.TimeFrame.Days,
                         fromdate=fromdate,
                         #todate=todate,
                         compression=1,
                         ohlcv_limit=10000,
                         drop_newest=True)  # , historical=True)
    cerebro.adddata(data)

    for ticker in tickers:
        data = store.getdata(dataname=f"{ticker}/USDT", name=ticker,
                             timeframe=bt.TimeFrame.Days,
                             fromdate=fromdate,
                             #todate=todate,
                             compression=1,
                             ohlcv_limit=10000,
                             drop_newest=True)  # , historical=True)
        cerebro.adddata(data)

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
    cerebro.addstrategy(TurtleTrendDaily,
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

    #cerebro.plot(iplot=False)
    cerebro.plot()



if __name__ == "__main__":

    '''TODO: LIVE TRADING CHECKLIST:
        (1) next: check len(self), so that prenext/nextstart won't error 
        (2) datafeeed: fromdate, enough time for backfill data
        (3) p.live_feed = True p.live_trading = True
        (4) check ticker.csv, which ticker to include
        (5) HERE, data feed should be Day
    '''
    #run_filefeed_backtest()
    #run_livefeed_backtest()
    #run_live_trading()
    run_random_three()
