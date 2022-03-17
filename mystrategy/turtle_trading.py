from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import argparse

import backtrader as bt
import datetime
import time
from backtrader import ResamplerDaily

from backtrader import logger
log = logger.get_logger(__name__)


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


class Turtle(bt.Strategy):
    params = (
        #("short_period", 20),
        # ("long_period",  50 ),
        ("breakout_period", 20),
        ("exit_period", 10),
        ("max_units", 4),  # 1N
        ("risk_per_trade", 0.01),  # risk per trade 1% of the account
        ("portfolio_frac", 0.98),
    )

    def __init__(self):
        self.val_start = self.broker.get_cash()  # keep the starting cash
        self.order = None
        self.N = 0
        self.unit_size = 0
        self.roi = 0

        # define lines
        self.short_period_atr = bt.ind.AverageTrueRange(self.data1, period=self.p.breakout_period)
        # self.long_period_atr = bt.ind.AverageTrueRange(self.data1, period=self.p.long_period)()

        self.breakout_high = bt.ind.Highest(self.data1, period=self.p.breakout_period)
        self.breakout_low = bt.ind.Lowest(self.data1, period=self.p.breakout_period)

        self.exit_high = bt.ind.Highest(self.data1, period=self.p.exit_period)
        self.exit_low = bt.ind.Lowest(self.data1, period=self.p.exit_period)

        # buy signal
        self.buysignal = self.data0.high > self.breakout_high.highest()
        # sell signal
        self.sellsignal = self.data0.low < self.breakout_low.lowest()

        # exit signal
        self.exit_long_signal = self.data0.low < self.exit_low.lowest()
        self.exit_short_signal = self.data0.high > self.exit_high.highest()
        self.current_N_units = 0  # current unit should be smaller than max units permitted
        self.breakout_price = 0  # breakout price
        self.stop_loss = 0  # stop loss price; update during order complete

    def next(self):

        # logging every detailed info
        log.debug(
            (
                f"Data0 {len(self.data0): 07d} ",
                f"{self.data0.datetime.datetime(0):%Y-%m-%d %H:%M:%S} ",
                f"High:{self.data0.high[0]: 4f} ",
                f"Low:{self.data0.low[0]: 4f} ",
                f"Data1 {len(self.data1): 05d} ",
                f"{self.data1.datetime.datetime(0):%Y-%m-%d %H:%M:%S} ",
                f"Signals Buy:{self.buysignal[0]} ",
                f"Sell:{self.sellsignal[0]} ",
                f"Breakout-Highest:{self.breakout_high.highest[0]} ",
                f"Breakout-Lowest: {self.breakout_low.lowest[0]} ",
                f"ATR: {self.short_period_atr.atr[0]}"
            )
        )

        self.N = self.short_period_atr.atr[0]
        self.unit_size = self.val_start * self.p.risk_per_trade / self.N

        # trade here
        if not self.position:  # not in market
            # wait for the first trade
            if self.buysignal[0]:
                # buy
                self.breakout_price = self.breakout_high.highest[0]
                self.buy(size=self.unit_size)
                return

            elif self.sellsignal[0]:
                # sell
                self.breakout_price = self.breakout_low.lowest[0]
                self.sell(size=self.unit_size)
                return

        else:  # in market
            # check stop loss
            if self.position.size < 0 and self.data.high[0] > self.stop_loss:
                # short position && stop loss passed
                self.close()
                return
            if self.position.size > 0 and self.data.low[0] < self.stop_loss:
                # long position && stop loss passed
                self.close()
                return

            # check exit condition
            if self.position.size > 0 and self.exit_long_signal[0]:
                self.close()
                return
            if self.position.size < 0 and self.exit_short_signal[0]:
                self.close()
                return

            # if max units rechead, do nothing
            if self.current_N_units == self.p.max_units:
                # do nothing
                return

            # check need to add position ?
            if self.position.size < 0 and self.data.low[0] < self.breakout_price - 0.5 * self.N:
                # add short position
                self.sell(size=self.unit_size)
            if self.position.size > 0 and self.data.high[0] > self.breakout_price + 0.5 * self.N:
                # add position
                self.buy(size=self.unit_size)

    def notify_order(self, order):
        """Execute when buy or sell is triggered
        Notify if order was accepted or rejected
        """
        if order.alive():
            log.debug(f"Order is alive: {self.datas[0].datetime.datetime(0)}")

            # submitted, accepted, partial, created
            # Returns if the order is in a status in which it can still be executed
            return

        order_side = "Buy" if order.isbuy() else "Sell"
        if order.status == order.Completed:
            log.info(
                (
                    f"{order_side} Order Completed - {self.datas[0].datetime.datetime(0)} "
                    f"Size: {order.executed.size} "
                    f"@Price: {order.executed.price} "
                    f"Value: {order.executed.value:.2f} "
                    f"Comm: {order.executed.comm:.6f} "
                )
            )
            if self.current_N_units == 0:
                self.stop_loss = order.executed.price - self.N * 2 if order.isbuy() else order.executed.price + self.N * 2
            else:
                self.stop_loss = self.stop_loss + self.N * 0.5 if order.isbuy() else self.stop_loss - self.N * 0.5
            self.current_N_units += 1

        elif order.status in {order.Canceled, order.Margin, order.Rejected}:
            log.warn(
                (
                    f"{order_side} Order Canceled/Margin/Rejected"
                    f"Size: {order.created.size} "
                    f"@Price: {order.created.price} "
                    f"Value: {order.created.value:.2f} "
                    f"N Value: {self.N} "
                    f"Remaining Cash: {self.broker.get_cash()}"
                )
            )

        self.order = None  # indicate no order pending

    def notify_trade(self, trade):
        """Execute after each trade
        Calcuate Gross and Net Profit/loss"""

        # trade closed
        if trade.isclosed:
            self.current_N_units = 0  # reset
            self.stop_loss = 0
            self.breakout_price = 0
            log.info(
                f"Operational profit, Gross: {trade.pnl:.2f}, Net: {trade.pnlcomm:.2f}"
            )

    def stop(self):
        """ Calculate the actual returns """
        self.roi = (self.broker.get_value() / self.val_start) - 1.0
        val_end = self.broker.get_value()
        log.info(
            f"PARAMS:{self.p._getkwargs()}, "
            f"ROI: {100.0 * self.roi:.2f}%%, Start cash {self.val_start:.2f}, "
            f"End cash: {val_end:.2f}"
        )

def run_strategy():

    args = parse_args()

    # Get the dates from the args
    fromdate = datetime.datetime.strptime('2017-12-01', '%Y-%m-%d')
    todate = datetime.datetime.strptime('2018-01-31', '%Y-%m-%d')
    dataset_filename = args.dataname

    # Create a cerebro entity
    cerebro = bt.Cerebro(maxcpus=0,
                         runonce=False, # use line coupler, according to documents here can only be false
                         #optdatas=True,
                         #optreturn=True,
                         preload=False)

    # Set our desired cash start
    cerebro.broker.setcash(10000)

    # Create and Add a Data Feed
    data = bt.feeds.GenericCSVData(
        dataname=dataset_filename,
        dtformat="%Y-%m-%dT%H:%M:%S.%f",
        #fromdate=fromdate,
        #todate=todate,
        timeframe=bt.TimeFrame.Minutes,
    )
    cerebro.adddata(data)  # finer data should always be added first

    #####################################
    # Two Ways to Sample and Add Data2:
    # (1) do a clone and add filter, \
    # this still wouldn't work for preload, \
    # need manual turn down, otherwise cause errors
    # this one runs faster buy consumes more memory, results should be the same
    '''
    data2 = bt.DataClone(dataname=data)
    data2.addfilter(ResamplerDaily)
    cerebro.adddata(data2)
    '''

    # (2) use a resampledata func, \
    # this automatically turn down preload, so this always behaves right
    cerebro.resampledata(data, timeframe=bt.TimeFrame.Days)
    ######################################

    # Add a Commission and Support Fractional Size
    cerebro.broker.addcommissioninfo(BinanceComissionInfo())

    cerebro.addstrategy(Turtle)

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

    # Add PyFolio, but this is quite problematic
    # cerebro.addanalyzer(
    #     bt.analyzers.PyFolio, # PyFlio only work with daily data
    #     timeframe=bt.TimeFrame.Minutes,
    #     compression=1440
    # )

    # Add a writer
    # csv_out = 'test-1.csv'
    # cerebro.addwriter(bt.WriterFile, csv=True, out=csv_out)

    # Run Here
    tstart = time.time()  # time.clock()
    results = cerebro.run(runonce=False)
    tend = time.time()  # time.clock()

    st0 = results[0]
    log.info(
        (
            f"Sharp Ratio: {str(st0.analyzers.dailysharp.get_analysis())}, "
            f"All Time Return: {str(st0.analyzers.alltimereturn.get_analysis())}"
        )
    )

    # pyfoliozer = st0.analyzers.getbyname('pyfolio')
    # returns, positions, transactions, gross_lev = pyfoliozer.get_pf_items()
    # returns.to_csv("returns.csv")
    # positions.to_csv("positions.csv")
    # transactions.to_csv("transactions.csv")

    # pyfolio showtime
    # pf.create_full_tear_sheet(
    #     returns,
    #     positions=positions,
    #     transactions=transactions,
    #     #gross_lev=gross_lev,
    #     #live_start_date='2022-02-02',  # This date is sample specific
    #     round_trips=False )

    # cerebro.plot(iplot=False, style="bar")
    # cerebro.plot()

    # print out the result
    log.info(
        f"Total Run Time Used:', {str(tend - tstart)}"
    )

def run_optstrategy():

    args = parse_args()

    # Get the dates from the args
    fromdate = datetime.datetime.strptime('2017-12-01', '%Y-%m-%d')
    todate = datetime.datetime.strptime('2018-01-31', '%Y-%m-%d')
    dataset_filename = args.dataname

    # Create a cerebro entity
    cerebro = bt.Cerebro(maxcpus=0,
                         runonce=False, # use line coupler, according to documents here can only be false
                         optdatas=True,
                         optreturn=True,
                         preload=False)

    # Set our desired cash start
    cerebro.broker.setcash(10000)

    # Create and Add a Data Feed
    data = bt.feeds.GenericCSVData(
        dataname=dataset_filename,
        dtformat="%Y-%m-%dT%H:%M:%S.%f",
        #fromdate=fromdate,
        #todate=todate,
        timeframe=bt.TimeFrame.Minutes,
    )
    cerebro.adddata(data)  # finer data should always be added first
    cerebro.resampledata(data, timeframe=bt.TimeFrame.Days)

    # Add a Commission and Support Fractional Size
    cerebro.broker.addcommissioninfo(BinanceComissionInfo())

    # Add a strategy
    cerebro.optstrategy(
        Turtle,
        breakout_period=range(15, 26),
        exit_period=range(5, 26),
        #max_units=range(2, 6),
        #risk_per_trade=np.arange(0.005, 0.1, 0.005)
    )

    # Add Analyzer
    # Add Daily SharpeRatio
    # According to Ernest Chan's book:
    # if the daily Sharpe ratio multiplied by the square root of the number days (n)\
    # in the backtest is greater than or equal to the critical value 2.326, \
    # then the p-value is smaller than or equal to 0.01.
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


    #######################################################
    # Opt Run Here
    #######################################################
    # clock the start of the process
    tstart = time.time()  # time.clock()

    # Run over everything
    stratruns = cerebro.run(runonce=False, stdstats=False)  # must turn off stdstats for unknown reason

    # clock the end of the process
    tend = time.time()  # time.clock()

    # log Analyzers for all runs
    for stratrun in stratruns:
        for strat in stratrun:
            log.info(
                (
                    f"PARAMS:{str(strat.p._getkwargs())}, "
                    f"Daily Sharp: {str(strat.analyzers.dailysharp.get_analysis())}"
                )
            )

    # print out the result
    log.info(
        f"Total Run Time Used:', {str(tend - tstart)}"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description='Turtle Trading!!!')

    parser.add_argument('--dataname', default='data/BNB.csv', required=False,
                        help='File Data to Load')

    parser.add_argument('--runopt', action='store_true',
                        help='Use next by next instead of runonce')

    parser.add_argument('--plot', required=False, action='store_true',
                        help='Plot the chart')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    if not args.runopt:
        run_strategy()
    else:
        run_optstrategy()