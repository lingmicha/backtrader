from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import backtrader as bt
import datetime
import time

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
        ('printout', True)
    )

    def __init__(self):
        self.val_start = self.broker.get_cash()  # keep the starting cash
        self.order = None
        self.N = 0
        self.unit_size = 0
        self.roi = 0

        # define lines
        self.short_period_atr = bt.ind.AverageTrueRange(self.data1, period=self.p.breakout_period)()
        # self.long_period_atr = bt.ind.AverageTrueRange(self.data1, period=self.p.long_period)()

        self.breakout_high = bt.ind.Highest(self.data1, period=self.p.breakout_period)()
        self.breakout_low = bt.ind.Lowest(self.data1, period=self.p.breakout_period)()

        self.exit_high = bt.ind.Highest(self.data1, period=self.p.exit_period)()
        self.exit_low = bt.ind.Lowest(self.data1, period=self.p.exit_period)()

        # buy signal
        self.buysignal = self.data0.high > self.breakout_high.highest
        # sell signal
        self.sellsignal = self.data0.low < self.breakout_low.lowest

        # exit signal
        self.exit_long_signal = self.data0.low < self.exit_low.lowest
        self.exit_short_signal = self.data0.high > self.exit_high.highest
        self.current_N_units = 0  # current unit should be smaller than max units permitted
        self.breakout_price = 0  # breakout price
        self.stop_loss = 0  # stop loss price; update during order complete

    def next(self):

        # print(f"Data0 DateTime: {self.data0.datetime.datetime(0)}",
        #       f"Buy Signal :{self.buysignal[0]}",
        #       f"Current High :{self.data0.high[0]}",
        #       f"Data1 DateTime: {self.data1.datetime.datetime(0)}",
        #       f"Breakout Highest: {self.breakout_high.highest[0]}",
        #       f"ATR Short: {self.short_period_atr.atr[0]}",
        #       f"N : {self.N[0]}"
        #       )

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
            # print(f"Order is alive: {self.datas[0].datetime.datetime(0)}")

            # submitted, accepted, partial, created
            # Returns if the order is in a status in which it can still be executed
            return

        order_side = "Buy" if order.isbuy() else "Sell"
        if order.status == order.Completed:
            # print(
            #     (
            #         f"{order_side} Order Completed - Datetime{self.datas[0].datetime.datetime(0)} "
            #         f"Size: {order.executed.size} "
            #         f"@Price: {order.executed.price} "
            #         f"Value: {order.executed.value:.2f} "
            #         f"Comm: {order.executed.comm:.6f} "
            #     )
            # )
            if self.current_N_units == 0:
                self.stop_loss = order.executed.price - self.N * 2 if order.isbuy() else order.executed.price + self.N * 2
            else:
                self.stop_loss = self.stop_loss + self.N * 0.5 if order.isbuy() else self.stop_loss - self.N * 0.5
            self.current_N_units += 1

        elif order.status in {order.Canceled, order.Margin, order.Rejected}:
            print(f"{order_side} Order Canceled/Margin/Rejected"
                  f"{order.position}")
            # print(
            #     (
            #         f"{order_side} Order Completed - Datetime{self.datas[0].datetime.datetime(0)} "
            #         f"Size: {order.executed.size} "
            #         f"@Price: {order.executed.price} "
            #         f"Value: {order.executed.value:.2f} "
            #         f"Comm: {order.executed.comm:.6f} "
            #     )
            # )
        self.order = None  # indicate no order pending

    def notify_trade(self, trade):
        """Execute after each trade
        Calcuate Gross and Net Profit/loss"""

        # trade closed
        if trade.isclosed:
            self.current_N_units = 0  # reset
            self.stop_loss = 0
            self.breakout_price = 0
            #print(f"Operational profit, Gross: {trade.pnl:.2f}, Net: {trade.pnlcomm:.2f}")

    def stop(self):
        """ Calculate the actual returns """
        self.roi = (self.broker.get_value() / self.val_start) - 1.0
        val_end = self.broker.get_value()
        print(
            f"PARAMS:{self.p._getkwargs()}, "
            f"ROI: {100.0 * self.roi:.2f}%%, Start cash {self.val_start:.2f}, "
            f"End cash: {val_end:.2f}"
        )


def runstrategy():
    # Create a cerebro entity
    cerebro = bt.Cerebro(maxcpus=0,
                         runonce=True,
                         #optdatas=False,
                         optreturn=True,
                         preload=True)

    # Set our desired cash start
    cerebro.broker.setcash(10000)

    # Get the dates from the args
    fromdate = datetime.datetime.strptime('2017-12-01', '%Y-%m-%d')
    todate = datetime.datetime.strptime('2018-01-31', '%Y-%m-%d')
    dataset_filename = 'data/BNB.csv'

    # Create a Data Feed
    data = bt.feeds.GenericCSVData(
        dataname=dataset_filename,
        dtformat="%Y-%m-%dT%H:%M:%S.%f",
        # fromdate=fromdate,
        # todate=todate,
        timeframe=bt.TimeFrame.Ticks
    )

    # Add the Data Feed to Cerebro
    # cerebro.adddata(data)
    cerebro.resampledata(data, timeframe=bt.TimeFrame.Minutes, compression=1)
    cerebro.resampledata(data, timeframe=bt.TimeFrame.Days)

    # Add a Commission and Support Fractional Size
    cerebro.broker.addcommissioninfo(BinanceComissionInfo())

    # Add a strategy
    cerebro.optstrategy(
        Turtle,
        #short_period=range(5, 30),
        breakout_period=range(5, 30),
        exit_period=range(5, 25),
        #max_units=range(2, 6),
        #risk_per_trade=np.arange(0.005, 0.1, 0.005)
    )

    # Add Oberserver
    # cerebro.addobserver(bt.observers.DrawDown)
    # cerebro.addobserver(bt.observers.DrawDown_Old)

    # Analyzer
    cerebro.addanalyzer(
        bt.analyzers.TimeReturn,
        _name="alltime_roi",
        timeframe=bt.TimeFrame.NoTimeFrame
    )

    # cerebro.addanalyzer(
    #     bt.analyzers.TradeAnalyzer,
    #     _name="trade_analysis",
    # )

    # cerebro.addanalyzer(
    #     bt.analyzers.TimeReturn,
    #     data=data,
    #     _name="benchmark",
    #     timeframe=bt.TimeFrame.NoTimeFrame,
    # )

    # cerebro.addanalyzer(
    #     bt.analyzers.PyFolio, # PyFlio only work with daily data
    #     timeframe=bt.TimeFrame.Minutes,
    #     compression=1440
    # )

    # Print out the starting conditions
    # print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

    # Add a writer
    # csv_out = 'test-1.csv'
    # cerebro.addwriter(bt.WriterFile, csv=True, out=csv_out)

    # Run over everything
    # cerebro.run(runonce=False)
    # cerebro.run()
    # results = cerebro.run()
    # st0 = results[0]

    # pyfoliozer = st0.analyzers.getbyname('pyfolio')
    # returns, positions, transactions, gross_lev = pyfoliozer.get_pf_items()
    # returns.to_csv("returns.csv")
    # positions.to_csv("positions.csv")
    # transactions.to_csv("transactions.csv")

    # for alyzer in st0.analyzers:
    #     alyzer.print()

    # Print out the final result
    # print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

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

    # clock the start of the process
    tstart = time.time()  # time.clock()

    # Run over everything
    stratruns = cerebro.run()

    # clock the end of the process
    tend = time.time()  # time.clock()

    print('==================================================')
    for stratrun in stratruns:
        print('**************************************************')
        for strat in stratrun:
            print('--------------------------------------------------')
            print(strat.p._getkwargs())
            for alyzer in strat.analyzers:
                alyzer.print()
    print('==================================================')

    # print out the result
    print('Time used:', str(tend - tstart))


if __name__ == '__main__':
    runstrategy()
