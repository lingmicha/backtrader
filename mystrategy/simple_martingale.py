from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import backtrader as bt
import datetime
import argparse

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


class DynamicHighest(bt.Indicator):
    lines = ('dyn_highest',)
    #plotlines = dict( dyn_highest = dict(_skipnan=False))

    def __init__(self):
        self._tradeopen = False

    def tradeopen(self, yesno):
        self._tradeopen = yesno

    def next(self):
        if self._tradeopen:
            self.lines.dyn_highest[0] = max(self.data[0], self.dyn_highest[-1])

class DynamicLowest(bt.Indicator):
    lines = ('dyn_lowest',)
    #plotlines = dict( dyn_lowest = dict(_skipnan=False))

    def __init__(self):
        self._tradeopen = False
        self.counter = 0

    def tradeopen(self, yesno):
        self._tradeopen = yesno

    def next(self):
        if self._tradeopen:
            self.lines.dyn_lowest[0] = min( self.data[0], self.dyn_lowest[-1])

class MartingaleSize():

    def __init__(self):
        self.double_time = 1
        self.init_cap = 20

    def double(self):
        self.double_time *= 2

    def reset(self):
        self.double_time = 1

    def getsize(self, cash):
        try_size = self.init_cap * self.double_time
        self.double()
        if cash < try_size:
            return cash
        return try_size

def pct_chg(p1,p2):
    return (p1-p2)/p2

class Martingale(bt.Strategy):
    # sell mark:    price higher 1.3% than average holding cost
    # sell trigger: price retract from high 0.3%
    # buy mark:     price lower than 4% of average holding cost
    # buy trigger:  price rebounce 0.5% after buy mark

    params = (
        ("sell_mark", 0.013),
        ("sell_signal", -0.003),
        ("buy_mark", -0.04),
        ("buy_signal", 0.005),
        # Percentage of portfolio for a trade. Something is left for the fees
        # otherwise orders would be rejected
        ("portfolio_frac", 0.98),
        ("debug", False),

    )

    def __init__(self):
        self.val_start = self.broker.getcash()  # keep the starting cash
        self.size = None
        self.order = None

        self.dyn_highest = DynamicHighest( self.data.high, subplot=False )
        self.dyn_lowest = DynamicLowest( self.data.low, subplot=False )

        self.reset_mark()

        self.size_calc = MartingaleSize()

    def log(self, txt, dt=None):
        if self.p.debug:
            dt = dt or self.data.datetime[0]
            dt = bt.num2date(dt)
            print('%s, %s' % (dt.isoformat(), txt))


    def reset_mark(self):
        self.sell_mark = False
        self.buy_mark = False

    def next(self):

        available_cash = self.broker.getcash() * self.p.portfolio_frac

        if not self.position:  # not in the market
            # buy in now
            self.size_calc.reset() # reset to 1
            self.size =  self.size_calc.getsize( available_cash ) / self.data.lines.close[0]  # strating from 10u
            self.order = self.buy(size=self.size)

            # reset all mark
            self.reset_mark()

            print(
                "Enter Market:"
                f"DateTime {self.datas[0].datetime.datetime(0)}, "
                f"Price {self.data[0]:.2f}, "
                f"Amount {self.size}"
            )

        else:  # in the market

            if self.p.debug:
                print(
                    (
                        f"TradeOpen: {self.dyn_highest._tradeopen},"
                        f"Dyn Highest:{self.dyn_highest[0]},"
                        f"Dyn Lowest:{self.dyn_lowest[0]},"
                    )
                )

            # process buy/sell first
            if self.sell_mark and (
                    pct_chg(self.data.lines.low[0], self.dyn_highest[0]) < self.p.sell_signal):
                # sell all
                self.order = self.close()
                self.reset_mark()

                print(
                    (
                        f"Leave Market:"
                        f"DateTime {self.datas[0].datetime.datetime(0)}, "
                        f"Market Price {self.data[0]:.2f}, "
                        f"Market Low {self.data.lines.low[0]:.2f},"
                        f"Position Cost {self.position.price}， "
                        f"Position Size {self.position.size}， "
                        f"Recent High {self.dyn_highest[0]} "
                    )
                )

                return

            if self.buy_mark and (
                    pct_chg(self.data.lines.high[0], self.dyn_lowest[0]) > self.p.buy_signal):
                # buy
                self.size = self.size_calc.getsize( available_cash ) / self.data.lines.close[0]  # strating from 10u
                self.order = self.buy(size=self.size)
                self.reset_mark()

                print(
                    (
                        f"Double Bet:"
                        f"DateTime {self.datas[0].datetime.datetime(0)}, "
                        f"Market Price {self.data[0]:.2f},"
                        f"Market High  {self.data.lines.high[0]:.2f},"
                        f"Position Cost {self.position.price}， "
                        f"Position Size {self.position.size}， "
                        f"Recent Low {self.dyn_lowest[0]} "
                    )
                )
                return

            if (pct_chg(self.data.lines.high, self.position.price) > self.p.sell_mark):
                self.sell_mark = True

            if (pct_chg(self.data.lines.low, self.position.price) < self.p.buy_mark):
                self.buy_mark = True

        #if self.order:
        #    return  # pending order execution. Waiting in orderbook

        #print(
        #        f"DateTime {self.datas[0].datetime.datetime(0)}, "
        #        f"Price {self.data[0]:.2f}, "
        #        f"Position {self.position.upopened}"
        #    )

    def notify_order(self, order):
        """Execute when buy or sell is triggered
        Notify if order was accepted or rejected
        """
        if order.alive():
            print(f"Order is alive: {self.datas[0].datetime.datetime(0)}")

            # submitted, accepted, partial, created
            # Returns if the order is in a status in which it can still be executed
            return

        order_side = "Buy" if order.isbuy() else "Sell"
        if order.status == order.Completed:
            print(
                (
                    f"{order_side} Order Completed - Datetime{self.datas[0].datetime.datetime(0)} "
                    f"Size: {order.executed.size} "
                    f"@Price: {order.executed.price} "
                    f"Value: {order.executed.value:.2f} "
                    f"Comm: {order.executed.comm:.6f} "
                )
            )
        elif order.status in {order.Canceled, order.Margin, order.Rejected}:
            print(f"{order_side} Order Canceled/Margin/Rejected")
        self.order = None  # indicate no order pending

    def notify_trade(self, trade):
        """Execute after each trade
        Calcuate Gross and Net Profit/loss"""

        self.dyn_highest.tradeopen(trade.isopen)
        self.dyn_lowest.tradeopen(trade.isopen)

        # trade closed
        if trade.isclosed:
            print(f"Operational profit, Gross: {trade.pnl:.2f}, Net: {trade.pnlcomm:.2f}")

    def stop(self):
        """ Calculate the actual returns """
        self.roi = (self.broker.getvalue() / self.val_start) - 1.0
        val_end = self.broker.getvalue()
        print(
            f"ROI: {100.0 * self.roi:.2f}%%, Start cash {self.val_start:.2f}, "
            f"End cash: {val_end:.2f}"
        )

def runstrategy():

    args = parse_args()

    # Create a cerebro entity
    cerebro = bt.Cerebro()

    # Get the dates from the args
    fromdate = datetime.datetime.strptime(args.fromdate, '%Y-%m-%d')
    todate = datetime.datetime.strptime(args.todate, '%Y-%m-%d')
    dataset_filename = args.data

    # Datas are in a subfolder of the samples. Need to find where the script is
    # because it could have been called from anywhere
    # modpath = os.path.dirname(os.path.abspath(sys.argv[0]))
    # datapath = os.path.join(modpath, 'data/bitmex/XBT_USD_2020-10-01_2020-10-31.csv')
    # dataset_filename = 'data/bitmex/XBT_USD_2020-10-01_2020-10-31.csv'
    # dataset_filename = 'data/bitmex/XBT_USD_2022-02-01_2022-02-28.csv'
    # dataset_filename = '2022.2-PEOPLEUSDT.csv'
    # dataset_filename = 'PEOPLEUSDT.csv'


    # Create a Data Feed
    data = bt.feeds.GenericCSVData(
        dataname=dataset_filename,
        dtformat="%Y-%m-%dT%H:%M:%S",
        timeframe=bt.TimeFrame.Ticks
    )

    # Add the Data Feed to Cerebro
    # cerebro.adddata(data)
    cerebro.resampledata(data, timeframe=bt.TimeFrame.Minutes, compression=1)

    # Add a Commission and Support Fractional Size
    cerebro.broker.addcommissioninfo(BinanceComissionInfo())

    # Set our desired cash start
    cerebro.broker.setcash( args.cash )

    # Add a strategy
    cerebro.addstrategy(Martingale)

    # Add Oberserver
    cerebro.addobserver(bt.observers.DrawDown)
    #cerebro.addobserver(bt.observers.DrawDown_Old)

    # Analyzer
    cerebro.addanalyzer(
        bt.analyzers.SharpeRatio,
        _name='mysharpe',
        timeframe= bt.TimeFrame.Days,
        compression=1440,
        annualize=True)

    cerebro.addanalyzer(
        bt.analyzers.TimeReturn,
        _name="alltime_roi",
        timeframe=bt.TimeFrame.NoTimeFrame
    )

    cerebro.addanalyzer(
        bt.analyzers.TradeAnalyzer,
        _name="trade_analysis",
    )

    cerebro.addanalyzer(
        bt.analyzers.TimeReturn,
        data=data,
        _name="benchmark",
        timeframe=bt.TimeFrame.NoTimeFrame,
    )

    # cerebro.addanalyzer(
    #     bt.analyzers.PyFolio, # PyFlio only work with daily data
    #     timeframe=bt.TimeFrame.Days,
    #     compression=1440
    # )

    # Print out the starting conditions
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

    # Add a writer
    #csv_out = 'test.csv'
    #cerebro.addwriter(bt.WriterFile, csv=args.writercsv, out=csv_out)

    # Run over everything
    #cerebro.run(runonce=False)
    #cerebro.run()
    results = cerebro.run()
    st0 = results[0]

    # pyfoliozer = st0.analyzers.getbyname('pyfolio')
    # returns, positions, transactions, gross_lev = pyfoliozer.get_pf_items()

    for alyzer in st0.analyzers:
        alyzer.print()

    # Print out the final result
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

    # pyfolio showtime
    # pf.create_full_tear_sheet(
    #     returns,
    #     positions=positions,
    #     transactions=transactions,
    #     #gross_lev=gross_lev,
    #     #live_start_date='2022-02-02',  # This date is sample specific
    #     round_trips=False )

    #cerebro.plot(iplot=False, style="bar")
    #cerebro.plot()



def parse_args():

    parser = argparse.ArgumentParser(description='Simple Martingale Strategy')

    parser.add_argument('--data', '-d',
                        default='PEOPLEUSDT.csv',
                        help='data to add to the system')

    parser.add_argument('--fromdate', '-f',
                        default='2021-12-01',
                        help='Starting date in YYYY-MM-DD format')

    parser.add_argument('--todate', '-t',
                        default='2022-03-09',
                        help='Ending date in YYYY-MM-DD format')

    parser.add_argument('--writercsv', '-wcsv', action='store_true',
                        help='Tell the writer to produce a csv stream')

    parser.add_argument('--cash', default=10000, type=int,
                        help='Starting Cash')

    return parser.parse_args()


if __name__ == '__main__':

    runstrategy()

