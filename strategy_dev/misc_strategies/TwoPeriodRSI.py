from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import backtrader as bt
import datetime
import argparse
from backtrader_plotting import Bokeh
from backtrader_plotting.schemes import Tradimo

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

class TwoPeriodRSI(bt.Strategy):

    ''' Larry Connor’s 2-period RSI
        Refrence: https://www.elearnmarkets.com/blog/how-to-trade-larry-connors-2-period-rsi/
        Buy Condition:
        (1) 2 Period RSI close below 10 for two period
        (2) price above long EMA ( commit to long trend )
        (3) price below short EMA ( short retract )

        Stop Loss:
        (1) N = ATR

        Exit:
        (1) 1 Risk-Reward Ratio ( +N )

        Tested Theory:
        Buy Condition:
            long EMA=60, short EMA = 3 doesn't have many trades, since EMA=60 it not long enough
            long EMA=200, short EMA = 5
        Exit:
            2 Period RSI > 90  ==>Not Working, not responsive enough
            + N is more likely to yield winning trades
            5-EMA also yield winning trades
    '''

    params = (
        ("long_ema_period", 200),
        ("short_ema_period", 5),
        ("rsi_period", 2),
        ("atr_period", 60),
        ("risk_per_trade", 0.01),  # risk per trade 1% of the account
        ("rsi_overbought", 90),
        ("rsi_oversold", 10),
        ("debug", False),

    )

    def __init__(self):

        self.Long_EMA = bt.ind.EMA(self.data, period=self.p.long_ema_period)
        self.Short_EMA = bt.ind.EMA(self.data, period=self.p.short_ema_period)
        self.ATR = bt.ind.ATR(self.data, period=self.p.atr_period)
        self.RSI = bt.ind.RSI(self.data, period=self.p.rsi_period, safediv=True)

        self.val_start = self.broker.getcash()

    def next(self):

        # if in market check exit/stop loss
        if self.position:
            if self.data.low[0] < self.stop_loss:
                self.close()
                return

            if self.data.high[0] >= self.Short_EMA.ema[0]:
                self.close()
                return

        else: # not in market, check enter conditon
            if self.data.close[0] > self.Long_EMA.ema[0] \
                and self.data.close[0] < self.Short_EMA.ema[0] \
                and self.RSI.rsi[0] <= self.p.rsi_oversold \
                and self.RSI.rsi[-1] <= self.p.rsi_oversold:

                N = self.ATR.atr[0]
                size = self.p.risk_per_trade * self.broker.getcash()/ N
                self.buy(size=size)

                return

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

            if order.isbuy():
                N = self.ATR.atr[0]
                self.stop_loss = order.executed.price - N
                self.exit = order.executed.price + N

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

    def notify_trade(self, trade):
        """Execute after each trade
        Calcuate Gross and Net Profit/loss"""

        # trade closed
        if trade.isclosed:
            self.stop_loss = 0
            print(
                f"Operational profit, Gross: {trade.pnl:.2f}, Net: {trade.pnlcomm:.2f}"
            )

    def stop(self):
        """ Calculate the actual returns """
        self.roi = (self.broker.getvalue() / self.val_start) - 1.0
        val_end = self.broker.get_value()
        print(
            f"PARAMS:{self.p._getkwargs()}, "
            f"ROI: {100.0 * self.roi:.2f}%%, Start cash {self.val_start:.2f}, "
            f"End cash: {val_end:.2f}"
        )

def runstrategy():

    args = parse_args()

    # Create a cerebro entity
    cerebro = bt.Cerebro( runonce=False, preload=False )

    # Get the dates from the args
    fromdate = datetime.datetime.strptime(args.fromdate, '%Y-%m-%d')
    todate = datetime.datetime.strptime(args.todate, '%Y-%m-%d')
    dataset_filename = args.data

    # Create a Data Feed
    data = bt.feeds.GenericCSVData(
        #fromdate=fromdate,
        #todate=todate,
        dataname=dataset_filename,
        dtformat="%Y-%m-%dT%H:%M:%S.%f",
        timeframe=bt.TimeFrame.Ticks
    )

    # Add the Data Feed to Cerebro
    cerebro.resampledata(data, timeframe=bt.TimeFrame.Days, compression=1)

    # Add a Commission and Support Fractional Size
    cerebro.broker.addcommissioninfo(BinanceComissionInfo())

    # Set our desired cash start
    cerebro.broker.setcash( args.cash )

    # Add a strategy
    cerebro.addstrategy(TwoPeriodRSI)

    # Add Oberserver
    cerebro.addobserver(bt.observers.DrawDown)

    # Analyzer
    cerebro.addanalyzer(
        bt.analyzers.SharpeRatio,
        _name='mysharpe',
        timeframe= bt.TimeFrame.Days,
        annualize=True)

    cerebro.addanalyzer(
        bt.analyzers.TimeReturn,
        timeframe=bt.TimeFrame.NoTimeFrame,
        _name='alltimereturn')

    # Add PyFolio, but this is quite problematic
    cerebro.addanalyzer(
        bt.analyzers.PyFolio,  # PyFlio only work with daily data
        timeframe=bt.TimeFrame.Days,
    )

    # Print out the starting conditions
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

    # Run over everything
    #cerebro.run(runonce=False)
    #cerebro.run()
    results = cerebro.run()
    st0 = results[0]
    #
    pyfoliozer = st0.analyzers.getbyname('pyfolio')
    returns, positions, transactions, gross_lev = pyfoliozer.get_pf_items()

    returns.to_csv('returns.csv')
    positions.to_csv('positions.csv')
    transactions.to_csv('transactions.csv')

    # for alyzer in st0.analyzers:
    #     alyzer.print()

    # Print out the final result
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

    # backtrader plot
    b = Bokeh(style='bar', plot_mode='single', scheme=Tradimo())
    cerebro.plot(b)
    #cerebro.plot()


def parse_args():

    parser = argparse.ArgumentParser(description='Simple Martingale Strategy')

    parser.add_argument('--data', '-d',
                        default='/Users/michael/Projects/backtrader/mystrategy/data/BNB.csv',
                        #default='/Users/michael/Projects/algo-trading/My_Stuff/data/LUNAUSDT-2020-08-21-2022-03-21.csv',
                        help='data to add to the system')

    parser.add_argument('--fromdate', '-f',
                        default='2022-02-01',
                        help='Starting date in YYYY-MM-DD format')

    parser.add_argument('--todate', '-t',
                        default='2022-03-09',
                        help='Ending date in YYYY-MM-DD format')

    parser.add_argument('--cash', default=10000, type=int,
                        help='Starting Cash')

    return parser.parse_args()


if __name__ == '__main__':

    runstrategy()

