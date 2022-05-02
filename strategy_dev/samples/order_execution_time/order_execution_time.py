import backtrader as bt
import pandas as pd
from datetime import datetime
from tabulate import tabulate

class OrderExecTiming(bt.Strategy):
    '''
        Use a 4hr bar to generate a daily bar
        Buy on a daily bar.
        See the difference between cheat-on-close(coc) price and non-coc price

        4hr bar arrives at:
            00:00:00, 04:00:00, 08:00:00, 12:00:00, 16:00:00, 20:00:00
        the daily bar arrives at:
            23:59:59 with values from
            00:00:00, 04:00:00, 08:00:00, 12:00:00, 16:00:00, 20:00:00
            and generates a timestamp at 23:59:59
        IMPORTANT: LIVE FEED WOULD HAVE DELAY of 20:00:00 bar so it won't be included

        notify_timer price:
            case[1]: trading on data0(4hr):
                timer entry: 2021-12-22 00：00：00
                trading price: 2021-12-21 04：00：00 open price,
                                since data0 would be updated when executing order
                | DATETIME            | TICKER   | STATUS    |   AMOUNT |   PRICE |   COMM |
                |---------------------|----------|-----------|----------|---------|--------|
                | 2021-12-21 04:00:00 | BNB.csv  | COMPLETED |        2 |  527.68 |      0 |

            case[2]: trading on data1(resampled day):
                timer entry: 2021-12-22 00：00：00
                trading price: 2021-12-21 open price, since data1 holds only 12-21 daily data.
                | DATETIME                   | TICKER   | STATUS    |   AMOUNT |   PRICE |   COMM |
                |----------------------------|----------|-----------|----------|---------|--------|
                | 2021-12-21 23:59:59.999989 | BNB.csv  | COMPLETED |        2 |  523.96 |      0 |

            case[3]: trading on data1, with cheat=True, and broke cheat-on-open
                same as case[2], order at openning price of previous day's day bar
                // broker cheat-on-close has no effect

            case[4]: trading on data0, with cheat=True, and broker cheat-on-open
                order executed at price of data0 00:00:00

            OBSERVATION: notify_timer would come before strategy rollover, i.e.:
                CASE: trading on data0:
                notify_timer：2021-12-21 00:00:00
                    - strategy: 2021-12-20 20:00:00
                    - data0:    2021-12-21 00:00:00
                    - data1:    2021-12-20 23:59:59.999989
                then strategy next： 2021-12-21 00:00:00
                    - strategy: 2021-12-21 00:00:00
                    - data0:    2021-12-21 00:00:00
                    - data1:    2021-12-20 23:59:59.999989
                then order execution after data0 update but before strategy rollover,ie,
                    - data0:    2021-12-21 04:00:00

                CASE: trading on data1:
                notify_timer：2021-12-22 00:00:00
                    - strategy: 2021-12-21 20:00:00
                    - data0:    2021-12-22 00:00:00
                    - data1:    2021-12-21 23:59:59.999989
                then order execution with:
                    - data1:    2021-12-21 23:59:59.999989 open price => WRONG!!!!
                then strategy next： 2021-12-22 00:00:00
                    - strategy: 2021-12-22 00:00:00
                    - data0:    2021-12-22 00:00:00
                    - data1:    2021-12-21 23:59:59.999989

            notify_timer + cheat-on-close price:
                case1: trading on data0: no effect, which is right, data0 is openning of the day
                case2: trading on data1: would decrease the day by one. STILL WRONG PRICE !!!

            CONCLUSION: !!!IMPORTANT!!!
                - with notify_timer, always trading on the smaller timeframe data
                - with next, always trading on the smaller timeframe data,
                  since the larger data is always delayed, hence give out-dated price.
                -

    '''
    def __init__(self):
        self.add_timer(
            when=bt.timer.SESSION_START,
            cheat=True,
        )
        self.counter = 0

    def notify_timer(self, timer, when, *args, **kwargs):
        if len(self) == 0:
            return

        print(f"strategy notify_timer when {when} ",
              f"strategy when {self.datetime.datetime(0)} "
              f"data0: {self.datas[0].datetime.datetime(0)}, {self.datas[0].open[0]}/{self.datas[0].close[0]} ",
              f"data1: {self.datas[1].datetime.datetime(0)}, {self.datas[1].open[0]}/{self.datas[1].close[0]}"
              )

        self.counter += 1

        if self.counter % 20 == 0 :
            self.buy(self.datas[0], size=2)

    def next(self):
        print(f"strategy next when {self.datetime.datetime(0)} "
              f"data0: {self.datas[0].datetime.datetime(0)}, {self.datas[0].open[0]}/{self.datas[0].close[0]} "
              f"data1: {self.datas[1].datetime.datetime(0)}, {self.datas[1].open[0]}/{self.datas[1].close[0]}"
              )

        # self.counter += 1
        #
        # if self.counter % 20 == 0 :
        #     self.buy(self.datas[1], size=2)


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
            o = {
                'DATETIME': order.p.data.datetime.datetime(0),
                'TICKER': order.p.data._name,
                'STATUS': 'COMPLETED',
                'AMOUNT': order.executed.size,
                'PRICE': order.executed.price,
                'COMM': order.executed.comm,
            }
            print(tabulate([o], headers='keys', tablefmt='github'))


        elif order.status in {order.Canceled, order.Margin, order.Rejected}:
            o = {
                'DATETIME': order.p.data.datetime.datetime(0),
                'TICKER': order.p.data._name,
                'STATUS': 'Margin' if order.status == 7 else 'Rejected',
                'AMOUNT': order.created.size,
                'PRICE': order.created.price,
                'COMM': order.created.comm,
            }
            print(tabulate([o], headers='keys', tablefmt='github'))



if __name__ == "__main__":

    ticker = 'BNB.csv'
    fromdate = datetime.strptime('2021-12-01', '%Y-%m-%d')
    todate = datetime.strptime('2022-03-30', '%Y-%m-%d')

    cerebro = bt.Cerebro(stdstats=False,
                         runonce=False,
                         preload=False,
                         )


    df = pd.read_csv(f"{ticker}",
                parse_dates=True,
                index_col=0
                )
    data = bt.feeds.PandasData(dataname=df,
                         name=ticker,
                         timeframe=bt.TimeFrame.Minutes,
                         compression=240,
                         plot=False,
                         fromdate=fromdate,
                         todate=todate
                         )

    cerebro.adddata(data)
    cerebro.resampledata(data,
                         name=ticker,
                         timeframe=bt.TimeFrame.Days
                         )
    cerebro.addstrategy(OrderExecTiming)
    #cerebro.broker.set_coc(True)
    cerebro.broker.set_coo(True)
    cerebro.run()
