import backtrader as bt
from tabulate import tabulate
import pandas as pd
from datetime import datetime

class PnLStrategy(bt.Strategy):

    def __init__(self):
        self.trades = []

    def next(self):

        # print position
        for t in self.trades:
            self.print_trade(t)
        self.print_position()

        # buy 2 share at day 10
        if len(self) / 5 == 2:
            self.buy(self.datas[0],size=4)

        # sell 1 at day 15
        if len(self) / 5 == 3:
            self.sell(self.datas[0],size=2)

        # sell 1 at day 20
        if len(self) / 5 == 4:
            self.sell(self.datas[0],size=2)

        # buy 1 at day 20
        if len(self) / 5 == 5:
            self.buy(self.datas[0], size=2)

    def print_position(self):
        pos = self.getposition(self.datas[0])
        if pos.size != 0 :
            print(
                tabulate(
                    [
                        {
                            'DATE': self.datas[0].datetime.date(0),
                            'TICKER': self.datas[0]._name,
                            'AMOUNT': pos.size,
                            'PRICE': pos.price,
                            'CURPRICE': self.datas[0].close[0],
                        }
                    ],
                    headers='keys'
                )
            )

    def print_trade(self, trade):
        print(
            tabulate(
                [
                    {
                        'OPENDATE': trade.open_datetime(),
                        'TICKER': trade.data._name,
                        'AMOUNT': trade.size,
                        'PRICE': trade.price,
                        'VALUE': trade.value,
                        'PNL': trade.pnl,
                    }
                ],
                headers='keys'
            )
        )


    def notify_trade(self, trade):
        self.print_trade(trade)

if __name__ == '__main__':

    cerebro = bt.Cerebro(preload=False, runonce=False)
    cerebro.broker.setcash(10000)
    cerebro.broker.set_coc(True)

    ticker = 'LUNA'
    fromdate = datetime.strptime('2021-01-01', '%Y-%m-%d')
    todate = datetime.strptime('2021-03-01', '%Y-%m-%d')

    data_path = '../../datas/binance-spot-20220330'
    df = pd.read_csv(f"{data_path}/{ticker}.csv",
                     parse_dates=True,
                     index_col=0)

    if len(df) > 20:  # at least 20 day's bar
        cerebro.adddata(bt.feeds.PandasData(dataname=df,
                                            name=ticker,
                                            fromdate=fromdate,
                                            todate=todate,
                                            plot=False))

    cerebro.addstrategy(PnLStrategy)
    cerebro.run()

    cerebro.plot(iplot=False)[0][0]