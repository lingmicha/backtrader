
import backtrader as bt
from datetime import datetime
import pandas as pd

'''
    MixStrategy uses two timeframe:
    minute vs day bars
    it uses a timer to do positions rebalance,
    uses a next to check stop loss
'''

class MixStrategy(bt.Strategy):

    def __init__(self):
        self.add_timer(
            when=bt.timer.SESSION_START,
            #offset=self.p.offset,
            #repeat=self.p.repeat,
            #weekdays=self.p.weekdays,
        )

    def next(self):
        print(f'strategy next with datetime:{self.datas[0].datetime.datetime(0)}')

    def notify_timer(self, timer, when, *args, **kwargs):
        print('strategy notify_timer with tid {}, when {}'.
              format(timer.p.tid, when,))


if __name__ == "__main__":

    cerebro = bt.Cerebro(preload=False, runonce=False)
    cerebro.broker.setcash(10000)
    cerebro.broker.set_coc(True)

    ticker = 'PEOPLE'

    fromdate = datetime.strptime('2021-12-24', '%Y-%m-%d')
    todate = datetime.strptime('2022-03-29', '%Y-%m-%d')

    data_path = '.'
    data_file = '2021-2022-PEOPLE.csv'
    df = pd.read_csv(f"{data_path}/{data_file}",
                     parse_dates=True,
                     index_col=0)

    data = bt.feeds.PandasData(dataname=df,
                                        name=ticker,
                                        fromdate=fromdate,
                                        todate=todate,
                                        plot=False)
    cerebro.adddata(data)
    cerebro.resampledata(data,
                         name=f"{ticker}_daybar",
                         timeframe=bt.TimeFrame.Days,
                         )

    cerebro.addstrategy(MixStrategy)
    cerebro.run()

    cerebro.plot(iplot=False)[0][0]



