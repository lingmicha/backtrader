import pickle
import signal
from time import sleep

import backtrader as bt
import pandas as pd
import os.path

'''
This is a demo program about how to 
Resume Trading from a proper state
'''


class ResumeTrading(bt.Strategy):

    STATE_FILE = 'state.out'

    def __init__(self):

        signal.signal(signal.SIGINT, self.sigstop)

        self.rerun = False
        self.crypto = []

        if os.path.exists(ResumeTrading.STATE_FILE):
            # Getting back the objects:
            with open(ResumeTrading.STATE_FILE, 'rb') as f:  # Python 3: open(..., 'rb')
                size, price = pickle.load(f)

            # Setting Positions
            data_pos = self.getposition()
            data_pos.set(size, price)

            os.rename(ResumeTrading.STATE_FILE, ResumeTrading.STATE_FILE+'.bak')
            self.rerun = True


    def nextstart(self):
        if self.rerun:
            print("RESUME PROPERLY......")
        else:
            order1 = self.buy(size=10)
            order2 = self.buy(size=5)

        print('========IN NEXTSTART========')
        size = self.getposition().size
        price = self.getposition().price

        print(f"POSITIONS FOR {self.data._name}: "
              f"SIZE {size} "
              f"PRICE {price} "
             )
        print('========END NEXTSTART========')

    def next(self):
        print("SLEEP 5s")
        sleep(5)

    def notify_order(self, order):
        if order.status == order.Completed:
            print(f"BUY {order.p.data._name} "
                  f"SIZE {order.executed.size} "
                    f"PRICE {order.executed.price} "
                  )

            self.crypto.append(order)

    def sigstop(self, a, b):
        print('STOPPING BACKTRADER......')

        # close all position
        size = self.getposition().size
        price = self.getposition().price

        print(f"POSITIONS FOR {self.data._name}: "
              f"SIZE {size} "
              f"PRICE {price} "
              )

        with open(ResumeTrading.STATE_FILE, 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump([size, price], f)

        self.env.runstop()

if __name__ == '__main__':

    cerebro = bt.Cerebro(stdstats=False)

    df = pd.read_csv('2017-2022-BNB.csv', parse_dates=True, index_col=0)
    data = bt.feeds.PandasData(dataname=df, name='BNB', timeframe=bt.TimeFrame.Days)

    cerebro.adddata(data)
    cerebro.addstrategy(ResumeTrading)

    cerebro.run()

    # END