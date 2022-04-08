import backtrader as bt
import pandas as pd

class MyStrategy(bt.Strategy):

    def __init__(self):
        return

    def next(self):
        print(f"NEXT-DATA0: {self.data.datetime.datetime(0):%Y-%m-%d %H:%M:%S},  "
              f"Ticks: {self.data[0]} "
              f"STRATEGY-LEN: {len(self)}")

def run_backtest():
    data_path = '../../algo-trading/My_Stuff/notebook/Turtle_Trend/data'
    tickers = pd.read_csv(data_path + '/tickers.csv', header=None)[1].tolist()

    cerebro = bt.Cerebro(stdstats=False, preload=False, runonce=False)

    bitcoin = 'AVAX'
    df = pd.read_csv(f"{data_path}/{bitcoin}.csv",
                     parse_dates=True,
                     index_col=0)
    cerebro.resampledata(bt.feeds.PandasData(dataname=df, name=bitcoin, plot=False, timeframe=bt.TimeFrame.Minutes),
                         timeframe=bt.TimeFrame.Days)
    cerebro.addstrategy(MyStrategy)
    cerebro.run()

if __name__ == '__main__':
    run_backtest()