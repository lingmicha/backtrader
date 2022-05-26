from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import datetime
from backtrader.feed import DataBase
from backtrader import date2num
from sqlalchemy import create_engine, text, engine
from urllib.parse import quote


class MySQLData(DataBase):
    params = (
        ('dbHost', None),
        ('dbUser', None),
        ('dbPWD', None),
        ('dbName', None),
        ('dbPort',3306),
        ('symbol', 'BTC/USDT'),
        ('fromdate', datetime.datetime.min),
        ('todate', datetime.datetime.max),
        ('name', ''),
        )

    def __init__(self):
        driver = "mysql"
        url = engine.URL(driver, self.p.dbUser, self.p.dbPWD, self.p.dbHost, self.p.dbPort, self.p.dbName )
        self.engine = create_engine( url )
        self.result = None

    def start(self):
        with self.engine.connect() as conn:
            self.result = conn.execute(
                text("SELECT datetime, open, high, low, close, volume FROM day_ohlcv WHERE symbol = :symbol AND datetime between :start AND :end ORDER BY datetime ASC "),
                {"symbol": self.p.symbol, "start": self.p.fromdate.strftime("%Y-%m-%d"), "end": self.p.todate.strftime("%Y-%m-%d")},
            )

    def stop(self):
        self.engine.dispose()

    def _load(self):
        one_row = self.result.fetchone()
        if one_row is None:
            return False
        self.lines.datetime[0] = date2num(one_row[0])
        self.lines.open[0] = float(one_row[1])
        self.lines.high[0] = float(one_row[2])
        self.lines.low[0] = float(one_row[3])
        self.lines.close[0] = float(one_row[4])
        self.lines.volume[0] = int(one_row[5])
        self.lines.openinterest[0] = -1
        return True
