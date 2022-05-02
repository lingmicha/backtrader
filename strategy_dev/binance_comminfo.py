import backtrader as bt

class CryptoCommInfo(bt.CommInfoBase):
    params = (
        ("commission", 0.00018),
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


class BinancePerpetualFutureCommInfo(CryptoCommInfo):
    params = (
        ("commission", 0.00036),
    )


class BinanceSpotCommInfo(CryptoCommInfo):
    params = (
        ("commission", 0.000750),
    )
