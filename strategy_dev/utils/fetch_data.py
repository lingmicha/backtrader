import os
import json
import time
import ccxt
import sys



def get_config(exchange_id='binance', market='spot', environment='sandbox'):
    script_dir = os.path.dirname(__file__)
    script_name = 'params-' + environment + '-' + market + '.json'
    abs_file_path = os.path.join(script_dir, '../config/' ,  script_name)
    with open(abs_file_path, 'r') as f:
        params = json.load(f)

    config = {'exchange_id': exchange_id,
              'apiKey': params[exchange_id]["apikey"],
              'secret': params[exchange_id]["secret"],
              'enableRateLimit': True,
              'nonce': lambda: str(int(time.time() * 1000)),
              'options': {
                  'defaultType': market,
              },
              }

    return config


# Get CCXT Exchange Object
def get_exchange(config=None, check=False):

    if not hasattr(get_exchange, "exchange"):
        import ccxt

        if config is None:
            raise Exception("First time call get_exchange must provide a valid config")

        exchange_class = getattr(ccxt, config['exchange_id'])
        get_exchange.exchange = exchange_class(config)

        if check:
            print(get_exchange.exchange.requiredCredentials)  # prints required credentials
            get_exchange.exchange.checkRequiredCredentials()  # raises AuthenticationError

    return get_exchange.exchange


# Get USDT quote Coins in a List
def get_listings(exchange, quote='USDT'):
    import re

    # Load all markets, which is trading pairs for SPOT markets
    pairs = exchange.load_markets()
    print(f"Total Number of Trading Pairs:{len(pairs)}")

    # Use Base to determine unique number of coins trading
    coins = set(map(lambda x: x["base"], pairs.values()))
    print(f"Total Number of Trading Coins:{len(coins)}")

    # Use Base and quote to determine unique USDT quoted coins
    quoted_coins = set(map(lambda y: y["base"], filter(lambda x: x["quote"] == quote, pairs.values())))
    print(f"Total Number of {quote} Quoted Coins:{len(quoted_coins)}")

    # apply override here:
    stable_coins = ['BUSD', 'UST', 'DAI', 'USDC', 'TUSD', "TUSD", "USDP",'USDT']
    quoted_coins = [x for x in quoted_coins if x not in stable_coins]
    print(f"Total Number of {quote} Quoted Coins After Removing Stable Coins:{len(quoted_coins)}")

    fiats = ["AUD", "BIDR", "BRL", "EUR", "GBP", "RUB", "TRY", "IDRT", "UAH", "NGN", "VAI"]
    quoted_coins = [x for x in quoted_coins if x not in fiats]
    print(f"Total Number of {quote} Quoted Coins After Removing Fiats:{len(quoted_coins)}")

    # 1 - remove all *UP, *DOWN coins
    if exchange.exchange_id == 'binance':
        quoted_coins = list(filter(lambda x: re.match("\w+UP$", x) is None, quoted_coins))
        print(f"Total Number of {quote} Quoted Coins After Removing ^UP coins:{len(quoted_coins)}")

        quoted_coins = list(filter(lambda x: re.match("\w+DOWN$", x) is None, quoted_coins))
        print(f"Total Number of {quote} Quoted Coins After Removing ^DOWN coins:{len(quoted_coins)}")

        quoted_coins = list(filter(lambda x: re.match("\w+BULL$", x) is None, quoted_coins))
        print(f"Total Number of {quote} Quoted Coins After Removing ^BULL coins:{len(quoted_coins)}")

        quoted_coins = list(filter(lambda x: re.match("\w+BEAR$", x) is None, quoted_coins))
        print(f"Total Number of {quote} Quoted Coins After Removing ^BEAR coins:{len(quoted_coins)}")

        binance_remove_list = ["NPXS", "BCC", "BEAR", "BKRW", "BSV", "BTT", "BULL", \
                       "BZRX", "ERD", "HC", "KEEP", "LEND", \
                       "MCO", "NANO", "NU", "PAX", "RGT", "STORM", "STRAT", "USDS", "USDSB", "VEN", \
                       "XZC"]
        quoted_coins = [x for x in quoted_coins if x not in binance_remove_list]
        print(f"Total Number of {quote} Quoted Coins After Removing Binance Overrides:{len(quoted_coins)}")

    elif exchange.exchange_id == 'huobi':
        huobi_remove_list = ['BCHA','BOR','BOT','CELR','HBC','MINA','OKB','QNT','ROSE','SLP','YAMV2','LEND']
        quoted_coins = [x for x in quoted_coins if x not in huobi_remove_list]
        print(f"Total Number of {quote} Quoted Coins After Removing Huobi Overrides:{len(quoted_coins)}")

        # New Tickers might not have data during the period:
        # ['AMP', 'BRWL', 'CAKE', 'CEEK', 'DAO', 'EGS', 'FUSE', 'GMT', 'GQ', 'ONIT', 'RACA', 'SPRT', 'VISION', 'XCN',
        #  'XCUR', 'XDEFI']

    return list(quoted_coins)


# Get OHLCV data from Binance for a USDT coin

def get_ohlcv(exchange, symbol, from_date, to_date, timeframe='1d', rlimit=1):
    from datetime import datetime
    import pytz
    import time

    start = datetime.strptime(from_date, "%Y-%m-%d").replace(tzinfo=pytz.UTC)
    start_timestamp = int(start.timestamp() * 1000)

    end = datetime.strptime(to_date, "%Y-%m-%d").replace(tzinfo=pytz.UTC)
    end_timestamp = int(end.timestamp() * 1000)

    # set timeframe in msecs
    if timeframe == '1d':
        tf_multi = 24 * 60 * 60 * 1000
    elif timeframe == '1m':
        tf_multi = 60 * 1000
    else:
        raise Exception('timeframe not supported')

    hold = 30
    data = []

    # -----------------------------------------------------------------------------
    # ADDED:
    if exchange.has['fetchOHLCV'] == 'emulated':
        print(exchange.id, " cannot fetch old historical OHLCVs, because it has['fetchOHLCV'] =",
              exchange.has['fetchOHLCV'])
        sys.exit()
    # -----------------------------------------------------------------------------

    candle_no = (end_timestamp - start_timestamp) / tf_multi + 1

    while start_timestamp < end_timestamp:
        try:
            ohlcvs = exchange.fetch_ohlcv(symbol, timeframe, since=start_timestamp)

            if rlimit is not None:
                time.sleep(rlimit)
            # --------------------------------------------------------------------
            # ADDED:
            # check if returned ohlcvs are actually
            # within the from_timestamp > ohlcvs > end range
            if len(ohlcvs) == 0 :
                # Huobi may return empty ohlcvs if timestamp early than earliest.
                if len(data) == 0:
                    start_timestamp += tf_multi * 10 # increase timestamp and retry
                    continue
                else:
                    break
            if (ohlcvs[0][0] > end_timestamp):
                print(exchange.id, "got a candle out of range! has['fetchOHLCV'] =", exchange.has['fetchOHLCV'])
                break
            if (ohlcvs[-1][0] > end_timestamp):
                # filter timestamp within range
                ohlcvs = list(filter(lambda x: x[0] <= end_timestamp, ohlcvs))
                data += ohlcvs
                break
            # ---------------------------------------------------------------------

            start_timestamp = ohlcvs[-1][0] + tf_multi
            data += ohlcvs
            print(str(len(data)) + ' of ' + str(int(candle_no)) + ' candles loaded...')
        except (ccxt.ExchangeError, ccxt.AuthenticationError, ccxt.ExchangeNotAvailable, ccxt.RequestTimeout) as error:
            print('Got an error', type(error).__name__, error.args, ', retrying in', hold, 'seconds...')
            time.sleep(hold)
            break

    # use a set to remove duplicates
    data = list(map(lambda x: (datetime.fromtimestamp(x[0] / 1000)
                               .astimezone(pytz.utc),
                               x[1], x[2], x[3], x[4], x[5]), data))

    return data


"""
    Datasource: Coingecko
    Map Binance Symbol to Coingecko id, and return a dict
    NOTE: THE DATA QUALITY IS NOT GOOD!
"""
def fetch_mapping(exchange_id):
    # exchange_id ={ 'binance', 'huobi' , etc}
    from pycoingecko import CoinGeckoAPI
    import time

    cg = CoinGeckoAPI()

    all_tickers = []
    page = 1
    next_page = True

    while next_page:

        print(f"fetch page{page}...")
        tickers = cg.get_exchanges_tickers_by_id(id=exchange_id, order='volume_desc', page=page)
        tickers = tickers["tickers"]

        all_tickers = [*all_tickers, *tickers]
        page += 1
        time.sleep(1)

        if len(tickers) == 0:
            next_page = False

    mapping = set(map(lambda x: (x['base'], x['coin_id']), all_tickers))

    return dict(mapping)

'''
    Datasource: Coingecko
    NOTE: THE DATA QUALITY IS NOT GOOD!
'''
def get_usd_close_price(coingecko_id, from_date, end_date):
    from datetime import datetime, timedelta
    import pytz
    import time

    time.sleep(1)  # avoid overhitting cg service

    start = datetime.strptime(from_date, "%Y-%m-%d").replace(tzinfo=pytz.UTC)
    end = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=pytz.UTC)

    # get_coin_market_chart_range_by_id would auto-adjust to daily data when interval>90day
    # so need to check interval
    if (end - start) <= timedelta(days=90):
        raise Exception("end_date should be 90 days after from_date")

    start_timestamp = int(datetime.timestamp(start))
    end_timestamp = int(datetime.timestamp(end))

    if not hasattr(get_marketcap, "cg"):
        from pycoingecko import CoinGeckoAPI
        get_marketcap.cg = CoinGeckoAPI()

    market_chart = get_marketcap.cg.get_coin_market_chart_range_by_id(id=coingecko_id, vs_currency='usd',
                                                                      from_timestamp=start_timestamp,
                                                                      to_timestamp=end_timestamp)
    close_prices = market_chart['prices']
    #market_caps = market_chart['market_caps']

    close_prices = list(map(lambda x: (datetime.fromtimestamp(x[0] / 1000)
                                     .astimezone(pytz.utc),
                                     x[1]),
                          close_prices))

    return close_prices

'''
    Datasource: Coingecko
    NOTE: THE DATA QUALITY IS NOT GOOD!
'''
def get_usd_market_chart(coingecko_id, from_date, end_date):
    from datetime import datetime, timedelta
    import pytz
    import time

    time.sleep(1)  # avoid overhitting cg service

    start = datetime.strptime(from_date, "%Y-%m-%d").replace(tzinfo=pytz.UTC)
    end = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=pytz.UTC)

    # get_coin_market_chart_range_by_id would auto-adjust to daily data when interval>90day
    # so need to check interval
    if (end - start) <= timedelta(days=90):
        raise Exception("end_date should be 90 days after from_date")

    start_timestamp = int(datetime.timestamp(start))
    end_timestamp = int(datetime.timestamp(end))

    if not hasattr(get_marketcap, "cg"):
        from pycoingecko import CoinGeckoAPI
        get_marketcap.cg = CoinGeckoAPI()

    market_chart = get_marketcap.cg.get_coin_market_chart_range_by_id(id=coingecko_id, vs_currency='usd',
                                                                      from_timestamp=start_timestamp,
                                                                      to_timestamp=end_timestamp)
    close_prices = market_chart['prices']
    market_caps = market_chart['market_caps']
    market_chart = []

    if len(close_prices) != len(market_caps):
        raise Exception("market cap length not equal to close price")

    for i, close in enumerate(close_prices):
        bar = (datetime.fromtimestamp(close[0] / 1000)
               .astimezone(pytz.utc),
               close[1],
               market_caps[i][1])
        market_chart.append(bar)

    return market_chart

'''
    Datasource: Coingecko
    NOTE: THE DATA QUALITY IS NOT GOOD!
'''
def get_marketcap(coingecko_id, from_date, end_date):
    from datetime import datetime, timedelta
    import pytz
    import time

    time.sleep(1)  # avoid overhitting cg service

    start = datetime.strptime(from_date, "%Y-%m-%d").replace(tzinfo=pytz.UTC)
    end = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=pytz.UTC)

    # get_coin_market_chart_range_by_id would auto-adjust to daily data when interval>90day
    # so need to check interval
    if (end - start) <= timedelta(days=90):
        raise Exception("end_date should be 90 days after from_date")

    start_timestamp = int(datetime.timestamp(start))
    end_timestamp = int(datetime.timestamp(end))

    if not hasattr(get_marketcap, "cg"):
        from pycoingecko import CoinGeckoAPI
        get_marketcap.cg = CoinGeckoAPI()

    market_chart = get_marketcap.cg.get_coin_market_chart_range_by_id(id=coingecko_id, vs_currency='usd',
                                                                      from_timestamp=start_timestamp,
                                                                      to_timestamp=end_timestamp)
    market_cap = market_chart["market_caps"]
    market_cap = list(map(lambda x: (datetime.fromtimestamp(x[0] / 1000)
                                     .astimezone(pytz.utc),
                                     x[1]),
                          market_cap))

    return market_cap

'''
    Datasource: Coingecko
    NOTE: THE DATA QUALITY IS NOT GOOD!
'''
def fetch_binance_ticker_historical_usd(start,end):
    # For each USDT COIN, get ohlcv, get marketcap
    # THIS FUNC CAN ONLY RETRIEVE DAY BAR
    import time
    import pandas as pd

    mapping = fetch_mapping('binance')

    for i, coin in enumerate( mapping.keys() ):
        print(f"Start Processing {i}th coin: {coin}...")

        cg_id = mapping[coin]  # would raise key error

        market_charts = get_usd_market_chart(cg_id, start, end)

        market_charts_df = pd.DataFrame(market_charts,
                                       columns=["date", "close",'market_cap'],
                                       )
        market_charts_df.set_index("date", inplace=True)
        market_charts_df.sort_index(inplace=True)

        market_charts_df.to_csv("data/" + coin + ".csv")

    # Write out usdt_coins list
    mapping_list = [(k, v) for k, v in mapping.items()]
    coins_df = pd.DataFrame(mapping_list)
    coins_df.columns = ["ticker",'cg_id']
    coins_df.to_csv("data/tickers.csv", header=None)

    return

'''
    Datasource: Coingecko
    NOTE: THE DATA QUALITY IS NOT GOOD!
'''
def fetch_binance_ticker_with_marketcap(start, end,):
    # For each USDT COIN, get ohlcv, get marketcap
    # THIS FUNC CAN ONLY RETRIEVE DAY BAR
    import time
    import pandas as pd

    config = get_config(exchange_id='binance', market='spot', environment='production')
    exchange = get_exchange(config)
    usdt_coins = get_usdt_coins(exchange)
    mapping = fetch_mapping('binance')

    for i, coin in enumerate((sorted(usdt_coins))):
        print(f"Start Processing {i}th coin: {coin}...")

        if coin not in mapping:
            print(f"Skip {coin}, not found in coingecko mapping...")
            continue

        cg_id = mapping[coin]  # would raise key error

        marketcap = get_marketcap(cg_id, start, end)
        ohlcv = get_ohlcv(exchange, coin + "/USDT", start, end, '1d') # here can only day bar

        marketcap_df = pd.DataFrame(marketcap, columns=["date", "market_cap"])
        marketcap_df.sort_index(inplace=True)

        ohlcv_df = pd.DataFrame(ohlcv, columns=["date", "open", "high", "low", "close", "volume"])
        ohlcv_df.sort_index(inplace=True)

        marketcap_df.set_index("date", inplace=True)
        ohlcv_df.set_index("date", inplace=True)

        # merge defaults to inner join
        df = pd.merge(ohlcv_df, marketcap_df, left_index=True, right_index=True)

        df.to_csv("data/" + coin + ".csv")

    # Write out usdt_coins list
    coins_df = pd.DataFrame(usdt_coins)
    coins_df.columns = ["symbol"]
    coins_df.to_csv("data/tickers.csv", header=None)

    return


def fetch_exchange_bar(exchange_id, market, environment, start, end, quote, timeframe ):
    # timeframe : '1d' or '1m'
    # For each USDT COIN, get ohlcv, get marketcap
    import time
    import pandas as pd
    import ccxt

    config = get_config(exchange_id, market=market, environment=environment)
    exchange = get_exchange(config)

    coins = get_listings(exchange, quote=quote)
    sorted_coins = sorted(coins)

    for i, coin in enumerate(sorted_coins):
        print(f"Start Processing {i}th coin: {coin}...")
        ohlcv = get_ohlcv(exchange, f"{coin}/{quote}", start, end, timeframe=timeframe)
        ohlcv_df = pd.DataFrame(ohlcv, columns=["date", "open", "high", "low", "close", "volume"])
        ohlcv_df.sort_index(inplace=True)
        ohlcv_df.set_index("date", inplace=True)
        ohlcv_df.to_csv("data/" + coin + ".csv")

    # Write out usdt_coins list
    coins_df = pd.DataFrame(sorted_coins)
    coins_df.columns = ["symbol"]
    coins_df.to_csv("data/tickers.csv", header=None)

def fetch_exchange_bar_via_btc(exchange_id, market, environment, start, end, quote, timeframe ):
    # timeframe : '1d' or '1m'
    # For each USDT COIN, get ohlcv, get marketcap
    import time
    import pandas as pd
    import ccxt

    config = get_config(exchange_id, market=market, environment=environment)
    exchange = get_exchange(config)

    coins = get_listings(exchange, quote='BTC')
    sorted_coins = sorted(coins)

    # First Fetch BTC Bars
    ohlcv_btc = get_ohlcv(exchange, f"BTC/{quote}", start, end, timeframe=timeframe)
    ohlcv_btc_df = pd.DataFrame(ohlcv_btc, columns=["date", "open", "high", "low", "close", "volume"])
    ohlcv_btc_df.set_index("date", inplace=True)
    ohlcv_btc_df.sort_index(inplace=True)
    ohlcv_btc_df.drop(columns=["open", "high", "low", "volume"], inplace=True)
    ohlcv_btc_df.rename(columns={"close" : "close_btc"}, inplace=True)
    print(f"The Earliest Date for BTC is {ohlcv_btc[0][0]}")

    for i, coin in enumerate(sorted_coins):
        print(f"Start Processing {i}th coin: {coin}...")
        ohlcv = get_ohlcv(exchange, f"{coin}/BTC", start, end, timeframe=timeframe)
        ohlcv_df = pd.DataFrame(ohlcv, columns=["date", "open", "high", "low", "close", "volume"])
        ohlcv_df.set_index("date", inplace=True)
        ohlcv_df.sort_index(inplace=True)
        ohlcv_df.drop(columns=["open", "high", "low", "volume"], inplace=True)
        ohlcv_df.rename(columns={"close": "close_base"}, inplace=True)

        # Convert quote price
        df = pd.merge( ohlcv_df, ohlcv_btc_df, left_index=True, right_index=True )
        df['close'] = df['close_base'] * df['close_btc']
        #df.drop(columns=["close_base", "close_btc"])

        df.to_csv("data/" + coin + ".csv")

    # Write out usdt_coins list
    coins_df = pd.DataFrame(sorted_coins)
    coins_df.columns = ["symbol"]
    coins_df.to_csv("data/tickers.csv", header=None)



if __name__ == "__main__":

    # All Configures Here
    exchange_id = 'binance'
    market = 'future'
    environment = 'production'

    start = '2020-01-01'
    end = '2022-03-30'
    quote = 'USDT'
    timeframe = '1m'

    fold_path = '../data'

    fetch_exchange_bar(exchange_id, market, environment, start, end, quote, timeframe)
    #fetch_exchange_bar_via_btc(exchange_id, market, environment, start, end, quote, timeframe)