import vectorbt as vbt

cols = ['Open', 'High', 'Low', 'Close', 'Volume']
# ohlcv = vbt.YFData.download('BTC-USD').get(cols)
# ohlcv = vbt.YFData.download('EURUSD=X').get(cols)
ohlcv = vbt.YFData.download('TSLA').get(cols)
# print(ohlcv)

closing_prices = ohlcv.get('Close')
highs = ohlcv.get('High')
lows = ohlcv.get('Low')

# Bollinger Bands Example
# bb = vbt.BBANDS.run(closing_prices)
# entries = bb.
# exits =

# RSI Example
rsi = vbt.RSI.run(closing_prices)
exits = rsi.rsi_crossed_below(30)
entries = rsi.rsi_crossed_above(70)

# MA Cross Example
# fast_ma = vbt.MA.run(closing_prices, 13) # Regular: 13, Golden Cross: 50
# slow_ma = vbt.MA.run(closing_prices, 51) # Regular: 51, Golden Cross: 200
# entries = fast_ma.ma_crossed_above(slow_ma)
# exits = fast_ma.ma_crossed_below(slow_ma)



portfolio = vbt.Portfolio.from_signals(closing_prices, entries, exits, freq="H", init_cash=10000)

portfolio.plot().show()