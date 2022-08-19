import vectorbt as vbt

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pytz
from dateutil.parser import parse
import ipywidgets as widgets
from copy import deepcopy
from tqdm import tqdm
import imageio
from IPython import display
import plotly.graph_objects as go
import itertools
import dateparser
import gc

# Enter your parameters here
seed = 42
symbol = 'BTC-USD'
metric = 'total_return'

start_date = datetime(2018, 1, 1, tzinfo=pytz.utc)  # time period for analysis, must be timezone-aware
end_date = datetime(2020, 1, 1, tzinfo=pytz.utc)
time_buffer = timedelta(days=100)  # buffer before to pre-calculate SMA/EMA, best to set to max window
freq = '1D'

vbt.settings.portfolio['init_cash'] = 100.  # 100$
vbt.settings.portfolio['fees'] = 0.0025  # 0.25%
vbt.settings.portfolio['slippage'] = 0.0025  # 0.25%

# Download data with time buffer
cols = ['Open', 'High', 'Low', 'Close', 'Volume']
ohlcv_wbuf = vbt.YFData.download(symbol, start=start_date-time_buffer, end=end_date).get(cols)

ohlcv_wbuf = ohlcv_wbuf.astype(np.float64)
    
# print(ohlcv_wbuf.shape)
# print(ohlcv_wbuf.columns)

# Create a copy of data without time buffer
wobuf_mask = (ohlcv_wbuf.index >= start_date) & (ohlcv_wbuf.index <= end_date) # mask without buffer

ohlcv = ohlcv_wbuf.loc[wobuf_mask, :]

# print(ohlcv.shape)

# Plot the OHLC data
# ohlcv_wbuf.vbt.ohlcv.plot().show_svg() 
# remove show_svg() to display interactive chart!
ohlcv_wbuf.vbt.ohlcv.plot().show()

## SINGLE WINDOW COMBINATION

fast_window = 30
slow_window = 80

# Pre-calculate running windows on data with time buffer
fast_ma = vbt.MA.run(ohlcv_wbuf['Open'], fast_window)
slow_ma = vbt.MA.run(ohlcv_wbuf['Open'], slow_window)

print(fast_ma.ma.shape)
print(slow_ma.ma.shape)

# Remove time buffer
fast_ma = fast_ma[wobuf_mask]
slow_ma = slow_ma[wobuf_mask]

# there should be no nans after removing time buffer
assert(~fast_ma.ma.isnull().any()) 
assert(~slow_ma.ma.isnull().any())

print(fast_ma.ma.shape)
print(slow_ma.ma.shape)



# Generate crossover signals
dmac_entries = fast_ma.ma_crossed_above(slow_ma)
dmac_exits = fast_ma.ma_crossed_below(slow_ma)

fig = ohlcv['Open'].vbt.plot(trace_kwargs=dict(name='Price'))
fig = fast_ma.ma.vbt.plot(trace_kwargs=dict(name='Fast MA'), fig=fig)
fig = slow_ma.ma.vbt.plot(trace_kwargs=dict(name='Slow MA'), fig=fig)
fig = dmac_entries.vbt.signals.plot_as_entry_markers(ohlcv['Open'], fig=fig)
fig = dmac_exits.vbt.signals.plot_as_exit_markers(ohlcv['Open'], fig=fig)

fig.show_svg()

# Signal stats
print(dmac_entries.vbt.signals.stats(settings=dict(other=dmac_exits)))

# Plot signals
fig = dmac_entries.vbt.signals.plot(trace_kwargs=dict(name='Entries'))
dmac_exits.vbt.signals.plot(trace_kwargs=dict(name='Exits'), fig=fig).show_svg()

# Build partfolio, which internally calculates the equity curve

# Volume is set to np.inf by default to buy/sell everything
# You don't have to pass freq here because our data is already perfectly time-indexed
dmac_pf = vbt.Portfolio.from_signals(ohlcv['Close'], dmac_entries, dmac_exits)

# Print stats
print(dmac_pf.stats())

# Plot trades
print(dmac_pf.trades.records)
dmac_pf.trades.plot().show_svg()

# Now build portfolio for a "Hold" strategy
# Here we buy once at the beginning and sell at the end
hold_entries = pd.Series.vbt.signals.empty_like(dmac_entries)
hold_entries.iloc[0] = True
hold_exits = pd.Series.vbt.signals.empty_like(hold_entries)
hold_exits.iloc[-1] = True
hold_pf = vbt.Portfolio.from_signals(ohlcv['Close'], hold_entries, hold_exits)

# Equity
fig = dmac_pf.value().vbt.plot(trace_kwargs=dict(name='Value (DMAC)'))
hold_pf.value().vbt.plot(trace_kwargs=dict(name='Value (Hold)'), fig=fig).show_svg()

# Now we will implement an interactive window slider to easily compare windows by their performance.

min_window = 2
max_window = 100

perf_metrics = ['total_return', 'positions.win_rate', 'positions.expectancy', 'max_drawdown']
perf_metric_names = ['Total return', 'Win rate', 'Expectancy', 'Max drawdown']

windows_slider = widgets.IntRangeSlider(
    value=[fast_window, slow_window],
    min=min_window,
    max=max_window,
    step=1,
    layout=dict(width='500px'),
    continuous_update=True
)
dmac_fig = None
dmac_img = widgets.Image(
    format='png',
    width=vbt.settings['plotting']['layout']['width'],
    height=vbt.settings['plotting']['layout']['height']
)
metrics_html = widgets.HTML()

def on_value_change(value):
    global dmac_fig
    
    # Calculate portfolio
    fast_window, slow_window = value['new']
    fast_ma = vbt.MA.run(ohlcv_wbuf['Open'], fast_window)
    slow_ma = vbt.MA.run(ohlcv_wbuf['Open'], slow_window)
    fast_ma = fast_ma[wobuf_mask]
    slow_ma = slow_ma[wobuf_mask]
    dmac_entries = fast_ma.ma_crossed_above(slow_ma)
    dmac_exits = fast_ma.ma_crossed_below(slow_ma)
    dmac_pf = vbt.Portfolio.from_signals(ohlcv['Close'], dmac_entries, dmac_exits)

    # Update figure
    if dmac_fig is None:
        dmac_fig = ohlcv['Open'].vbt.plot(trace_kwargs=dict(name='Price'))
        fast_ma.ma.vbt.plot(trace_kwargs=dict(name='Fast MA'), fig=dmac_fig)
        slow_ma.ma.vbt.plot(trace_kwargs=dict(name='Slow MA'), fig=dmac_fig)
        dmac_entries.vbt.signals.plot_as_entry_markers(ohlcv['Open'], fig=dmac_fig)
        dmac_exits.vbt.signals.plot_as_exit_markers(ohlcv['Open'], fig=dmac_fig)
    else:
        with dmac_fig.batch_update():
            dmac_fig.data[1].y = fast_ma.ma
            dmac_fig.data[2].y = slow_ma.ma
            dmac_fig.data[3].x = ohlcv['Open'].index[dmac_entries]
            dmac_fig.data[3].y = ohlcv['Open'][dmac_entries]
            dmac_fig.data[4].x = ohlcv['Open'].index[dmac_exits]
            dmac_fig.data[4].y = ohlcv['Open'][dmac_exits]
    dmac_img.value = dmac_fig.to_image(format="png")
    
    # Update metrics table
    sr = pd.Series([dmac_pf.deep_getattr(m) for m in perf_metrics], 
                   index=perf_metric_names, name='Performance')
    metrics_html.value = sr.to_frame().style.set_properties(**{'text-align': 'right'}).render()
    
windows_slider.observe(on_value_change, names='value')
on_value_change({'new': windows_slider.value})

dashboard = widgets.VBox([
    widgets.HBox([widgets.Label('Fast and slow window:'), windows_slider]),
    dmac_img,
    metrics_html
])
dashboard

dashboard.close() # after using, release memory and notebook metadata

gc.collect()