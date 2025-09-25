# examples/quickstart.py
import pandas as pd
import numpy as np
import talib
##import tensorflow as tf
import backtrader as bt
import plotly.express as px

# 1. Pandas + Numpy
data = pd.DataFrame({
    "price": np.linspace(100, 120, 30)
})
print("Sample data:\n", data.head())

# 2. TA-Lib (SMA)
data["sma"] = talib.SMA(data["price"], timeperiod=5)
print("With SMA:\n", data.head(10))

# 3. TensorFlow sanity check
print("TensorFlow version:", tf.__version__)
print("Is GPU available?", tf.config.list_physical_devices("GPU"))

# 4. Backtrader (just instantiation)
cerebro = bt.Cerebro()
print("Backtrader engine ready:", cerebro)

# 5. Plotly visualization
fig = px.line(data, y=["price", "sma"], title="Price vs SMA")
fig.show()
print("Quickstart example completed.") 