"""
Author: Idan Malka - 17/06/2023
Apache License Version 2.0 - http://www.apache.org/licenses/

Backtesting Platform:
Graphical User Interface CustomTkinter to model and predict btc price using several predictors
Develop set toolkit to generate strategies with risk management (EMA, TPO, VWAP, IO, Divergences, Value areas, nPOC,... )
Backtest strategies on Crypto currencies databases (APIs: Polygon, Bybit, yfinance)
"""

from GUI import Main

#Starts all the threads and pages.
if __name__ == "__main__":
    global gui
    gui = Main()
    gui.mainloop()
