!pip install yahoo_fin
!pip install --upgrade pandas_datareader

# Import dependencies
import math
import numpy as np
import pandas as pd
import datetime
import scipy.stats as stats
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
import yfinance as yf
from yahoo_fin import options
from collections import defaultdict

# import data
def get_data(stocks, start, end):
    stockData = yf.download(stocks, start, end)
    stockData = stockData['Close']
    returns = stockData.pct_change()
    meanReturns = returns.mean()
    covMatrix = returns.cov()
    return meanReturns, covMatrix

stockList = ['MSFT', 'TSLA', 'NVDA', 'AAPL']
stocks = [stock for stock in stockList]
endDate = datetime.datetime.now()
startDate = endDate - datetime.timedelta(days=300)

meanReturns, covMatrix = get_data(stocks, startDate, endDate)

weights = np.random.random(len(meanReturns))
weights /= np.sum(weights)

# Monte Carlo Method
mc_sims = 1000 # number of simulations
T = 100 #timeframe in days

meanM = np.full(shape=(T, len(weights)), fill_value=meanReturns)
meanM = meanM.T

portfolio_sims = np.full(shape=(T, mc_sims), fill_value=0.0)

initialPortfolio = 10000

for m in range(mc_sims):
    Z = np.random.normal(size=(T, len(weights)))  # uncorrelated RVs
    L = np.linalg.cholesky(covMatrix)  # Cholesky decomposition to Lower Triangular Matrix
    dailyReturns = meanM + np.inner(L, Z)  # Correlated daily returns for individual stocks
    portfolio_sims[:, m] = np.cumprod(np.inner(weights, dailyReturns.T) + 1) * initialPortfolio

plt.plot(portfolio_sims)
plt.ylabel('Portfolio Value ($)')
plt.xlabel('Days')
plt.title('MC simulation of a stock portfolio')
plt.show()

def mcVaR(returns, alpha=5):
    """ Input: pandas series of returns
        Output: percentile on return distribution to a given confidence level alpha
    """
    if isinstance(returns, pd.Series):
        return np.percentile(returns, alpha)
    else:
        raise TypeError("Expected a pandas data series.")

def mcCVaR(returns, alpha=5):
    """ Input: pandas series of returns
        Output: CVaR or Expected Shortfall to a given confidence level alpha
    """
    if isinstance(returns, pd.Series):
        belowVaR = returns <= mcVaR(returns, alpha=alpha)
        return returns[belowVaR].mean()
    else:
        raise TypeError("Expected a pandas data series.")

portResults = pd.Series(portfolio_sims[-1,:])

VaR = initialPortfolio - mcVaR(portResults, alpha=5)
CVaR = initialPortfolio - mcCVaR(portResults, alpha=5)

print('VaR_5 ${}'.format(round(VaR, 2)))
print('CVaR_5 ${}'.format(round(CVaR, 2)))

def black_scholes(S, K, T, r, vol):
    # Calculate d1 and d2
    d1 = (np.log(S / K) + (r + 0.5 * vol ** 2) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)

    # Calculate the call price
    call_price = S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)

    return call_price

# initial derivative parameters
OStock = yf.download(stocks[1]) # The stock where we're investigating the option value for
S = OStock['Close'].iloc[-1]  # stock price

r = 0.0549  # risk-free rate (%)
N = 10  # number of time steps
M = 1000  # number of simulations

expir_dates = yf.Ticker(stocks[1]).options
first_expiry = datetime.datetime.strptime(expir_dates[0], "%Y-%m-%d")

T = ((first_expiry - datetime.datetime.today()).days + 1) / 365  # time in years
print(T)

# Get the option chain for the first expiration date
option_chain = yf.Ticker(stocks[1]).option_chain(expir_dates[0])

# Find the option with the highest volume
all_options = pd.concat([option_chain.calls])
highest_volume_option = all_options.loc[all_options['volume'].idxmax()]

# Extract strike price and last price (market value)
K = highest_volume_option['strike']  # strike price
market_value = highest_volume_option['lastPrice']  # market price of option
vol = highest_volume_option['impliedVolatility']  # volatility (%)
print("expiration date: ", expir_dates[0])
print("Strike Price: ", K)
print("Market val: ", market_value)

# precompute constants
dt = T / N
nudt = (r - 0.5 * vol ** 2) * dt
volsdt = vol * np.sqrt(dt)
lnS = np.log(S)

# Monte Carlo Method
Z = np.random.normal(size=(N, M))
delta_lnSt = nudt + volsdt * Z
lnSt = lnS + np.cumsum(delta_lnSt, axis=0)
lnSt = np.concatenate((np.full(shape=(1, M), fill_value=lnS), lnSt))

# Compute Expectation and SE
ST = np.exp(lnSt)
CT = np.maximum(0, ST - K)
C0 = np.exp(-r * T) * np.sum(CT[-1]) / M

sigma = np.sqrt(np.sum((CT[-1] - C0) ** 2) / (M - 1))
SE = sigma / np.sqrt(M)

print("Call value is ${0} with SE +/- {1}".format(np.round(C0, 2), np.round(SE, 2)))

x1 = np.linspace(C0 - 3 * SE, C0 - 1 * SE, 100)
x2 = np.linspace(C0 - 1 * SE, C0 + 1 * SE, 100)
x3 = np.linspace(C0 + 1 * SE, C0 + 3 * SE, 100)

s1 = stats.norm.pdf(x1, C0, SE)
s2 = stats.norm.pdf(x2, C0, SE)
s3 = stats.norm.pdf(x3, C0, SE)

plt.fill_between(x1, s1, color='tab:blue', label='> StDev')
plt.fill_between(x2, s2, color='cornflowerblue', label='1 StDev')
plt.fill_between(x3, s3, color='tab:blue')

plt.plot([C0, C0], [0, max(s2) * 1.1], 'k', label='Monte Carlo Value')
plt.plot([market_value, market_value], [0, max(s2) * 1.1], 'r', label='Market Value')

# Compute the Black-Scholes price
bs_price = black_scholes(S, K, T, r, vol)
print("Black-Scholes Call Price: ${:.2f}".format(bs_price))
plt.plot([bs_price, bs_price], [0, max(s2) * 1.1], 'g', label='BS Value')

plt.ylabel("Option Price")
plt.xlabel("Probability")
plt.legend()
plt.show()
