#Monte Carlo Simulation and Option Pricing
Overview
This project leverages Monte Carlo simulations to evaluate portfolio performance and employs the Black-Scholes model for option pricing using real market data. Developed in Python, it integrates financial data retrieval, statistical analysis, and visualization to assess investment strategies and market risk.

Credits
Some code concepts were adapted from QuantPy tutorials.

Features
Portfolio Simulation: Generates simulated portfolio values based on historical stock data.
Option Pricing: Calculates option prices using both Monte Carlo simulation and the Black-Scholes model.
Risk Assessment: Computes Value at Risk (VaR) and Conditional Value at Risk (CVaR) to quantify portfolio risk.
Visualization: Provides graphical representation of portfolio simulations and option pricing comparisons.
Dependencies
Python 3.x
Required Libraries: numpy, pandas, matplotlib, scipy, yfinance, yahoo_fin, pandas_datareader
Installation
Clone the repository: git clone <repository_url>
Install dependencies: pip install -r requirements.txt
Usage
Modify stockList and other parameters in the script as needed.
Run python monte_carlo_option_pricing.py to execute the program.
View simulation results and option pricing comparisons in the generated plots.
