'''Few lines of code just to extract implied volatility curves from data'''

import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
from scipy.stats import norm

def BSM(S, K, r, sgmBSM, tau):
    d1 = (np.log(S/K) + (r + sgmBSM**2*0.5)*tau) / (sgmBSM*np.sqrt(tau))
    d2 = d1 - sgmBSM*np.sqrt(tau)
    phi_one = norm.cdf(d1) 
    phi_two = norm.cdf(d2)
    
    Call = S*phi_one - K*np.exp(-r*tau)*phi_two
    Put = Call - S + K*np.exp(-r*tau)
    
    return Call, Put

spy = yf.Ticker("SPY")
opt = spy.option_chain('2021-02-22')
call = opt.calls
call = call.dropna()
strikes = call['strike']
iv = call['impliedVolatility']
plt.figure()
plt.plot(strikes,iv)
plt.xlabel('Strike Price')
plt.ylabel('Implied Volatility')
plt.title('SPY implied volatility 2021-03-26')
plt.grid()
plt.show()