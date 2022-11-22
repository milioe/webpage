import numpy as np
from scipy.stats import skew, kurtosis

def MV_criterion(weights, data):
    # parameters 
    Lambda = 1 # 3 is risk aversion
    W = 1 # wealth of the portfolio
    Wbar = 1+0.25/100 # risk free 
    
    # compute the portfolio returns
    portfolio_return = np.multiply(data, np.transpose(weights))
    portfolio_return = portfolio_return.sum(axis=1)
    
    # compute mean and volatiliy of the portfolio
    mean = np.mean(portfolio_return, axis=0)
    std = np.std(portfolio_return, axis=0)
    
    # Compute the criterion
    criterion =  Wbar ** (1 - Lambda) / (1 + Lambda) + Wbar ** (-Lambda) \
                * W * mean - Lambda / 2 * Wbar ** (-1 - Lambda) * W ** 2 * std **2
    criterion = -criterion
    
    return criterion


def SK_criterion(weights, data):
    Lambda = 3
    W = 1
    Wbar = 1+0.25/100
    portfolio_return = np.multiply(data, np.transpose(weights))
    portfolio_return = portfolio_return.sum(axis=1)
    
    mean = np.mean(portfolio_return, axis=0)
    std = np.std(portfolio_return, axis=0)
    skewness = skew(portfolio_return, 0)
    kurt = kurtosis(portfolio_return, 0)
    
    # Compute the criterion
    criterion = Wbar ** (1 - Lambda) / (1 + Lambda) + Wbar ** (-Lambda) \
    * W * mean - Lambda / 2 * Wbar ** (-1 - Lambda) * W ** 2 * std ** 2 \
    + Lambda * (Lambda + 1) / (6) * Wbar ** (-2 - Lambda) * W ** 3 * skewness \
    - Lambda * (Lambda + 1) * (Lambda + 2) / (24) * Wbar ** (-3 - Lambda) *\
     W ** 4 * kurt
    
    criterion = -criterion
    
    return criterion
    
    
def SR_criterion(weight, data):
    """ 
    -----------------------------------------------------------------------------
    | Output: Opposite Sortino ratio to do a mimization                         |
    -----------------------------------------------------------------------------
    | Inputs: -Weight (type ndarray numpy): Wheight for portfolio               |
    |         -data (type dataframe pandas): Returns of stocks                  |
    -----------------------------------------------------------------------------
    """
    # Compute portfolio returns
    portfolio_return = np.multiply(data, np.transpose(weight))
    portfolio_return = portfolio_return.sum(axis=1)

    # Compute mean, volatility of the portfolio
    mean = np.mean(portfolio_return, axis=0)
    std = np.std(portfolio_return, axis=0)

    # Compute the opposite of the Sharpe ratio
    Sharpe = mean / std
    Sharpe = -Sharpe
    return Sharpe



def SOR_criterion(weight, data):
    """ 
    -----------------------------------------------------------------------------
    | Output: Opposite Sortino ratio to do a m imization                        |
    -----------------------------------------------------------------------------
    | Inputs: -Weight (type ndarray numpy): Wheight for portfolio               |
    |         -data (type dataframe pandas): Returns of stocks                  |
    -----------------------------------------------------------------------------
    """
    # Compute portfolio returns
    portfolio_return = np.multiply(data, np.transpose(weight))
    portfolio_return = portfolio_return.sum(axis=1)

    # Compute mean, volatility of the portfolio
    mean = np.mean(portfolio_return, axis=0)
    std = np.std(portfolio_return[portfolio_return < 0], axis=0)

    # Compute the opposite of the Sharpe ratio
    Sortino = mean / std
    Sortino = -Sortino
    
    return Sortino