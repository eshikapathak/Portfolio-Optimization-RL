import pandas as pd
import numpy as np 
from scipy import optimize 
from scipy.optimize import linprog
from pypfopt.efficient_frontier import EfficientFrontier

# Script obtains maximal return portfolio using linear programming

def MaximizeReturns(MeanReturns, PortfolioSize):
        
  c = (np.multiply(-1, MeanReturns))
  A = np.ones([PortfolioSize,1]).T
  b=[1]
  res = linprog(c, A_ub = A, b_ub = b, bounds = (0,1), method = 'simplex') 
    
  return res

def MinimizeRisk(CovarReturns, PortfolioSize):
    
  def f(x, CovarReturns):
    func = np.matmul(np.matmul(x, CovarReturns), x.T) 
    return func

  def constraintEq(x):
    A=np.ones(x.shape)
    b=1
    constraintVal = np.matmul(A,x.T)-b 
    return constraintVal
    
  xinit=np.repeat(0.1, PortfolioSize)
  cons = ({'type': 'eq', 'fun':constraintEq})
  lb = 0
  ub = 1
  bnds = tuple([(lb,ub) for x in xinit])

  opt = optimize.minimize (f, x0 = xinit, args = (CovarReturns),  bounds = bnds, \
                             constraints = cons, tol = 10**-3)
    
  return opt

def MinimizeRiskConstr(MeanReturns, CovarReturns, PortfolioSize, R):
    
  def  f(x,CovarReturns):
    func = np.matmul(np.matmul(x,CovarReturns ), x.T)
    return func

  def constraintEq(x):
    AEq=np.ones(x.shape)
    bEq=1
    EqconstraintVal = np.matmul(AEq,x.T)-bEq 
    return EqconstraintVal
    
  def constraintIneq(x, MeanReturns, R):
    AIneq = np.array(MeanReturns)
    bIneq = R
    IneqconstraintVal = np.matmul(AIneq,x.T) - bIneq
    return IneqconstraintVal
    

  xinit=np.repeat(0.1, PortfolioSize)
  cons = ({'type': 'eq', 'fun':constraintEq},
          {'type':'ineq', 'fun':constraintIneq, 'args':(MeanReturns,R) })
  lb = 0
  ub = 1
  bnds = tuple([(lb,ub) for x in xinit])

  opt = optimize.minimize (f, args = (CovarReturns), method ='trust-constr',  \
                x0 = xinit,   bounds = bnds, constraints = cons, tol = 10**-3)
    
  return opt

def StockReturnsComputing(StockPrice, Rows, Columns): 
    StockReturn = np.zeros([Rows-1, Columns]) 
    for j in range(Columns):        # j: Assets 
        for i in range(Rows-1):     # i: Daily Prices 
            StockReturn[i,j]=((StockPrice[i+1, j]-StockPrice[i,j])/StockPrice[i,j])* 100 
        
    return StockReturn

def mvo_opt(mvo_df, stock_dimension):
    fst = mvo_df
    fst = fst.iloc[0*stock_dimension:0*stock_dimension+stock_dimension, :]
    tic = fst['tic'].tolist()

    mvo = pd.DataFrame()

    for k in range(len(tic)):
        mvo[tic[k]] = 0

    for i in range(mvo_df.shape[0]//stock_dimension):
        n = mvo_df
        n = n.iloc[i*stock_dimension:i*stock_dimension+stock_dimension, :]
        date = n['date'][i*stock_dimension]
        mvo.loc[date] = n['close'].tolist()


    portfolioSize = stock_dimension #set portfolio size

    StockData = mvo.head(mvo.shape[0]-336)
    TradeData = mvo.tail(336)
    TradeData.to_numpy()

    arStockPrices = np.asarray(StockData)
    [Rows, Cols]=arStockPrices.shape
    arReturns = StockReturnsComputing(arStockPrices, Rows, Cols)

    meanReturns = np.mean(arReturns, axis = 0)
    covReturns = np.cov(arReturns, rowvar=False)
    np.set_printoptions(precision=3, suppress = True)
    # print('Mean returns of assets in k-portfolio 1\n', meanReturns)
    # print('Variance-Covariance matrix of returns\n', covReturns)

    ef_mean = EfficientFrontier(meanReturns, covReturns, weight_bounds=(0, 0.5))
    raw_weights_mean = ef_mean.max_sharpe()
    cleaned_weights_mean = ef_mean.clean_weights()
    mvo_weights = np.array([1000000 * cleaned_weights_mean[i] for i in range(stock_dimension)])
    LastPrice = np.array([1/p for p in StockData.tail(1).to_numpy()[0]])
    Initial_Portfolio = np.multiply(mvo_weights, LastPrice)
    Portfolio_Assets = TradeData @ Initial_Portfolio
    MVO_result = pd.DataFrame(Portfolio_Assets, columns=["Mean Var"])

    return MVO_result








