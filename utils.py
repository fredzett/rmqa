import numpy as np
import pandas as pd
from pathlib import Path

import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from patsy import dmatrices


DATAPATH = './data/'

class Datasets():
    'Helper class to load data sets'

    ## Loading functions
    @classmethod
    def dax_monthly(self, dtype="numpy"):
        'Loads monthly dax data'
        path = DATAPATH + "Dax_monthly_prices.csv"
        df = pd.read_csv(path,index_col="Date")
        if dtype.lower() == "numpy":
            return df["Price"].values
        elif dtype.lower() == "pandas":
            return df
        else: 
            raise AttributeError("dtype should be 'numpy' or 'pandas'")
            
    @classmethod
    def dax_daily(self, dtype="numpy"):
        path = DATAPATH + "Dax_daily.csv"
        df = pd.read_csv(path,index_col="Date")
        if dtype.lower() == "numpy":
            return df["Adj Close"].dropna().values
        elif dtype.lower() == "pandas":
            return df
        else: 
            raise AttributeError("dtype should be 'numpy' or 'pandas'")
        

    @classmethod
    def salaries(self):
        # TODO!
        'Loads salaries data as numpy'
        path = DATAPATH + "salaries.csv"
        df = pd.read_csv(path)
        data = [list(df.to_numpy()[:,i]) for i in range(df.shape[1])] # convert to list of len 3
        low, medium, high = data[0], data[1], data[2]
        return low, medium, high
    
    
    @classmethod
    def online_spent(self, col=["spend"], dtype="numpy"):
        '''Loads online spend data (source: https://raw.githubusercontent.com/TaddyLab/MBAcourse/master/examples/web-browsers.csv)
        
        Columns:
        - id
        - anychildren
        - broadband
        - hispanic
        - race
        - region
        - spend
        
        '''
        path = DATAPATH + "online_spent.csv"
        df = pd.read_csv(path)
        
        if not isinstance(col,list): col = [col]
        
        if not col == ["all"]: df = df[col]

        
        if dtype.lower() == "numpy":
            return df.values
        elif dtype.lower() == "pandas":
            return df
        else:
            raise AttributeError("dtype should be 'numpy' or 'pandas'")
            
    @classmethod
    def fake_spent_children(self, n=5_000):
        '''Create fake spent data
        
        Input:
        n = size of sample
        
        Output:
        data = (n,2) array, with 
        
        sales = fake online spent data
        children = fake has children data (1 = yes, 0=no)
        '''
        np.random.seed(123875)
        p = 0.36
        sales = np.random.choice(np.arange(10000,50001), n)
        children = np.random.choice([0,1], n, p=[p,1-p])
        data = np.vstack((sales, children)).T
        return data
    
    @classmethod
    def credit_card_debt(self, n=70):
        'Creates fake credit card debt data (n=70)'
        np.random.seed(1235)
        return random_normal_clip(10000,4000,n,98,50000)
    
    @classmethod
    def advertising(self, dtype="numpy", col=["TV", "sales"]):
        'Loads advertising data set from ISLR'
        path = DATAPATH + "advertising.csv"
        df = pd.read_csv(path)
        
        if not isinstance(col,list): col = [col]
        
        if not col == ["all"]: df = df[col]

        
        if dtype.lower() == "numpy":
            return df.values
        elif dtype.lower() == "pandas":
            return df
        else:
            raise AttributeError("dtype should be 'numpy' or 'pandas'")
            
    @classmethod        
    def cars(self, dtype="numpy", col=["mpg", "horsepower"]):
        path = DATAPATH + "Auto.csv"
        df = pd.read_csv(path)
        
        if not isinstance(col, list): col = [col]
        if not col == ["all"]: df = df[col]
            
        if dtype.lower() == "numpy":
            return df.values
        elif dtype.lower() == "pandas":
            return df
        
        else:
            raise AttributeError("dtype should be 'numpy' or 'pandas'")
    
    @classmethod
    def fashion(self):
        path = DATAPATH + "Fashion.csv"
        df = pd.read_csv(path)
        return df

###### Helper functions

def random_normal_clip(loc,std,size, low,high):
    '''
    Draw random samples from a (clipped) normal (Gaussian) distribution. 
    
    - low = minimal value for draw 
    - high = maximum value for draw
    
    (if low = -np.inf and high=np.inf normal distribution is returned)
    Draw is repeated until sample size is reached (with low/high condition)
    
    '''
    data = []
    while len(data) != size:
        v = np.random.normal(loc,std)
        if (v > low) & (v < high): data.append(v)
    return data



##### PLOTS
def _plot_formatter(ax, xlabel, ylabel, title):
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    if title: ax.set_title(title)
        
    for s in ["top","right"]:
        ax.spines[s].set_visible(False)
    return ax

def plot_line(x,y,xlabel=None, ylabel=None, title=None,zero_origin=True):
    '''
    Plots simple line chart
    
    INPUT:
    x = data for x-axis
    y = data for y-axis; if list line is drawn per list element
    xlabel, ylabel, title = string with labels
    zero_origin = if True x-axis goes through 0
    '''
    
    fig, ax = plt.subplots(figsize=(9,7))
    ax = _plot_formatter(ax,xlabel,ylabel, title)

    if (not isinstance(y, list)): y = [y]
    
    for series in y:  
        ax.plot(x,series)

     
    if zero_origin: 
        ax.spines['bottom'].set_position('zero')
        ax.spines['left'].set_position('zero')
    
    return fig, ax


def plot_step(x,y,xlabel=None, ylabel=None, title=None, exclusion_points=True):
    '''
    Plots step chart (for discrete cdf)
    
    INPUT:
    x = data for x-axis
    y = data for y-axis
    xlabel, ylabel, title = string with labels
    exclusion_points = if True point until valid is marked
    '''
    
    fig, ax = plt.subplots(figsize=(9,7))
    ax = _plot_formatter(ax,xlabel,ylabel, title)

    ax.hlines(y,xmin=x, xmax=x+1,color="darkblue")
    
    if exclusion_points: 
        sc = ax.scatter(x+1,y,color="black")
        sc.set_facecolor("none")
     
    return fig, ax


def plot_bar(x,y,xlabel=None, ylabel=None, title=None, zero_origin=True):
    '''
    Plots simple bar chart
    
    INPUT:
    x = data for x-axis
    y = data for height; if list line is drawn per list element
    xlabel, ylabel, title = string with labels
    zero_origin = if True x-axis goes through 0
    '''
    
    fig, ax = plt.subplots(figsize=(9,7))
    ax = _plot_formatter(ax,xlabel,ylabel, title)
    
    if not isinstance(y, list): y = [y]
        
    for series in y:
        ax.bar(x,series)
       
    if zero_origin: ax.spines['bottom'].set_position('zero')
        
    return fig, ax


def plot_hist(data, show_prob=False, xlabel=None, ylabel=None, title=None):
    
    fig, ax = plt.subplots(figsize=(9,7))
    ax = _plot_formatter(ax,xlabel,ylabel, title)
     
    values, height = np.unique(data,return_counts=True)
    
    if show_prob: height = height/len(data)
    
    ax.bar(values,height)

    return fig, ax

def plot_density(data,kde=True,xlabel=None, ylabel=None, title=None):
    fig, ax = plt.subplots(figsize=(9,7))
    ax = _plot_formatter(ax,xlabel,ylabel, title)
    
    if not isinstance(data, list): data = [data]
    
    for d in data:
        sns.distplot(d,ax=ax, kde=kde)
    return fig, ax


def lreg_summary(X,y, make_intercept=True):
    if make_intercept: X = sm.add_constant(X)
    return sm.OLS(y,X).fit().summary()


def make_y_X(s: str, data:pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    '''Generates two dataframes based on model formulation
    
    Function is essentially a wrapper around patsy.dmatrices

    INPUT:
    s = string representing the model (patsy style)

    data = pandas dataframe holding the variables 

    OUTPUT:
    y = dataframe holding dependent variable
    X = dataframe holding independent variable(s)
    '''

    y, X = dmatrices(s, data=data, return_type="dataframe")
    return y, X