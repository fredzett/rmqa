import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns


# Check if in interactive mode (i.e. in a Notebook environment)
def is_interactive():
    import __main__ as main
    return not hasattr(main, '__file__')


DATAPATH = "../../02_Datasets/"

if is_interactive():
    kernel = str(get_ipython())
    if "google.colab" in kernel:
        DATAPATH = "Datasets"
    elif "ipykernel" in kernel:
        DATAPATH = "../../02_Datasets/"
    else:
        DATAPATH = "../../02_Datasets/" # CHECK if there is another alternative we need
    print(kernel)



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
    def salaries(self):
        # TODO!
        'Loads salaries data as numpy'
        path = DATAPATH + "salaries.csv"
        df = pd.read_csv(path)
        data = [list(df.to_numpy()[:,i]) for i in range(df.shape[1])] # convert to list of len 3
        low, medium, high = data[0], data[1], data[2]
        return low, medium, high


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
    
    _, ax = plt.subplots(figsize=(9,7))
    ax = _plot_formatter(ax,xlabel,ylabel, title)

    if not isinstance(y, list): y = [y]
    
    for series in y:  
        ax.plot(x,series)
    
    # 
    if zero_origin: ax.spines['bottom'].set_position('zero')
    
    
    #for s in ["top","right"]:
    #    ax.spines[s].set_visible(False)
   
    return ax

