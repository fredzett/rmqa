import numpy as np
import pandas as pd
from pathlib import Path

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


class Datasets():
    'Helper class to load data sets'

    ## Loading functions
    @classmethod
    def dax_monthly(self, output="list"):
        print(Path.cwd())
        path = "../../02_Datasets/dax.txt"
        df = (pd.read_csv(path, usecols=["Adj Close"])
             .rename({"Adj Close": "Price"},axis=1)
             )
        return df


data = Datasets.dax_monthly()
print(data)
