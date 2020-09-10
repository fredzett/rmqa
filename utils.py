import numpy as np

def choice(data,n):
    return np.random.choice(data,size=n)

def normal(loc,scale,size):
    return np.random.normal(loc,scale,size)

def where(cond,conseq,alt):
    n_x = sum(cond)
    n_y = sum(~cond)
    ret = np.zeros(n_x+n_y)
    loc_x,scale_x = conseq
    loc_y,scale_y = alt
    x = normal(loc_x,scale_x,n_x)
    y = normal(loc_y,scale_y,n_y)
    ret[cond] = x
    ret[~cond] = y
    return ret