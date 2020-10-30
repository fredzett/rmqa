import streamlit as st
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from matplotlib import cm
import seaborn as sns 
import statsmodels.formula.api as smf


#@st.cache
def load_data():
    df = pd.read_csv("data/Advertising.csv") # read csv file into pandas
    df = df.drop(columns="Unnamed: 0")
    return df

def _ols_best(df):
    X = df.iloc[:,1].name
    y = df.iloc[:,0].name
    model = smf.ols(f"{y} ~ {X}", df).fit()
    b0,b1 = model.params.values
    yhat_best = model.predict(df.loc[:,X])
    return yhat_best, b0, b1

def _rss(y, yhat):
    return np.sum((y - yhat)**2)


def _mse(y,X,b0,b1):
    yhat = b0 + b1*X
    return _rss(y,yhat)/len(y)


def _loss_grid(x,y,w0s, w1s):
    'Calculate losses for many w0 and w1'
    WW0, WW1 = np.meshgrid(w0s, w1s)
    W0 = np.ravel(WW0).reshape(1,-1)
    W1 = np.ravel(WW1).reshape(1,-1)

    x = x.reshape(-1,1)
    y = y.reshape(-1,1)
    yhat = W0 + np.dot(x,W1)

    mse = np.sum((y-yhat)**2,axis=0)# / len(y) # Calculate RSS
    mse = mse.reshape(WW0.shape)
    return WW0, WW1, mse

def plot_surface(xx,yy,zz, b0, b1, rss):
    fig = plt.figure(figsize=(9,7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(xx,yy,zz, cmap=cm.coolwarm,
                           linewidth=2, antialiased=False, alpha=0.4, rstride=1, cstride=1)
    ax.set_xlabel("b0", size=12)
    ax.set_ylabel("b1", size=12)
    ax.set_zlabel("RSS", size=12)
    ax.set_zlim(0,np.max(zz))
    
    ax.scatter(b0,b1,rss, c="red", s=100)
    return fig, ax

def _tss(y):
    return np.sum((y - np.mean(y))**2)

def _r2(y , yhat):
    rss = _rss(y,yhat)
    tss = _tss(y)
    return (1 - rss / tss)

def _F(y,yhat,p=1):
    rss = _rss(y,yhat)
    tss = _tss(y)
    F = ((tss - rss)/p) / (rss/(len(y)-p-1))
    return F

def plot(df,show_line=False, show_errors=False, show_best=False):
    fig, ax = plt.subplots(figsize=(9,7))
    y = df.iloc[:,0]
    X = df.iloc[:,1]
    yhat = df.iloc[:,2]
    yhat_best = df.iloc[:,3]
    ax.scatter(X,y,s=50, label="Data")
    if show_line:
        ax.plot(X,yhat, linewidth=3, color="red", label="yhat")
    if show_errors: 
        ax.vlines(X,ymin=y, ymax=yhat, color="gray", alpha=0.5, label="Residuals")
    if show_best:
        ax.plot(X,yhat_best,linewidth=3, color="black", label="yhat best", alpha=0.2)

    scaler = 1.2
    ymin, ymax = np.min(y), np.max(y)
    ax.set_xlabel(f"{X.name}",size=12)
    ax.set_ylabel(f"{y.name}", size=12)
    ax.set_ylim(ymin*(1-scaler), ymax*(scaler))
    ax.set_title(f'Simple linear regression of {y.name} on {X.name}');
    ax.legend(frameon=False, loc="upper left")
    sns.despine()
    return fig

# Load data
data = load_data()
cols = data.columns


# Sidebar
st.sidebar.markdown("**Choose X and y**")
y = st.sidebar.selectbox("Choose y",options=cols, index=len(cols)-1)
X = st.sidebar.selectbox("Choose X", options=[c for c in cols if c != y])
df = data[[y,X]]
b0, b1 = 5., 0.
betas = [b0, b1]
df["yhat"] = betas[0]+ betas[1] * df[X]
df["yhat_best"], bbest0, bbest1 = _ols_best(df)


st.sidebar.markdown('---')
st.sidebar.markdown('**Choose coefficients**')
scaler = bbest0
MIN0, MAX0 = float(np.round(bbest0-scaler,3)), float(np.round(bbest0+scaler, 3))
b0 = st.sidebar.slider("Intercept (b0)",float(MIN0),float(MAX0),value=float(MIN0), step=0.001)
scaler = bbest1
MIN1, MAX1 = float(np.round(bbest1-scaler,3)), float(np.round(bbest1+scaler, 3))
b1 = st.sidebar.slider("Slope (b1)",MIN1, MAX1, value=MIN1, step=0.001)

st.sidebar.markdown('---')
st.sidebar.markdown('**Choose plot options**')
show_regression = st.sidebar.checkbox("Show yhat \n(based on coefficients)")
show_errors = st.sidebar.checkbox("Show errors")
show_best = st.sidebar.checkbox("Show yhat (based on optimization)")


st.header("Simple linear regression")
# Specification
show_specification = st.beta_expander("Show model specification")
with show_specification:
    st.markdown(r'''
    Relationship is defined as 
    $$
    \hat{y}  = \beta_0 + \beta_1 X 
    $$''')
    st.write(f"In our case this means:")
    st.write(f"**{y}** = `{b0:.2f}`+ `{b1:.2f}`**{X}**.")

# Plot
betas = [b0, b1]
df["yhat"] = betas[0]+ betas[1] * df[X]
df["yhat_best"], bbest0, bbest1 = _ols_best(df)
show_plot = st.beta_expander("Show data plot")
with show_plot:
    fig = plot(df , show_line=show_regression, show_errors=show_errors, show_best=show_best)
    st.pyplot(fig)

# Assessing accuracy
show_accuracy = st.beta_expander("Assess model accuracy")
ytrue, yhat, ybest = df[y], df["yhat"], df["yhat_best"]
with show_accuracy:
    

    col1, col2 = st.beta_columns(2)
    with col1:
        col1.markdown("**Chosen model**")
        st.write(r'$\beta_0$ = ', np.round(b0,3))
        st.write(r'$\beta_1$ = ', np.round(b1,3))
        rss = _rss(ytrue, yhat)
        st.write(r'$\text{RSS}$ = ', np.round(rss,3))
        r2 = _r2(ytrue, yhat)
        st.write(r'$R^2$ = ', np.round(r2,3))
        F = _F(ytrue, yhat)
        st.write(r'$F\text{-statistic}$ = ', np.round(F,3))

    with col2:
        col2.markdown("**Optimized model**")
        st.write(r'$\beta_0$ = ', np.round(bbest0,3))
        st.write(r'$\beta_1$ = ', np.round(bbest1,3))
        rss = _rss(ytrue, ybest)
        st.write(r'$\text{RSS}$ = ', np.round(rss,3))
        r2 = _r2(ytrue, ybest)
        st.write(r'$R^2$ = ', np.round(r2,3))
        F = _F(ytrue, ybest)
        st.write(r'$F\text{-statistic}$ = ', np.round(F,3))

show_losses = st.beta_expander("Show RSS")
with show_losses:
    col1, col2 = st.beta_columns(2)
    angle1 = col1.slider("Vertical",0,360,value=24, step=1)
    angle2 = col2.slider("Horizontal",0,360,value=318, step=1)
    st.markdown("**RSS vs. coefficients**")
    xtrue = df[X].values
    ytrue_np = ytrue.values
    w0s = np.linspace(MIN0, MAX0, 50)
    w1s = np.linspace(MIN1, MAX1, 50)
    xx,yy,losses = _loss_grid(xtrue,ytrue_np,w0s, w1s)
    col1, col2 = st.beta_columns(2)

    fig, ax = plot_surface(xx,yy,losses,b0, b1, _rss(ytrue,yhat ))
    ax.view_init(angle1, angle2)
    st.pyplot(fig)











