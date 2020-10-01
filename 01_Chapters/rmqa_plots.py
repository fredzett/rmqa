import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_pmf(x,probs, xlabel=None, ylabel="Probability",title=None):
    'Plots histogram with actual probabilities'
    sns.barplot(x,probs,color="blue");
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title);
    