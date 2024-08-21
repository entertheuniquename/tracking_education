#!/usr/bin/python3

import numpy as np
import numpy.matlib as ml
import matplotlib.pyplot as plt
import math
from IPython.display import display, Math, Latex
from IPython.display import Markdown as md

import estimator as e
import math

def make_true_data(x0,amount,stateModel,T):
    X = np.zeros((x0.shape[0], amount))
    X[:, 0] = x0.T
    for i in range(X.shape[1]-1):
        xx = stateModel(np.copy(X[:, i]),T)
        xx1 = xx.flatten()
        X[:, i+1] = xx.flatten()
    return X

def add_process_noise(X,Var):
    Xn = X + np.sqrt(Var) @ np.random.normal(loc=0, scale=1.0, size=(X.shape[0], X.shape[1]))
    return Xn

def make_meas(X,R,measureModel):
    Z = np.zeros((R.shape[0], X.shape[1]))
    for i in range(Z.shape[1]):
        zz = measureModel(np.copy(X[:, i]))
        Z[:, i] = zz.flatten()
    Zn = Z + np.sqrt(R) @ np.random.normal(loc=0, scale=math.sqrt(1.0), size=(Z.shape[0], Z.shape[1]))
    return Zn
