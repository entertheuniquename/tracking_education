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

def estimate(filter,x0,P0,Q0,R,Z,T):
        estimator = filter(x0,P0,Q0,R)
        est = np.zeros((x0.shape[0], Z.shape[1]-1))
        for col in range(est.shape[1]):
            z = Z[:, col + 1]
            xp = estimator.predict(T)
            m1 = np.array([z[0], z[1], z[2]])
            xc = estimator.correct(m1.T)
            est[:, col] = np.squeeze(xc[:])
        return est

def calc_err(X,filter,x0,P0,Q0,R,T,measureModel,gModel):
    Q = gModel(T)@Q0@gModel(T).T
    Xn = add_process_noise(X,Q)
    Zn = make_meas(Xn,R,measureModel)
    est = estimate(filter,x0,P0,Q0,R,Zn,T)
    err = est - Xn[:, 1:]
    return err

from tqdm import tqdm

def calc_std_err(X,filter,x0,P0,Q0,R,measureModel,gModel,T,num_iterations):
    var_err = np.zeros((X.shape[0], X.shape[1]-1))
    for i in tqdm(range(num_iterations)):
        err = calc_err(X,filter,x0,P0,Q0,R,T,measureModel,gModel)
        var_err += err ** 2
    var_err /= num_iterations
    return np.sqrt(var_err)

def test(filter,x0,P0,Q0,R,stateModel,measureModel,noiseTransitionModel,T,amount,iterations):
    X=make_true_data(x0,amount,stateModel,T)
    #Q = noiseTransitionModel(T)@Q0@noiseTransitionModel(T).T
    #Xn = add_process_noise(X,Q)
    #Zn = make_meas(Xn,R,measureModel)
    #est = estimate(filter,x0,P0,Q0,R,Zn,T)
    std_err = calc_std_err(X,filter,x0,P0,Q0,R,measureModel,noiseTransitionModel,T,iterations)
    return X, std_err
