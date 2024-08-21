#!/usr/bin/python3

import numpy as np
import numpy.matlib as ml
import matplotlib.pyplot as plt
import math
from IPython.display import display, Math, Latex
from IPython.display import Markdown as md

import estimator as e
import math

def diveLines(amount):
    line_str = amount/2
    line_dive = line_str
    if(line_str%2>0):
        line_str = math.ceil(line1)
        line_dive = math.floor(line2)
    line_str1 = line_str/2
    line_str2 = line_str1
    if(line_str1%2>0):
        line_str1 = math.ceil(line_str1)
        line_str2 = math.floor(line_str2)
    if(amount != (line_str1+line_str2+line_dive)):
        print("ERROR LINES!")
    return int(line_str1),int(line_dive),int(line_str2)

def make_true_data(x0,amount,stateModel,T,overload,delta_z):
    X = np.zeros((x0.shape[0], amount))
    X[:, 0] = x0.T
    [l1,l2,l3] = diveLines(amount)
    for i in range(X.shape[1]-1):
        xxx = np.copy(X[:, i])
        xxx[-2] = 0
        xxx[-3] = 0
        if (i>l1-1):
            xxx[-2] = overload
            xxx[-3] = delta_z
        if (i>l1+l2-1):
            xxx[-2] = 0
            xxx[-3] = 0
        xx = stateModel(xxx,T)
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

def estimate(filter,x0,P0,Q0,R,Z,T,**imm_params):
        mu=np.zeros((1,3))
        tp=np.zeros((3,3))
        is_imm = False
        for key, value in imm_params.items():
            if(key=="MU"):
                mu=value
                is_imm = True
            if(key=="TP"):
                tp=value

        if is_imm == True:
            estimator = filter(x0,P0,Q0,R,mu,tp)
        else:
            estimator = filter(x0,P0,Q0,R)
        est = np.zeros((x0.shape[0], Z.shape[1]-1))
        mus = np.zeros((3, Z.shape[1]-1))
        for col in range(est.shape[1]):
            z = Z[:, col + 1]
            xp = estimator.predict(T)
            m1 = np.array([z[0], z[1], z[2]])
            xc = estimator.correct(m1.T)
            est[:, col] = np.squeeze(xc[:])
            if is_imm ==True:
                mus[:, col] = np.squeeze(estimator.mu())
        if is_imm ==True:
            return est, mus
        else:
            return est

def calc_err(X,filter,x0,P0,Q0,R,T,measureModel,gModel,**args):
    mu=np.zeros((1,3))
    tp=np.zeros((3,3))
    is_imm = False
    for key, value in args.items():
        if(key=="MU"):
            mu=value
            is_imm = True

    Q = gModel(T)@Q0@gModel(T).T
    Xn = add_process_noise(X,Q)
    Zn = make_meas(Xn,R,measureModel)
    if is_imm == True:
        est, mus = estimate(filter,x0,P0,Q0,R,Zn,T,**args)
    else:
        est = estimate(filter,x0,P0,Q0,R,Zn,T,**args)
    err = est - Xn[:, 1:]
    return err

from tqdm import tqdm
#import multiprocessing
#process = multiprocessing.Process(target=task)

def calc_std_err(X,filter,x0,P0,Q0,R,measureModel,gModel,T,num_iterations,**args):
    #process.start()
    var_err = np.zeros((X.shape[0], X.shape[1]-1))
    for i in tqdm(range(num_iterations)):
        err = calc_err(X,filter,x0,P0,Q0,R,T,measureModel,gModel,**args)
        var_err += err ** 2
    var_err /= num_iterations
    #process.close()
    return np.sqrt(var_err)

def test(filter,x0,P0,Q0,R,stateModel,measureModel,noiseTransitionModel,T,amount,iterations,overload,delta_z,**args):
    X=make_true_data(x0,amount,stateModel,T,overload,delta_z)
    #Q = noiseTransitionModel(T)@Q0@noiseTransitionModel(T).T
    #Xn = add_process_noise(X,Q)
    #Zn = make_meas(Xn,R,measureModel)
    #est = estimate(filter,x0,P0,Q0,R,Zn,T)
    std_err = calc_std_err(X,filter,x0,P0,Q0,R,measureModel,noiseTransitionModel,T,iterations,**args)
    return X, std_err
