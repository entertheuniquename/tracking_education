#!/usr/bin/python3

import numpy as np
import numpy.matlib as ml
import matplotlib.pyplot as plt
import math
from IPython.display import display, Math, Latex
from IPython.display import Markdown as md

import estimator as e
import math

def make_true_data(x0,stateModel,T):    
    vx = 200.
    l1 = 2000
    points1 = int(l1/vx/T) #10
    az = 29.4
    points2 = int(vx/az/T) #6
    points3 = points2
    l4 = 6000
    points4 = int(l4/vx/T) #30
    g9 = 88.2
    w = g9/vx
    points5 = int(3.14/w/T) #7
    points6 = int(vx/az/T)
    points7 = points6
    l8 = 3000
    points8 = int(l8/vx/T) #15
    points9 = int(vx/az/T)
    points10 = points9
    l11 = 2000
    points11 = int(l11/vx/T) #10
    X = np.zeros((x0.shape[0],int(points1+points2+points3+points4+points5+points6+points7+points8+points9+points10+points11)))
    X[:, 0] = x0.T
    stage = 1
    for i in range(X.shape[1]-1):
        xxx = np.copy(X[:, i])
        model = e.BindFCA_10

        if i>0 and i <=points1:
            xxx[1] = vx
        if i>points1 and i<=points1+points2:
            xxx[-2] = az
            xxx[2] = -az
        elif i>points2+points1 and i <=points1+points2+points3:
            xxx[-2] = -az
            xxx[2] = az
        elif i>points2+points1+points3 and i <=points1+points2+points2+points4:
            xxx[-2] = 0
            xxx[2] = 0
        elif i>points2+points1+points3+points4 and i <=points1+points2+points2+points4+points5:
            xxx[-1] = -w
            model = e.BindFCTv_10
        elif i>points2+points1+points3+points4+points5 and i <=points1+points2+points2+points4+points5+points6:
            xxx[-1] = 0.
            xxx[-2] = az
            xxx[2] = -az
            model = e.BindFCA_10
        elif i>points2+points1+points3+points4+points5+points6 and i <=points1+points2+points2+points4+points5+points6+points7:
            xxx[-2] = -az
            xxx[2] = az
            model = e.BindFCA_10
        elif i>points2+points1+points3+points4+points5+points6+points7 and i <=points1+points2+points2+points4+points5+points6+points7+points8:
            xxx[1] = -vx
            xxx[-3] = 0.
            xxx[-2] = 0.
            xxx[2] = 0.
        elif i>points2+points1+points3+points4+points5+points6+points7+points8 and i <=points1+points2+points2+points4+points5+points6+points7+points8+points9:
            xxx[-2] = -az
            xxx[2] = az
        elif i>points2+points1+points3+points4+points5+points6+points7+points8+points9 and i <=points1+points2+points2+points4+points5+points6+points7+points8+points9+points10:
            xxx[-2] = az
            xxx[2] = -az
        elif i>points2+points1+points3+points4+points5+points6+points7+points8+points9+points10 and i <=points1+points2+points2+points4+points5+points6+points7+points8+points9+points10+points11:
            xxx[-2] = 0.
            xxx[2] = 0.

        xx = model(xxx,T)

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
        for key, value in imm_params.items():
            if(key=="imm_filters_amount"):
                imm_am=value
        mu=np.zeros((1,imm_am))
        tp=np.zeros((imm_am,imm_am))
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
        mus = np.zeros((imm_am, Z.shape[1]-1))
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
    for key, value in args.items():
        if(key=="imm_filters_amount"):
            imm_am=value
    mu=np.zeros((1,imm_am))
    tp=np.zeros((imm_am,imm_am))
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

def calc_std_err(X,filter,x0,P0,Q0,R,measureModel,gModel,T,num_iterations,**args):
    var_err = np.zeros((X.shape[0], X.shape[1]-1))
    for i in tqdm(range(num_iterations)):
        err = calc_err(X,filter,x0,P0,Q0,R,T,measureModel,gModel,**args)
        var_err += err ** 2
    var_err /= num_iterations
    return np.sqrt(var_err)

def test(filter,x0,P0,Q0,R,stateModel,measureModel,noiseTransitionModel,T,amount,iterations,revers_conner_rad,**args):
    X=make_true_data(x0,stateModel,T)
    std_err = calc_std_err(X,filter,x0,P0,Q0,R,measureModel,noiseTransitionModel,T,iterations,**args)
    return X, std_err
