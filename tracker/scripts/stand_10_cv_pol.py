#!/usr/bin/python3

import numpy as np
import numpy.matlib as ml
import matplotlib.pyplot as plt
import math
from IPython.display import display, Math, Latex
from IPython.display import Markdown as md

import estimator as e
import math

def rot_z(a):
    a = np.deg2rad(a)
    R = np.eye(3)
    ca = np.cos(a)
    sa = np.sin(a)
    R[0,0] = ca
    R[0,1] = -sa
    R[1,0] = sa
    R[1,1] = ca
    return R

def rot_y(a):
    a = np.deg2rad(a)
    R = np.eye(3,3)
    ca = np.cos(a)
    sa = np.sin(a)
    R[0,0] = ca
    R[0,2] = sa
    R[2,0] = -sa
    R[2,2] = ca
    return R

def sph2cart(measurement):
    cart = np.zeros(measurement.shape)
    for i in range(measurement.shape[1]):
        cart[0, i] = measurement[2, i] * math.cos(measurement[1, i]) * math.cos(measurement[0, i])
        cart[1, i] = measurement[2, i] * math.sin(measurement[1, i]) * math.cos(measurement[0, i])
        cart[2, i] = measurement[2, i] * math.sin(measurement[0, i])
    return cart

def sph2cartcov(sphCov, az, el, r):
    pa = 0
    pe = 1
    pr = 2
    pvr= 3

    azSig  = np.sqrt(sphCov[pa, pa])
    elSig  = np.sqrt(sphCov[pe, pe])
    rngSig = np.sqrt(sphCov[pr, pr])

    Rpos = np.diag([np.power(rngSig,2.0),
                    np.power(r*np.cos(np.deg2rad(el))*np.deg2rad(azSig), 2.0),
                    np.power(r*np.deg2rad(elSig), 2.0)])
    rot = rot_z(az)@rot_y(el).T
    posCov = rot@Rpos@rot.T
    if sphCov.shape==(4, 4):
        rrSig = np.sqrt(sphCov[pvr, pvr])
        crossVelSig = 10
        Rvel = np.diag([np.power(rrSig, 2.0), np.power(crossVelSig, 2.0), np.power(crossVelSig, 2.0)]);
        velCov = rot@Rvel@rot.T
    else:
        velCov = 100*np.eye(3)

    return posCov, velCov

def make_cartcov(meas, covMeas):
    az = np.rad2deg(meas[1,0])
    el = np.rad2deg(meas[0,0])
    r  = meas[2,0]
    sphCov = np.diag([covMeas[0,0], covMeas[2,2], covMeas[1,1]])
    [posCov, velCov] = sph2cartcov(sphCov, az, el, r)
    Hp = np.array([[1, 0, 0, 0, 0, 0],
                   [0, 0, 1, 0, 0, 0],
                   [0, 0, 0, 0, 1, 0]])
    Hv = np.array([[0, 1, 0, 0, 0, 0],
                   [0, 0, 0, 1, 0, 0],
                   [0, 0, 0, 0, 0, 1]])
    return Hp.T@posCov@Hp + Hv.T@velCov@Hv

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
        states_predict = np.zeros((x0.shape[0], Z.shape[1]-1))
        meases_predict = np.zeros((3, Z.shape[1]-1))
        for col in range(est.shape[1]):
            z = Z[:, col + 1]
            xp = estimator.predict(T)
            m1 = np.array([z[0], z[1], z[2]])
            xc = estimator.correct(m1.T)
            meas_predict = estimator.m_predict()
            state_predict = estimator.s_predict()
            est[:, col] = np.squeeze(xc[:])
            meases_predict[:, col] = np.squeeze(meas_predict[:])
            states_predict[:, col] = np.squeeze(state_predict[:])

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
    std_err = calc_std_err(X,filter,x0,P0,Q0,R,measureModel,noiseTransitionModel,T,iterations)
    return X, std_err
