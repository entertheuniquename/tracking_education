#!/usr/bin/python3

import numpy as np
import numpy.matlib as ml
import matplotlib.pyplot as plt
import math
from IPython.display import display, Math, Latex
from IPython.display import Markdown as md
#from arr2ltx import convert2latex, to_latex

import estimator as e
import math

Ptd = 0.9 #вероятность правильного обнаружения
pass_am = round(100*(1-Ptd)) #колличество пропусков в наборе из 100

T = 6
process_var = 1
meas_var = 300
velo_var = 30

R = np.diag([meas_var*meas_var, meas_var*meas_var, meas_var*meas_var])
Rvel = np.diag([velo_var*velo_var, velo_var*velo_var, velo_var*velo_var])

Rpol = np.diag([1e-4, 1e-4, 1.0])

Q0 = np.diag([process_var, process_var, process_var])
G = np.array([[T**2/2, 0,      0     ],
              [T,      0,      0     ],
              [0,      T**2/2, 0     ],
              [0,      T,      0     ],
              [0,      0,      T**2/2],
              [0,      0,      T     ]])

Q = G@Q0@G.T

initialState = np.array([-40., -200., 0., 0., 0., 0.])
initialState = initialState[:, np.newaxis]

u = np.zeros((6,1))
B = np.zeros((6,6))

max_speed = 4

def make_true_data(x0):
    X = np.zeros((x0.shape[0], 100))
    X[:, 0] = x0.T
    for i in range(X.shape[1]-1):
        xx = e.stateModel_CVx(np.copy(X[:, i]),T)
        xx1 = xx.flatten()
        X[:, i+1] = xx.flatten()
    return X

X=make_true_data(initialState)

def add_process_noise(X,Var):
    Xn = X + np.sqrt(Var) @ np.random.normal(loc=0, scale=1.0, size=(X.shape[0], X.shape[1]))#<-здесь[x]
    return Xn

Xn = add_process_noise(X,Q)

######################################################################################################
######################################################################################################

def make_meas(X, R):
    Z = np.zeros((R.shape[0], X.shape[1]))
    for i in range(Z.shape[1]):
        zz = e.measureModel_XXx(np.copy(X[:, i]))#<- здесь[x]
        Z[:, i] = zz.flatten()
    Zn = Z + np.sqrt(R) @ np.random.normal(loc=0, scale=math.sqrt(1.0), size=(Z.shape[0], Z.shape[1]))
    return Zn

Zn = make_meas(Xn, R)

######################################################################################################

def make_tu(TP,M):
    tu = (TP,M)
    return tu

def stepKFE(Zn):
    rnd = np.random.randint(0,98,pass_am)
    time = 0
    track = e.BindTrackKFE(make_tu(time,Zn[:, 0][:, np.newaxis]))
    est = np.zeros((6, Zn.shape[1]-1))#6 - bad
    for col in range(Zn.shape[1]-1):
        time = time+6
        z = Zn[:, col+1]
        if col in rnd:
            ee = track.step(time)
        else:
            ee = track.step(make_tu(time,z))
        est[:, col] = np.squeeze(ee[:])
    return est

est=stepKFE(Zn)

######################################################################################################
######################################################################################################

def make_meas_pol(X, R):
    Z = np.zeros((R.shape[0], X.shape[1]))
    for i in range(Z.shape[1]):
        xx = np.copy(X[:, i])
        zz = e.measureModel_XRx(np.copy(X[:, i]))
        Z[:, i] = zz.flatten()
    Zn = Z + np.sqrt(R) @ np.random.normal(loc=0, scale=math.sqrt(1.0), size=(Z.shape[0], Z.shape[1]))
    return Zn

Zn_pol = make_meas_pol(Xn, Rpol)

def sph2cart(measurement):
    cart = np.zeros(measurement.shape)
    for i in range(measurement.shape[1]):
        cart[0, i] = measurement[2, i] * math.cos(measurement[1, i]) * math.cos(measurement[0, i])
        cart[1, i] = measurement[2, i] * math.sin(measurement[1, i]) * math.cos(measurement[0, i])
        cart[2, i] = measurement[2, i] * math.sin(measurement[0, i])
    return cart

cart_pol = sph2cart(Zn_pol)

######################################################################################################

def stepEKFE(Znn):
    rnd = np.random.randint(0,98,pass_am)
    time = 0
    track = e.BindTrackEKFE(make_tu(time,Znn[:, 0][:, np.newaxis]))
    est = np.zeros((6, Znn.shape[1]-1))#6 - bad
    for col in range(Znn.shape[1]-1):
        time = time+6
        z = Znn[:, col+1]
        if col in rnd:
            ee = track.step(time)
        else:
            ee = track.step(make_tu(time,z))
        est[:, col] = np.squeeze(ee[:])
    return est

est2=stepEKFE(Zn_pol)

######################################################################################################
######################################################################################################

fig = plt.figure(figsize=(9,25))
ax1 = fig.add_subplot(4,1,1)
ax2 = fig.add_subplot(4,1,2)
ax3 = fig.add_subplot(4,1,3)
ax4 = fig.add_subplot(4,1,4)

ax1.plot(X[0, :], X[2, :], label='Truth')
ax1.plot(Xn[0, :], Xn[2, :], label='Truth+noise')
ax1.plot(Zn[0, :], Zn[2, :], label='Measurement', linestyle='', marker='+')
ax1.plot(est[0, :], est[2, :], label='Estimates')
#ax1.set_title("[x,vx,y,vy,z,vz]")
ax1.set_xlabel('x,met.')
ax1.set_ylabel('y,met.')
ax1.grid(True)

ax2.plot(X[0, :], X[2, :], label='Truth')
ax2.plot(Xn[0, :], Xn[2, :], label='Truth+noise')
ax2.plot(cart_pol[0, :], cart_pol[2, :], label='Measurement', linestyle='', marker='+')
ax2.plot(est2[0, :], est2[2, :], label='Estimates')
#ax2.set_title("[x,vx,y,vy,z,vz]")
ax2.set_xlabel('x,met.')
ax2.set_ylabel('y,met.')
ax2.grid(True)

plt.show()

######################################################################################################
######################################################################################################

def calc_err(X):
    Xn = add_process_noise(X,Q)
    Zn = make_meas(Xn,R)
    est = stepKFE(Zn)
    err = est - Xn[:, 1:]
    return err

def calc_err2(X):
    Xn = add_process_noise(X,Q)
    Zn = make_meas_pol(Xn,Rpol)
    est = stepEKFE(Zn)
    err = est - Xn[:, 1:]
    return err

from tqdm import tqdm

def calc_std_err(X):
    num_iterations = 2000
    var_err = np.zeros((X.shape[0], X.shape[1]-1))

    for i in tqdm(range(num_iterations)):
        err = calc_err(X)
        var_err += err ** 2

    var_err /= num_iterations
    return np.sqrt(var_err)

def calc_std_err2(X):
    num_iterations = 2000
    var_err = np.zeros((X.shape[0], X.shape[1]-1))

    for i in tqdm(range(num_iterations)):
        err = calc_err2(X)
        var_err += err ** 2

    var_err /= num_iterations
    return np.sqrt(var_err)

std_err = calc_std_err(X)
std_err2 = calc_std_err2(X)

######################################################################################################
######################################################################################################

plt.figure(figsize=(9,45))

# [x,vx,y,vy,z,vz]

plt.subplot(6, 1, 1)
plt.plot((np.arange(len(std_err[0, :]))+1)*T, std_err[0, :].T)
plt.plot((np.arange(len(std_err2[0, :]))+1)*T, std_err2[0, :].T)
plt.grid(True)
plt.title("[x,vx,y,vy,z,vz]")
plt.xlabel('Time,s')
plt.ylabel('std_x, met')

plt.subplot(6, 1, 2)
plt.plot((np.arange(len(std_err[1, :]))+1)*T, std_err[1, :].T)
plt.plot((np.arange(len(std_err2[1, :]))+1)*T, std_err2[1, :].T)
plt.grid(True)
plt.xlabel('Time,s')
plt.ylabel('std_vx, m/s')

plt.subplot(6, 1, 3)
plt.plot((np.arange(len(std_err[2, :]))+1)*T, std_err[2, :].T)
plt.plot((np.arange(len(std_err2[2, :]))+1)*T, std_err2[2, :].T)
plt.grid(True)
plt.xlabel('Time,s')
plt.ylabel('std_y, met')

plt.subplot(6, 1, 4)
plt.plot((np.arange(len(std_err[3, :]))+1)*T, std_err[3, :].T)
plt.plot((np.arange(len(std_err2[3, :]))+1)*T, std_err2[3, :].T)
plt.grid(True)
plt.xlabel('Time,s')
plt.ylabel('std_vy, m/s')

plt.subplot(6, 1, 5)
plt.plot((np.arange(len(std_err[4, :]))+1)*T, std_err[4, :].T)
plt.plot((np.arange(len(std_err2[4, :]))+1)*T, std_err2[4, :].T)
plt.grid(True)
plt.xlabel('Time,s')
plt.ylabel('std_z, met')

plt.subplot(6, 1, 6)
plt.plot((np.arange(len(std_err[5, :]))+1)*T, std_err[5, :].T)
plt.plot((np.arange(len(std_err2[5, :]))+1)*T, std_err2[5, :].T)
plt.grid(True)
plt.xlabel('Time,s')
plt.ylabel('std_vz, m/s')

plt.subplots_adjust(wspace=8.0, hspace=0.7)
plt.show()
