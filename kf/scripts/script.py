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

T = 6
process_var = 1
meas_var = 300
velo_var = 30

R = np.diag([meas_var*meas_var, meas_var*meas_var, meas_var*meas_var])
Rvel = np.diag([velo_var*velo_var, velo_var*velo_var, velo_var*velo_var])

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
        xx = e.stateModel_3Ax(np.copy(X[:, i]),T)
        xx1 = xx.flatten()
        X[:, i+1] = xx.flatten()
    return X

X=make_true_data(initialState)

def add_process_noise(X,Var):
    Xn = X + np.sqrt(Var) @ np.random.normal(loc=0, scale=1.0, size=(X.shape[0], X.shape[1]))#<-здесь[x]
    return Xn

Xn = add_process_noise(X,Q)

def make_meas(X, R):
    Z = np.zeros((R.shape[0], X.shape[1]))
    for i in range(Z.shape[1]):
        zz = e.measureModel_3Ax(np.copy(X[:, i]))#<- здесь[x]
        Z[:, i] = zz.flatten()
    Zn = Z + np.sqrt(R) @ np.random.normal(loc=0, scale=math.sqrt(1.0), size=(Z.shape[0], Z.shape[1]))
    return Zn

Zn = make_meas(Xn, R)

#######

def make_kalman_filter(measurement):
    Hp = np.array([[1, 0, 0, 0, 0, 0],
                   [0, 0, 1, 0, 0, 0],
                   [0, 0, 0, 0, 1, 0]])
    Hv = np.array([[0, 1, 0, 0, 0, 0],
                   [0, 0, 0, 1, 0, 0],
                   [0, 0, 0, 0, 0, 1]])
    x0 = Hp.T@measurement
    P0  = Hp.T@R@Hp + Hv.T@Rvel@Hv;
    kf = e.BindKFE(x0, P0, e.stateModel_3A(T), Q0, G, Hp, R)
    return kf

def step(Z, make_estimator):
    estimator = make_estimator(Z[:, 0][:, np.newaxis])
    est = np.zeros((initialState.shape[0], Z.shape[1]-1))
    for col in range(est.shape[1]):
        z = Z[:, col + 1]
        xp = estimator.predict(e.stateModel_3A(T),G,e.measureModel_3A())
        m1 = np.array([z[0], z[1], z[2]])
        xc = estimator.correct(e.measureModel_3A(),m1.T,R)
        est[:, col] = np.squeeze(xc[:])
    return est

est=step(Zn, make_kalman_filter)

fig = plt.figure(figsize=(9,22))
ax1 = fig.add_subplot(4,1,1)
ax2 = fig.add_subplot(4,1,2)
ax3 = fig.add_subplot(4,1,3)
ax4 = fig.add_subplot(4,1,4)


ax1.plot(X[0, :], X[2, :], label='Truth')
ax1.plot(Xn[0, :], Xn[2, :], label='Truth+noise')
ax1.plot(Zn[0, :], Zn[2, :], label='Truth+noise')
#ax1.plot(cart[0, :], cart[1, :], label='Measurement', linestyle='', marker='+')
ax1.plot(est[0, :], est[2, :], label='Estimates')
ax1.set_title("[x,vx,y,vy,z,vz]")
ax1.set_xlabel('x,met.')
ax1.set_ylabel('y,met.')
ax1.grid(True)

plt.show()

##########################################

def calc_err(X, make_estimator):
    Xn = add_process_noise(X,Q)
    Zn = make_meas(Xn,R)
    est = step(Zn, make_estimator)
    err = est - Xn[:, 1:]
    return err

from tqdm import tqdm

def calc_std_err(X, make_estimator):
    num_iterations = 2000
    var_err = np.zeros((X.shape[0], X.shape[1]-1))

    for i in tqdm(range(num_iterations)):
        err = calc_err(X, make_estimator)
        var_err += err ** 2

    var_err /= num_iterations
    return np.sqrt(var_err)

std_err = calc_std_err(X, make_kalman_filter)

plt.figure(figsize=(9,45))

# [x,vx,y,vy,z,vz]

plt.subplot(6, 1, 1)
plt.plot((np.arange(len(std_err[0, :]))+1)*T, std_err[0, :].T)
plt.grid(True)
plt.title("[x,vx,y,vy,z,vz]")
plt.xlabel('Time,s')
plt.ylabel('std_x, met')

plt.subplot(6, 1, 2)
plt.plot((np.arange(len(std_err[1, :]))+1)*T, std_err[1, :].T)
plt.grid(True)
plt.xlabel('Time,s')
plt.ylabel('std_vx, m/s')

plt.subplot(6, 1, 3)
plt.plot((np.arange(len(std_err[2, :]))+1)*T, std_err[2, :].T)
plt.grid(True)
plt.xlabel('Time,s')
plt.ylabel('std_y, met')

plt.subplot(6, 1, 4)
plt.plot((np.arange(len(std_err[3, :]))+1)*T, std_err[3, :].T)
plt.grid(True)
plt.xlabel('Time,s')
plt.ylabel('std_vy, m/s')

plt.subplot(6, 1, 5)
plt.plot((np.arange(len(std_err[4, :]))+1)*T, std_err[4, :].T)
plt.grid(True)
plt.xlabel('Time,s')
plt.ylabel('std_z, met')

plt.subplot(6, 1, 6)
plt.plot((np.arange(len(std_err[5, :]))+1)*T, std_err[5, :].T)
plt.grid(True)
plt.xlabel('Time,s')
plt.ylabel('std_vz, m/s')

plt.subplots_adjust(wspace=8.0, hspace=0.7)
plt.show()
