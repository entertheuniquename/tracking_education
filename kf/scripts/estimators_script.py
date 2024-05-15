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

def make_meas(X, R):
    Z = np.zeros((R.shape[0], X.shape[1]))
    for i in range(Z.shape[1]):
        zz = e.measureModel_XXx(np.copy(X[:, i]))#<- здесь[x]
        Z[:, i] = zz.flatten()
    Zn = Z + np.sqrt(R) @ np.random.normal(loc=0, scale=math.sqrt(1.0), size=(Z.shape[0], Z.shape[1]))
    return Zn

Zn = make_meas(Xn, R)

def make_kalman_filter(measurement):
    Hp = np.array([[1, 0, 0, 0, 0, 0],
                   [0, 0, 1, 0, 0, 0],
                   [0, 0, 0, 0, 1, 0]])
    Hv = np.array([[0, 1, 0, 0, 0, 0],
                   [0, 0, 0, 1, 0, 0],
                   [0, 0, 0, 0, 0, 1]])
    x0 = Hp.T@measurement
    P0  = Hp.T@R@Hp + Hv.T@Rvel@Hv;
    kf = e.BindKFE(x0, P0, e.stateModel_CV(T), Q0, G, Hp, R)
    return kf


def step(Z, make_estimator):
    estimator = make_estimator(Z[:, 0][:, np.newaxis])
    est = np.zeros((initialState.shape[0], Z.shape[1]-1))
    for col in range(est.shape[1]):
        z = Z[:, col + 1]
        xp = estimator.predict(T)
        m1 = np.array([z[0], z[1], z[2]])
        xc = estimator.correct(m1.T)
        est[:, col] = np.squeeze(xc[:])
    return est

est=step(Zn, make_kalman_filter)

######################################################################################################

def make_meas_pol(X, R):
    Z = np.zeros((R.shape[0], X.shape[1]))
    for i in range(Z.shape[1]):
        xx = np.copy(X[:, i])
        zz = e.measureModel_XRx(np.copy(X[:, i]))
        Z[:, i] = zz.flatten()
    Zn = Z + np.sqrt(R) @ np.random.normal(loc=0, scale=math.sqrt(1.0), size=(Z.shape[0], Z.shape[1]))
    return Zn

Zn_pol = make_meas_pol(Xn, Rpol)#+

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

cart_pol = sph2cart(Zn_pol)

def sph2cartcov(sphCov, az, el, r):
    pr = 2
    pa = 0
    pe = 1
    pvr= 3

    rngSig = np.sqrt(sphCov[pr, pr])
    azSig  = np.sqrt(sphCov[pa, pa])
    elSig  = np.sqrt(sphCov[pe, pe])


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
    r  = meas[2,0]
    az = np.rad2deg(meas[1,0])
    el = np.rad2deg(meas[0,0])
    sphCov = np.diag([covMeas[0,0], covMeas[2,2], covMeas[1,1]])
    [posCov, velCov] = sph2cartcov(sphCov, az, el, r)
    Hp = np.array([[1, 0, 0, 0, 0, 0],
                   [0, 0, 1, 0, 0, 0],
                   [0, 0, 0, 0, 1, 0]])
    Hv = np.array([[0, 1, 0, 0, 0, 0],
                   [0, 0, 0, 1, 0, 0],
                   [0, 0, 0, 0, 0, 1]])
    return Hp.T@posCov@Hp + Hv.T@velCov@Hv

def make_kalman_filter_pol(measurement):
    Hp = np.array([[1, 0, 0, 0, 0, 0],
                   [0, 0, 1, 0, 0, 0],
                   [0, 0, 0, 0, 1, 0]])
    x0 = sph2cart(measurement)
    x0 = Hp.T@x0
    P0  = make_cartcov(measurement, Rpol)
    ekf = e.BindEKFE(x0, P0, Q, Rpol)
    return ekf

def step_pol(Z, make_estimator):
    estimator = make_estimator(Z[:, 0][:, np.newaxis])
    est = np.zeros((initialState.shape[0], Z.shape[1]-1))
    for col in range(est.shape[1]):
        z = Z[:, col + 1]
        xp = estimator.predict(T)
        m1 = np.array([z[0], z[1], z[2]])
        xc = estimator.correct(m1.T)
        est[:, col] = np.squeeze(xc[:])
    return est

est_pol=step_pol(Zn_pol, make_kalman_filter_pol)

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
ax2.plot(cart_pol[0, :], cart_pol[1, :], label='Measurement', linestyle='', marker='+')
ax2.plot(est_pol[0, :], est_pol[2, :], label='Estimates')
#ax2.set_title("[x,vx,y,vy,z,vz]")
ax2.set_xlabel('x,met.')
ax2.set_ylabel('y,met.')
ax2.grid(True)

#ax3.plot(Zn_pol[0, :], label='Truth')
#ax3.plot(Zn_pol[1, :], label='Truth')
#ax3.plot(Zn_pol[2, :], label='Truth')
##ax3.set_title("[x,vx,y,vy,z,vz]")
#ax3.set_ylabel('r,a,e,met.')
#ax3.set_xlabel('iterations,met.')
#ax3.grid(True)

#ax4.plot(cart_pol[0, :], label='Truth')
#ax4.plot(cart_pol[1, :], label='Truth')
#ax4.plot(cart_pol[2, :], label='Truth')
##ax4.set_title("[x,vx,y,vy,z,vz]")
#ax4.set_ylabel('x,y,z,met.')
#ax4.set_xlabel('iterations,met.')
#ax4.grid(True)

plt.show()

######################################################################################################

def calc_err(X, make_estimator):
    Xn = add_process_noise(X,Q)
    Zn = make_meas(Xn,Rpol)
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

######################################################################################################

def calc_err_pol(X, make_estimator):
    Xn = add_process_noise(X,Q)
    Zn = make_meas_pol(Xn,Rpol)
    est = step_pol(Zn, make_estimator)
    err = est - Xn[:, 1:]
    return err

def calc_std_err_pol(X, make_estimator):
    num_iterations = 2000
    var_err = np.zeros((X.shape[0], X.shape[1]-1))

    for i in tqdm(range(num_iterations)):
        err = calc_err_pol(X, make_estimator)
        var_err += err ** 2

    var_err /= num_iterations
    return np.sqrt(var_err)

std_err_pol = calc_std_err_pol(X, make_kalman_filter_pol)

######################################################################################################

plt.figure(figsize=(9,45))

# [x,vx,y,vy,z,vz]

plt.subplot(6, 1, 1)
plt.plot((np.arange(len(std_err[0, :]))+1)*T, std_err[0, :].T)
plt.plot((np.arange(len(std_err_pol[0, :]))+1)*T, std_err_pol[0, :].T)
plt.grid(True)
plt.title("[x,vx,y,vy,z,vz]")
plt.xlabel('Time,s')
plt.ylabel('std_x, met')

plt.subplot(6, 1, 2)
plt.plot((np.arange(len(std_err[1, :]))+1)*T, std_err[1, :].T)
plt.plot((np.arange(len(std_err_pol[1, :]))+1)*T, std_err_pol[1, :].T)
plt.grid(True)
plt.xlabel('Time,s')
plt.ylabel('std_vx, m/s')

plt.subplot(6, 1, 3)
plt.plot((np.arange(len(std_err[2, :]))+1)*T, std_err[2, :].T)
plt.plot((np.arange(len(std_err_pol[2, :]))+1)*T, std_err_pol[2, :].T)
plt.grid(True)
plt.xlabel('Time,s')
plt.ylabel('std_y, met')

plt.subplot(6, 1, 4)
plt.plot((np.arange(len(std_err[3, :]))+1)*T, std_err[3, :].T)
plt.plot((np.arange(len(std_err_pol[3, :]))+1)*T, std_err_pol[3, :].T)
plt.grid(True)
plt.xlabel('Time,s')
plt.ylabel('std_vy, m/s')

plt.subplot(6, 1, 5)
plt.plot((np.arange(len(std_err[4, :]))+1)*T, std_err[4, :].T)
plt.plot((np.arange(len(std_err_pol[4, :]))+1)*T, std_err_pol[4, :].T)
plt.grid(True)
plt.xlabel('Time,s')
plt.ylabel('std_z, met')

plt.subplot(6, 1, 6)
plt.plot((np.arange(len(std_err[5, :]))+1)*T, std_err[5, :].T)
plt.plot((np.arange(len(std_err_pol[5, :]))+1)*T, std_err_pol[5, :].T)
plt.grid(True)
plt.xlabel('Time,s')
plt.ylabel('std_vz, m/s')

plt.subplots_adjust(wspace=8.0, hspace=0.7)
plt.show()
