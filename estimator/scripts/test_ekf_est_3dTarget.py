import numpy as np
import numpy.matlib as ml
import matplotlib.pyplot as plt
import math
from IPython.display import display, Math, Latex
from IPython.display import Markdown as md
from arr2ltx import convert2latex, to_latex

import estimator as e
import math

T = 0.2
process_var = 0.5

R = np.diag([1e-4, 1.0])
R3 = np.diag([1e-4, 1e-4, 1.0])

Qp = np.diag([process_var, process_var])
Qp3 = np.diag([process_var, process_var, process_var])

G = np.array([[T**2/2, 0,    ],
              [T,      0,    ],
              [0,      T**2/2],
              [0,      T,    ]])
G_3A = np.array([[T**2/2, 0,      0     ],
                 [T,      0,      0     ],
                 [0,      T**2/2, 0     ],
                 [0,      T,      0     ],
                 [0,      0,      T**2/2], 
                 [0,      0,      T     ]])
G_3B = np.array([[T**2/2, 0,      0     ],
                 [0,      T**2/2, 0     ],
                 [0,      0,      T**2/2],
                 [T,      0,      0     ],
                 [0,      T,      0     ], 
                 [0,      0,      T     ]])

Q = G@Qp@G.T
Q_3A = G_3A@Qp3@G_3A.T
Q_3B = G_3B@Qp3@G_3B.T

initialState = np.array([-40., -2., 0., 0.])
initialState3A = np.array([-40., -2., 0., 0., 0., 0.])
initialState3B = np.array([-40., 0., 0., -2., 0., 0.])

initialState = initialState[:, np.newaxis]
initialState3A = initialState3A[:, np.newaxis]
initialState3B = initialState3B[:, np.newaxis]

max_speed = 4

def make_true_data(x0):
    X = np.zeros((x0.shape[0], 100))
    X[:, 0] = x0.T
    for i in range(X.shape[1]-1):
        xx = e.stateModel(np.copy(X[:, i]), T)
        X[:, i+1] = xx.flatten()
    return X

def make_true_data3A(x0):
    X = np.zeros((x0.shape[0], 100))
    X[:, 0] = x0.T
    for i in range(X.shape[1]-1):
        xx = e.stateModel3A(np.copy(X[:, i]), T)
        X[:, i+1] = xx.flatten()
    return X

def make_true_data3B(x0):
    X = np.zeros((x0.shape[0], 100))
    X[:, 0] = x0.T
    for i in range(X.shape[1]-1):
        xx = e.stateModel3B(np.copy(X[:, i]), T)
        X[:, i+1] = xx.flatten()
    return X

X=make_true_data(initialState)
X_3A=make_true_data3A(initialState3A)
X_3B=make_true_data3B(initialState3B)

def add_process_noise(X, Var):
    Xn = X + np.sqrt(Var) @ np.random.normal(loc=0, scale=1.0, size=(X.shape[0], X.shape[1]))
    return Xn

Xn = add_process_noise(X, Q)
Xn_3A = add_process_noise(X_3A, Q_3A)
Xn_3B = add_process_noise(X_3B, Q_3B)

def make_meas(X, R):
    Z = np.zeros((R.shape[0], X.shape[1]))
    for i in range(Z.shape[1]):
        zz = e.measureModel(np.copy(X[:, i]))
        Z[:, i] = zz.flatten()
    Zn = Z + np.sqrt(R) @ np.random.normal(loc=0, scale=math.sqrt(1.0), size=(Z.shape[0], Z.shape[1]))
    return Zn

def make_meas3A(X, R):
    Z = np.zeros((R.shape[0], X.shape[1]))
    for i in range(Z.shape[1]):
        zz = e.measureModel3A(np.copy(X[:, i]))
        Z[:, i] = zz.flatten()
    Zn = Z + np.sqrt(R) @ np.random.normal(loc=0, scale=math.sqrt(1.0), size=(Z.shape[0], Z.shape[1]))
    return Zn

def make_meas3B(X, R):
    Z = np.zeros((R.shape[0], X.shape[1]))
    for i in range(Z.shape[1]):
        zz = e.measureModel3B(np.copy(X[:, i]))
        Z[:, i] = zz.flatten()
    Zn = Z + np.sqrt(R) @ np.random.normal(loc=0, scale=math.sqrt(1.0), size=(Z.shape[0], Z.shape[1]))
    return Zn

Zn = make_meas(Xn, R)
Zn_3A = make_meas3A(Xn_3A, R3)
Zn_3B = make_meas3B(Xn_3B, R3)

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
        cart[0, i] = measurement[1, i] * math.cos(measurement[0, i])
        cart[1, i] = measurement[1, i] * math.sin(measurement[0, i])
    return cart

def sph2cart3(measurement):
    cart = np.zeros(measurement.shape)
    for i in range(measurement.shape[1]):
        cart[0, i] = measurement[2, i] * math.cos(measurement[1, i]) * math.cos(measurement[0, i])
        cart[1, i] = measurement[2, i] * math.sin(measurement[1, i]) * math.cos(measurement[0, i])
        cart[2, i] = measurement[2, i] * math.sin(measurement[0, i])
    return cart

cart = sph2cart(Zn)
cart_3A = sph2cart3(Zn_3A)
cart_3B = sph2cart3(Zn_3B)

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
    az = np.rad2deg(meas[0,0])
    el = 0
    r  = meas[1,0]
    sphCov = np.diag([covMeas[0,0], 0, covMeas[1,1]])#?
    [posCov, velCov] = sph2cartcov(sphCov, az, el, r)
    posCov = posCov[0:2,0:2]
    velCov = velCov[0:2,0:2]
    Hp = np.array([[1, 0, 0, 0],
                   [0, 0, 1, 0]])
    Hv = np.array([[0, 1, 0, 0],
                   [0, 0, 0, 1]])
    return Hp.T@posCov@Hp + Hv.T@velCov@Hv

def make_cartcov3A(meas, covMeas):
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

def make_cartcov3B(meas, covMeas):
    az = np.rad2deg(meas[1,0])
    el = np.rad2deg(meas[0,0])
    r  = meas[2,0]
    sphCov = np.diag([covMeas[0,0], covMeas[2,2], covMeas[1,1]])
    [posCov, velCov] = sph2cartcov(sphCov, az, el, r)
    Hp = np.array([[1, 0, 0, 0, 0, 0],
                   [0, 1, 0, 0, 0, 0],
                   [0, 0, 1, 0, 0, 0]])
    Hv = np.array([[0, 0, 0, 1, 0, 0],
                   [0, 0, 0, 0, 1, 0],
                   [0, 0, 0, 0, 0, 1]])
    return Hp.T@posCov@Hp + Hv.T@velCov@Hv

def make_kalman_filter(measurement):
    Hp = np.array([[1, 0, 0, 0],
                   [0, 0, 1, 0]])
    x0 = sph2cart(measurement)
    x0 = Hp.T@x0
    P0  = make_cartcov(measurement, R)
    kf = e.Ekf(x0, P0, Q, R)
    return kf

def make_kalman_filter3A(measurement):
    Hp = np.array([[1, 0, 0, 0, 0, 0],
                   [0, 0, 1, 0, 0, 0],
                   [0, 0, 0, 0, 1, 0]])
    x0 = sph2cart3(measurement)
    x0 = Hp.T@x0
    P0  = make_cartcov3A(measurement, R3)
    kf = e.Ekf(x0, P0, Q_3A, R3)
    return kf

def make_kalman_filter3B(measurement):
    Hp = np.array([[1, 0, 0, 0, 0, 0],
                   [0, 1, 0, 0, 0, 0],
                   [0, 0, 1, 0, 0, 0]])
    x0 = sph2cart3(measurement)
    x0 = Hp.T@x0
    P0  = make_cartcov3B(measurement, R3)
    kf = e.Ekf(x0, P0, Q_3B, R3)
    return kf

def step(Z, make_estimator):
    estimator = make_estimator(Z[:, 0][:, np.newaxis])
    est = np.zeros((initialState.shape[0], Z.shape[1]-1))
    for col in range(est.shape[1]):
        z = Z[:, col + 1]
        xp, _ = estimator.predictStateModel(T)
        m1 = np.array([z[0], z[1]])
        xc, _ = estimator.correctMeasureModel(m1.T)
        est[:, col] = np.squeeze(xc[:])

    return est

def step3A(Z, make_estimator):
    estimator = make_estimator(Z[:, 0][:, np.newaxis])
    est = np.zeros((initialState3A.shape[0], Z.shape[1]-1))
    for col in range(est.shape[1]):
        z = Z[:, col + 1]
        xp, _ = estimator.predictStateModel3A(T)
        m1 = np.array([z[0], z[1], z[2]])
        xc, _ = estimator.correctMeasureModel3A(m1.T)
        est[:, col] = np.squeeze(xc[:])
    return est

def step3AA(Z, make_estimator):
    estimator = make_estimator(Z[:, 0][:, np.newaxis])
    est = np.zeros((initialState3A.shape[0], Z.shape[1]-1))
    for col in range(est.shape[1]):
        z = Z[:, col + 1]
        xp, _ = estimator.predictStateModel3AA(T)
        m1 = np.array([z[0], z[1], z[2]])
        xc, _ = estimator.correctMeasureModel3AA(m1.T)
        est[:, col] = np.squeeze(xc[:])
    return est

def step3B(Z, make_estimator):
    estimator = make_estimator(Z[:, 0][:, np.newaxis])
    est = np.zeros((initialState3B.shape[0], Z.shape[1]-1))
    for col in range(est.shape[1]):
        z = Z[:, col + 1]
        xp, _ = estimator.predictStateModel3B(T)
        m1 = np.array([z[0], z[1], z[2]])
        xc, _ = estimator.correctMeasureModel3B(m1.T)
        est[:, col] = np.squeeze(xc[:])
    return est

est=step(Zn, make_kalman_filter)
est_3A=step3A(Zn_3A, make_kalman_filter3A)
est_3AA=step3AA(Zn_3A, make_kalman_filter3A)
est_3B=step3B(Zn_3B, make_kalman_filter3B)

fig = plt.figure(figsize=(9,22))
ax1 = fig.add_subplot(4,1,1)
ax2 = fig.add_subplot(4,1,2)
ax3 = fig.add_subplot(4,1,3)
ax4 = fig.add_subplot(4,1,4)

ax1.plot(X[0, :], X[2, :], label='Truth')
ax1.plot(Xn[0, :], Xn[2, :], label='Truth+noise')
ax1.plot(cart[0, :], cart[1, :], label='Measurement', linestyle='', marker='+')
ax1.plot(est[0, :], est[2, :], label='Estimates')
ax1.set_title("[x,vx,y,vy]")
ax1.set_xlabel('x,met.')
ax1.set_ylabel('y,met.')
ax1.grid(True)

ax2.plot(X_3A[0, :], X_3A[2, :], label='Truth')
ax2.plot(Xn_3A[0, :], Xn_3A[2, :], label='Truth+noise')
ax2.plot(cart_3A[0, :], cart_3A[1, :], label='Measurement', linestyle='', marker='+')
ax2.plot(est_3A[0, :], est_3A[2, :], label='Estimates')
ax2.set_title("[x,vx,y,vy,z,vz]")
ax2.set_xlabel('x,met.')
ax2.set_ylabel('y,met.')
ax2.grid(True)

ax3.plot(X_3A[0, :], X_3A[2, :], label='Truth')
ax3.plot(Xn_3A[0, :], Xn_3A[2, :], label='Truth+noise')
ax3.plot(cart_3A[0, :], cart_3A[1, :], label='Measurement', linestyle='', marker='+')
ax3.plot(est_3AA[0, :], est_3AA[2, :], label='Estimates-J')
ax3.set_title("[x,vx,y,vy,z,vz] - analitic jacobian")
ax3.set_xlabel('x,met.')
ax3.set_ylabel('y,met.')
ax3.grid(True)

ax4.plot(X_3B[0, :], X_3B[2, :], label='Truth')
ax4.plot(Xn_3B[0, :], Xn_3B[2, :], label='Truth+noise')
ax4.plot(cart_3B[0, :], cart_3B[1, :], label='Measurement', linestyle='', marker='+')
ax4.plot(est_3B[0, :], est_3B[2, :], label='Estimates')
ax4.set_title("[x,y,z,vx,vy,vz]")
ax4.set_xlabel('x,met.')
ax4.set_ylabel('y,met.')
ax4.grid(True)

plt.show()

def calc_err(X, make_estimator):
    Xn = add_process_noise(X, Q)
    Zn = make_meas(Xn, R)
    est = step(Zn, make_estimator)
    err = est - Xn[:, 1:]
    return err

def calc_err3A(X, make_estimator):
    Xn = add_process_noise(X, Q_3A)
    Zn = make_meas3A(Xn, R3)
    est = step3A(Zn, make_estimator)
    err = est - Xn[:, 1:]
    return err

def calc_err3AA(X, make_estimator):
    Xn = add_process_noise(X, Q_3A)
    Zn = make_meas3A(Xn, R3)
    est = step3AA(Zn, make_estimator)
    err = est - Xn[:, 1:]
    return err

def calc_err3B(X, make_estimator):
    Xn = add_process_noise(X, Q_3B)
    Zn = make_meas3B(Xn, R3)
    est = step3B(Zn, make_estimator)
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

def calc_std_err3A(X, make_estimator):
    num_iterations = 2000
    var_err = np.zeros((X.shape[0], X.shape[1]-1))

    for i in tqdm(range(num_iterations)):
        err = calc_err3A(X, make_estimator)
        var_err += err ** 2

    var_err /= num_iterations
    return np.sqrt(var_err)

def calc_std_err3AA(X, make_estimator):
    num_iterations = 2000
    var_err = np.zeros((X.shape[0], X.shape[1]-1))

    for i in tqdm(range(num_iterations)):
        err = calc_err3AA(X, make_estimator)
        var_err += err ** 2

    var_err /= num_iterations
    return np.sqrt(var_err)

def calc_std_err3B(X, make_estimator):
    num_iterations = 2000
    var_err = np.zeros((X.shape[0], X.shape[1]-1))

    for i in tqdm(range(num_iterations)):
        err = calc_err3B(X, make_estimator)
        var_err += err ** 2

    var_err /= num_iterations
    return np.sqrt(var_err)

std_err = calc_std_err(X, make_kalman_filter)
std_err_3A = calc_std_err3A(X_3A, make_kalman_filter3A)
std_err_3AA = calc_std_err3AA(X_3A, make_kalman_filter3A)
std_err_3B = calc_std_err3B(X_3B, make_kalman_filter3B)


plt.figure(figsize=(9,45))

# [x,vx,y,vy]

plt.subplot(22, 1, 1)
plt.plot((np.arange(len(std_err[0, :]))+1)*T, std_err[0, :].T)
plt.grid(True)
plt.title("[x,vx,y,vy]")
plt.xlabel('Time,s')
plt.ylabel('std_x, met')

plt.subplot(22, 1, 2)
plt.plot((np.arange(len(std_err[1, :]))+1)*T, std_err[1, :].T)
plt.grid(True)
plt.xlabel('Time,s')
plt.ylabel('std_vx, m/s')

plt.subplot(22, 1, 3)
plt.plot((np.arange(len(std_err[2, :]))+1)*T, std_err[2, :].T)
plt.grid(True)
plt.xlabel('Time,s')
plt.ylabel('std_y, met')

plt.subplot(22, 1, 4)
plt.plot((np.arange(len(std_err[3, :]))+1)*T, std_err[3, :].T)
plt.grid(True)
plt.xlabel('Time,s')
plt.ylabel('std_vy, m/s')

# [x,vx,y,vy,z,vz]

plt.subplot(22, 1, 5)
plt.plot((np.arange(len(std_err_3A[0, :]))+1)*T, std_err_3A[0, :].T)
plt.grid(True)
plt.title("[x,vx,y,vy,z,vz]")
plt.xlabel('Time,s')
plt.ylabel('std_x, met')

plt.subplot(22, 1, 6)
plt.plot((np.arange(len(std_err_3A[1, :]))+1)*T, std_err_3A[1, :].T)
plt.grid(True)
plt.xlabel('Time,s')
plt.ylabel('std_vx, m/s')

plt.subplot(22, 1, 7)
plt.plot((np.arange(len(std_err_3A[2, :]))+1)*T, std_err_3A[2, :].T)
plt.grid(True)
plt.xlabel('Time,s')
plt.ylabel('std_y, met')

plt.subplot(22, 1, 8)
plt.plot((np.arange(len(std_err_3A[3, :]))+1)*T, std_err_3A[3, :].T)
plt.grid(True)
plt.xlabel('Time,s')
plt.ylabel('std_vy, m/s')

plt.subplot(22, 1, 9)
plt.plot((np.arange(len(std_err_3A[4, :]))+1)*T, std_err_3A[4, :].T)
plt.grid(True)
plt.xlabel('Time,s')
plt.ylabel('std_z, met')

plt.subplot(22, 1, 10)
plt.plot((np.arange(len(std_err_3A[5, :]))+1)*T, std_err_3A[5, :].T)
plt.grid(True)
plt.xlabel('Time,s')
plt.ylabel('std_vz, m/s')

# [x,vx,y,vy,z,vz] - analitic jacobian

plt.subplot(22, 1, 11)
plt.plot((np.arange(len(std_err_3AA[0, :]))+1)*T, std_err_3AA[0, :].T)
plt.grid(True)
plt.title("[x,vx,y,vy,z,vz] - analitic jacobian")
plt.xlabel('Time,s')
plt.ylabel('std_x, met')

plt.subplot(22, 1, 12)
plt.plot((np.arange(len(std_err_3AA[1, :]))+1)*T, std_err_3AA[1, :].T)
plt.grid(True)
plt.xlabel('Time,s')
plt.ylabel('std_vx, m/s')

plt.subplot(22, 1, 13)
plt.plot((np.arange(len(std_err_3AA[2, :]))+1)*T, std_err_3AA[2, :].T)
plt.grid(True)
plt.xlabel('Time,s')
plt.ylabel('std_y, met')

plt.subplot(22, 1, 14)
plt.plot((np.arange(len(std_err_3AA[3, :]))+1)*T, std_err_3AA[3, :].T)
plt.grid(True)
plt.xlabel('Time,s')
plt.ylabel('std_vy, m/s')

plt.subplot(22, 1, 15)
plt.plot((np.arange(len(std_err_3AA[4, :]))+1)*T, std_err_3AA[4, :].T)
plt.grid(True)
plt.xlabel('Time,s')
plt.ylabel('std_z, met')

plt.subplot(22, 1, 16)
plt.plot((np.arange(len(std_err_3AA[5, :]))+1)*T, std_err_3AA[5, :].T)
plt.grid(True)
plt.xlabel('Time,s')
plt.ylabel('std_vz, m/s')

# [x,y,z,vx,vy,vz]

plt.subplot(22, 1, 17)
plt.plot((np.arange(len(std_err_3B[0, :]))+1)*T, std_err_3B[0, :].T)
plt.grid(True)
plt.title("[x,y,z,vx,vy,vz]")
plt.xlabel('Time,s')
plt.ylabel('std_x, met')

plt.subplot(22, 1, 18)
plt.plot((np.arange(len(std_err_3B[1, :]))+1)*T, std_err_3B[1, :].T)
plt.grid(True)
plt.xlabel('Time,s')
plt.ylabel('std_y, met')

plt.subplot(22, 1, 19)
plt.plot((np.arange(len(std_err_3B[2, :]))+1)*T, std_err_3B[2, :].T)
plt.grid(True)
plt.xlabel('Time,s')
plt.ylabel('std_z, met')

plt.subplot(22, 1, 20)
plt.plot((np.arange(len(std_err_3B[3, :]))+1)*T, std_err_3B[3, :].T)
plt.grid(True)
plt.xlabel('Time,s')
plt.ylabel('std_vx, m/s')

plt.subplot(22, 1, 21)
plt.plot((np.arange(len(std_err_3B[4, :]))+1)*T, std_err_3B[4, :].T)
plt.grid(True)
plt.xlabel('Time,s')
plt.ylabel('std_vy, m/s')

plt.subplot(22, 1, 22)
plt.plot((np.arange(len(std_err_3B[5, :]))+1)*T, std_err_3B[5, :].T)
plt.grid(True)
plt.xlabel('Time,s')
plt.ylabel('std_vz, m/s')

plt.subplots_adjust(wspace=8.0, hspace=0.7)
plt.show()
