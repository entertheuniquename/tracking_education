#!/usr/bin/python3

import math
import numpy as np
import matplotlib.pyplot as plt
import measgen as gen
import estimator as e

print("imports - DONE!")

T = 6
process_var = 0.01
process_var_w = 0.000001
meas_std = 300.
velo_std = 30.
w_std = 0.098

Q0 = np.diag([process_var, process_var, process_var, process_var_w])
G = np.array([[T**2/2, 0,      0     , 0],
              [T,      0,      0     , 0],
              [0,      T**2/2, 0     , 0],
              [0,      T,      0     , 0],
              [0,      0,      T**2/2, 0],
              [0,      0,      T     , 0],
              [0,      0,      0     , T]])
Q = G@Q0@G.T

# Матрица ошибок измерения
R = np.diag([meas_std*meas_std, meas_std*meas_std, meas_std*meas_std])
Rvel = np.diag([velo_std*velo_std, velo_std*velo_std, velo_std*velo_std])

# Векторы входных данных
x0 = np.array([30000., -200., 0., 0., 0., 0., 0])#radian

Hp = np.array([[1., 0., 0., 0., 0., 0., 0.],
               [0., 0., 1., 0., 0., 0., 0.],
               [0., 0., 0., 0., 1., 0., 0.]])
Hv = np.array([[0., 1., 0., 0., 0., 0., 0.],
               [0., 0., 0., 1., 0., 0., 0.],
               [0., 0., 0., 0., 0., 1., 0.]])

P0  = Hp.T@R@Hp + Hv.T@Rvel@Hv;
P0[6,6] = w_std*w_std

print("make data - DONE!")

[X, Xn, Zn] = gen.generate([np.array([30000., -200.,   0.,   0.,   0.,   0.,    0.]),
                            np.array([  None,  None, None, None, None, None, 0.098]),
                            np.array([  None,  None, None, None, None, None,    0.]),
                            np.array([  None,  300., None, None, None, None,  None])],
                           [50,8,50,10],
                           Q, R, T);

print("Zn["+str(Zn.shape[0])+","+str(Zn.shape[1])+"]")

print("measurement - DONE!")

kf = e.BindKF7_CV(x0,P0,Q0,R)
print("kf - DONE!")

ekf = e.BindEKF7_CT(x0,P0,Q0,R)
print("ekf - DONE!")

ekf0 = e.BindEKFE_xyz_ct(x0,P0,Q,R)
print("ekf0 - DONE!")

ekf2 = e.BindEKF27_CT(x0,P0,Q0,R)
print("ekf2 - DONE!")

mu=[0.99,0.01]
tp=[[0.95,0.05],
    [0.05,0.95]]

imm = e.BindIMM7(x0,P0,Q0,R,mu,tp)

print("imm - DONE!")

est_kf = np.zeros((x0.shape[0], Zn.shape[1]-1))
est_ekf = np.zeros((x0.shape[0], Zn.shape[1]-1))
est_ekf0 = np.zeros((x0.shape[0], Zn.shape[1]-1))
est_ekf2 = np.zeros((x0.shape[0], Zn.shape[1]-1))
est_imm = np.zeros((x0.shape[0], Zn.shape[1]-1))
mus = np.zeros((2, Zn.shape[1]-1))
for col in range(est_imm.shape[1]):
    z = Zn[:, col + 1]

    xp_kf = kf.predict(T)
    xp_ekf = ekf.predict(T)
    xp_ekf0 = ekf0.predict(T)
    xp_ekf2 = ekf2.predict(T)
    xp_imm = imm.predict(T)

    m1 = np.array([z[0], z[1], z[2]])

    xc_kf = kf.correct(m1.T)
    xc_ekf = ekf.correct(m1.T)
    xc_ekf0 = ekf0.correct(m1.T)
    xc_ekf2 = ekf2.correct(m1.T)
    xc_imm = imm.correct(m1.T)

    est_kf[:, col] = np.squeeze(xc_kf[:])
    est_ekf[:, col] = np.squeeze(xc_ekf[:])
    est_ekf0[:, col] = np.squeeze(xc_ekf0[:])
    est_ekf2[:, col] = np.squeeze(xc_ekf2[:])
    est_imm[:, col] = np.squeeze(xc_imm[:])
    mus[:, col] = np.squeeze(imm.mu())

print("estimations - DONE!")

fig = plt.figure(figsize=(9,25))

ax1 = fig.add_subplot(2,1,1)
ax1.plot(X[0, :], X[2, :], label='true', marker='', color='black')
ax1.plot(Zn[0, :], Zn[1, :], label='measurement', marker='x', color='grey')
ax1.plot(est_kf[0, :], est_kf[2, :], label='kf', marker='', color='red')
ax1.plot(est_ekf[0, :], est_ekf[2, :], label='ekf', marker='', color='orange')
ax1.plot(est_ekf0[0, :], est_ekf0[2, :], label='ekf_oldstyle', marker='', color='gold')
ax1.plot(est_ekf2[0, :], est_ekf2[2, :], label='ekf_base', marker='', color='tomato')
ax1.plot(est_imm[0, :], est_imm[2, :], label='imm', marker='', color='purple')
ax1.set_title("X(Y)")
plt.legend()
ax1.set_xlabel('x,met')
ax1.set_ylabel('y,met')
ax1.grid(True)

ax2 = fig.add_subplot(2,1,2)
ax2.plot(mus[0, :], label='kf', marker='', color='red')
ax2.plot(mus[1, :], label='ekf', marker='', color='orange')
ax2.set_title("MU")
plt.legend()
ax2.set_xlabel('iteration')
ax2.set_ylabel('mu')
ax2.grid(True)

plt.show()
