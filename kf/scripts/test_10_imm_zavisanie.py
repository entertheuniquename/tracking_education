#!/usr/bin/python3

import numpy as np
import numpy.matlib as ml
import matplotlib.pyplot as plt
import math
from IPython.display import display, Math, Latex
from IPython.display import Markdown as md

import estimator as e
import math
import stand_10_zavisanie as stand

T = 6.
process_var = 1.
process_var_w = 0.0002
meas_std = 300.
velo_std = 30.
acc_std = 3.
w_std = 0.392

Rp = np.diag([pow(meas_std,2), pow(meas_std,2), pow(meas_std,2)])
Rv = np.diag([pow(velo_std,2), pow(velo_std,2), pow(velo_std,2)])
Ra = np.diag([pow(acc_std,2), pow(acc_std,2), pow(acc_std,2)])
Rw = np.diag([pow(w_std,2)])

Q0 = np.diag([process_var, process_var, process_var, process_var_w])

G = e.BindG_10

Q = G(T)@Q0@G(T).T

x0_2g = np.array([30000., 100., 0., 0., 0., 0., 0., 0., 0., 0.])
x0_2g = x0_2g[:, np.newaxis]


Hp = np.array([[1., 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 1., 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 1., 0, 0, 0]])
Hv = np.array([[0, 1., 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 1., 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 1., 0, 0]])
Ha = np.array([[0, 0, 1., 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 1., 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 1., 0]])
Hw = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1.]])
P0  = Hp.T@Rp@Hp + Hv.T@Rv@Hv + Ha.T@Ra@Ha + Hw.T@Rw@Hw

u = np.zeros((10,1))
B = np.zeros((10,10))

amount = 100

X2G = stand.make_true_data(x0_2g,T)

Xn2G = stand.add_process_noise(X2G,Q)

Zn2G = stand.make_meas(Xn2G,Rp,e.BindHXX_10)

mu = np.array([0.33333,0.33333,0.33333])

tp = np.array([[0.95, 0.025, 0.025],
               [0.025, 0.95, 0.025],
               [0.025, 0.025, 0.95]])

mu4 = np.array([0.25,0.25,0.25,0.25])

tp4 = np.array([[0.91, 0.03, 0.03, 0.03],
                [0.03, 0.91, 0.03, 0.03],
                [0.03, 0.03, 0.91, 0.03],
                [0.03, 0.03, 0.03, 0.91]])

imm_est_2g, mus_2g = stand.estimate(e.BindIMM_10_KFCV_EKFCT_KFCA,x0_2g,P0,Q0,Rp,Zn2G,T,MU=mu,TP=tp,imm_filters_amount=3)

[fff0, imm_std_err_2g] = stand.test(e.BindIMM_10_KFCV_EKFCT_KFCA,
                                   x0_2g,P0,Q0,Rp,
                                   e.BindHXX_10,
                                   e.BindG_10,
                                   6,100,2000,0.098,MU=mu,TP=tp,imm_filters_amount=3)

fig = plt.figure("Тест для исследования работы фильтра IMM с состояниями типа [x,vx,ax,y,vy,ay,z,vz,az,w] и измерениями типа [x,y,z] условиях зависания и смены курса",figsize=(21,11))

ax1 = fig.add_subplot(4,1,1)
ax1.plot(X2G[0, :], X2G[3, :], label='true(2G)', marker='', color='black')
ax1.plot(Zn2G[0, :], Zn2G[1, :], label='measurement(2G)', marker='x', color='grey')
ax1.plot(imm_est_2g[0, :], imm_est_2g[3, :], label='imm[3](2G)', marker='', color='red')
ax1.set_title("Y(X)")
ax1.set_xlabel('x,met')
ax1.set_ylabel('y,met')
ax1.grid(True)
plt.legend()

ax0 = fig.add_subplot(4,1,2)
ax0.plot(X2G[0, :], X2G[6, :], label='true(2G)', marker='', color='black')
ax0.plot(Zn2G[0, :], Zn2G[2, :], label='measurement(2G)', marker='x', color='grey')
ax0.plot(imm_est_2g[0, :], imm_est_2g[6, :], label='imm[3](2G)', marker='', color='red')
ax0.set_title("Z(X)")
ax0.set_xlabel('x,met')
ax0.set_ylabel('z,met')
ax0.grid(True)
plt.legend()

ax2 = fig.add_subplot(4,4,9)
ax2.plot((np.arange(len(imm_std_err_2g[0, :]))+1)*T, imm_std_err_2g[0, :].T, label='imm[3](2G)', marker='', color='red')
ax2.set_title("std_err_x(iteration)")
ax2.set_xlabel('Time,s')
ax2.set_ylabel('std_err_x, met')
ax2.grid(True)
plt.legend()

ax3 = fig.add_subplot(4,4,13)
ax3.plot((np.arange(len(imm_std_err_2g[1, :]))+1)*T, imm_std_err_2g[1, :].T, label='imm[3](2G)', marker='', color='red')
ax3.set_title("err_vx(iteration)")
ax3.set_xlabel('Time,s')
ax3.set_ylabel('std_err_vx, met')
ax3.grid(True)
plt.legend()

ax4 = fig.add_subplot(4,4,10)
ax4.plot((np.arange(len(imm_std_err_2g[3, :]))+1)*T, imm_std_err_2g[3, :].T, label='imm[3](2G)', marker='', color='red')
ax4.set_title("std_err_y(iteration)")
ax4.set_xlabel('Time,s')
ax4.set_ylabel('std_err_y, met')
ax4.grid(True)
plt.legend()

ax5 = fig.add_subplot(4,4,14)
ax5.plot((np.arange(len(imm_std_err_2g[4, :]))+1)*T, imm_std_err_2g[4, :].T, label='imm[3](2G)', marker='', color='red')
ax5.set_title("err_vy(iteration)")
ax5.set_xlabel('Time,s')
ax5.set_ylabel('std_err_vy, met')
ax5.grid(True)
plt.legend()

ax6 = fig.add_subplot(4,4,11)
ax6.plot((np.arange(len(imm_std_err_2g[6, :]))+1)*T, imm_std_err_2g[6, :].T, label='imm[3](2G)', marker='', color='red')
ax6.set_title("std_err_z(iteration)")
ax6.set_xlabel('Time,s')
ax6.set_ylabel('std_err_z, met')
ax6.grid(True)
plt.legend()

ax7 = fig.add_subplot(4,4,15)
ax7.plot((np.arange(len(imm_std_err_2g[7, :]))+1)*T, imm_std_err_2g[7, :].T, label='imm[3](2G)', marker='', color='red')
ax7.set_title("err_vz(iteration)")
ax7.set_xlabel('Time,s')
ax7.set_ylabel('std_err_vz, met')
ax7.grid(True)
plt.legend()

ax9 = fig.add_subplot(2,4,8)
ax9.plot(mus_2g[0, :], label='KF+CV', marker='', color='green')
ax9.plot(mus_2g[1, :], label='EKF+CT', marker='', color='orange')
ax9.plot(mus_2g[2, :], label='KF+CA', marker='', color='red')
ax9.set_title("mu[3](2G)")
ax9.set_xlabel('iteration')
ax9.set_ylabel('mu')
ax9.grid(True)
plt.legend()

plt.show()
