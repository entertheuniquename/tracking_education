#!/usr/bin/python3

import numpy as np
import numpy.matlib as ml
import matplotlib.pyplot as plt
import math
from IPython.display import display, Math, Latex
from IPython.display import Markdown as md

import estimator as e
import math
import stand_10_ca_xx_dive as stand

T = 6
process_var = 1
process_var_w = 0.000000001
meas_std = 300
velo_std = 30
acc_std = 3
w_std = 0

Rp = np.diag([pow(meas_std,2), pow(meas_std,2), pow(meas_std,2)])
Rv = np.diag([pow(velo_std,2), pow(velo_std,2), pow(velo_std,2)])
Ra = np.diag([pow(acc_std,2), pow(acc_std,2), pow(acc_std,2)])
Rw = np.diag([pow(w_std,2)])

Q0 = np.diag([process_var, process_var, process_var, process_var_w])

G = e.BindG_10

Q = G(T)@Q0@G(T).T

x05g = np.array([200., 200., 0., 0., 0., 0., 300000., 0., 0., 0.])
x05g = x05g[:, np.newaxis]

x03g = np.array([200., 200., 0., 0., 0., 0., 0., 0., 0., 0.])
x03g = x03g[:, np.newaxis]

Hp = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]])
Hv = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]])
Ha = np.array([[0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]])
Hw = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])
P0  = Hp.T@Rp@Hp + Hv.T@Rv@Hv + Ha.T@Ra@Ha + Hw.T@Rw@Hw

u = np.zeros((10,1))
B = np.zeros((10,10))

amount = 100

X5G = stand.make_true_data(x05g,amount,e.BindFCT_10,T,49.,-3396.)
X3G = stand.make_true_data(x03g,amount,e.BindFCT_10,T,29.4,436.)

Xn5G = stand.add_process_noise(X5G,Q)
Xn3G = stand.add_process_noise(X3G,Q)

Zn5G = stand.make_meas(Xn5G,Rp,e.BindHXX_10)
Zn3G = stand.make_meas(Xn3G,Rp,e.BindHXX_10)

ekf_ca_est_5g = stand.estimate(e.BindEKF_10_CA_XX,x05g,P0,Q0,Rp,Zn5G,T)
ekf_ca_est_3g = stand.estimate(e.BindEKF_10_CA_XX,x03g,P0,Q0,Rp,Zn3G,T)

[eee1, ekf_ca_std_err_5g] = stand.test(e.BindEKF_10_CA_XX,
                                       x05g,P0,Q0,Rp,
                                       e.BindFCT_10,
                                       e.BindHXX_10,
                                       e.BindG_10,
                                       6,100,2000,49.,-3396.)

[eee2, ekf_ca_std_err_3g] = stand.test(e.BindEKF_10_CA_XX,
                                       x03g,P0,Q0,Rp,
                                       e.BindFCT_10,
                                       e.BindHXX_10,
                                       e.BindG_10,
                                       6,100,2000,29.4,436.)

fig = plt.figure("Тест для исследования KF с моделью CА при пикировании и кабрировании с перегрузками в 5G И 3G, соответственно, и состояниями типа [x,vx,ax,y,vy,ay,z,vz,az,w] и измерениями типа [x,y,z]",figsize=(21,11))

ax1 = fig.add_subplot(3,2,1)
ax1.plot(X5G[0, :], X5G[3, :], label='true(down-5G)', marker='', color='black')
ax1.plot(X3G[0, :], X3G[3, :], label='true(up-3G)', marker='', color='black')
ax1.plot(Zn5G[0, :], Zn5G[1, :], label='measurement(down-5G)', marker='x', color='grey')
ax1.plot(Zn3G[0, :], Zn3G[1, :], label='measurement(up-3G)', marker='x', color='grey')
ax1.plot(ekf_ca_est_5g[0, :], ekf_ca_est_5g[3, :], label='ekf_ca(down-5G)', marker='', color='red')
ax1.plot(ekf_ca_est_3g[0, :], ekf_ca_est_3g[3, :], label='ekf_ca(up-3G)', marker='', color='brown')
ax1.set_title("Y(X)")
ax1.set_xlabel('x,met')
ax1.set_ylabel('y,met')
ax1.grid(True)
plt.legend()

ax8 = fig.add_subplot(3,2,2)
ax8.plot(X5G[0, :], X5G[6, :], label='true(down-5G)', marker='', color='black')
ax8.plot(X3G[0, :], X3G[6, :], label='true(up-3G)', marker='', color='black')
ax8.plot(Zn5G[0, :], Zn5G[2, :], label='measurement(down-5G)', marker='x', color='grey')
ax8.plot(Zn3G[0, :], Zn3G[2, :], label='measurement(up-3G)', marker='x', color='grey')
ax8.plot(ekf_ca_est_5g[0, :], ekf_ca_est_5g[6, :], label='ekf_ca(down-5G)', marker='', color='red')
ax8.plot(ekf_ca_est_3g[0, :], ekf_ca_est_3g[6, :], label='ekf_ca(up-3G)', marker='', color='brown')
ax8.set_title("Z(X)")
ax8.set_xlabel('x,met')
ax8.set_ylabel('z,met')
ax8.grid(True)
plt.legend()

ax2 = fig.add_subplot(3,3,4)
ax2.plot((np.arange(len(ekf_ca_std_err_5g[0, :]))+1)*T, ekf_ca_std_err_5g[0, :].T, label='ekf_ca(down-5G)', marker='', color='red')
ax2.plot((np.arange(len(ekf_ca_std_err_3g[0, :]))+1)*T, ekf_ca_std_err_3g[0, :].T, label='ekf_ca(up-3G)', marker='', color='brown')
ax2.set_title("std_err_x(iteration)")
ax2.set_xlabel('Time,s')
ax2.set_ylabel('std_err_x, met')
ax2.grid(True)
plt.legend()

ax3 = fig.add_subplot(3,3,7)
ax3.plot((np.arange(len(ekf_ca_std_err_5g[1, :]))+1)*T, ekf_ca_std_err_5g[1, :].T, label='ekf_ca(down-5G)', marker='', color='red')
ax3.plot((np.arange(len(ekf_ca_std_err_3g[1, :]))+1)*T, ekf_ca_std_err_3g[1, :].T, label='ekf_ca(up-3G)', marker='', color='brown')
ax3.set_title("err_vx(iteration)")
ax3.set_xlabel('Time,s')
ax3.set_ylabel('std_err_vx, met')
ax3.grid(True)
plt.legend()

ax4 = fig.add_subplot(3,3,5)
ax4.plot((np.arange(len(ekf_ca_std_err_5g[3, :]))+1)*T, ekf_ca_std_err_5g[3, :].T, label='ekf_ca(down-5G)', marker='', color='red')
ax4.plot((np.arange(len(ekf_ca_std_err_3g[3, :]))+1)*T, ekf_ca_std_err_3g[3, :].T, label='ekf_ca(up-3G)', marker='', color='brown')
ax4.set_title("std_err_y(iteration)")
ax4.set_xlabel('Time,s')
ax4.set_ylabel('std_err_y, met')
ax4.grid(True)
plt.legend()

ax5 = fig.add_subplot(3,3,8)
ax5.plot((np.arange(len(ekf_ca_std_err_5g[4, :]))+1)*T, ekf_ca_std_err_5g[4, :].T, label='ekf_ca(down-5G)', marker='', color='red')
ax5.plot((np.arange(len(ekf_ca_std_err_3g[4, :]))+1)*T, ekf_ca_std_err_3g[4, :].T, label='ekf_ca(up-3G)', marker='', color='brown')
ax5.set_title("err_vy(iteration)")
ax5.set_xlabel('Time,s')
ax5.set_ylabel('std_err_vy, met')
ax5.grid(True)
plt.legend()

ax6 = fig.add_subplot(3,3,6)
ax6.plot((np.arange(len(ekf_ca_std_err_5g[6, :]))+1)*T, ekf_ca_std_err_5g[6, :].T, label='ekf_ca(down-5G)', marker='', color='red')
ax6.plot((np.arange(len(ekf_ca_std_err_3g[6, :]))+1)*T, ekf_ca_std_err_3g[6, :].T, label='ekf_ca(up-3G)', marker='', color='brown')
ax6.set_title("std_err_z(iteration)")
ax6.set_xlabel('Time,s')
ax6.set_ylabel('std_err_z, met')
ax6.grid(True)
plt.legend()

ax7 = fig.add_subplot(3,3,9)
ax7.plot((np.arange(len(ekf_ca_std_err_5g[7, :]))+1)*T, ekf_ca_std_err_5g[7, :].T, label='ekf_ca(down-5G)', marker='', color='red')
ax7.plot((np.arange(len(ekf_ca_std_err_3g[7, :]))+1)*T, ekf_ca_std_err_3g[7, :].T, label='ekf_ca(up-3G)', marker='', color='brown')
ax7.set_title("err_vz(iteration)")
ax7.set_xlabel('Time,s')
ax7.set_ylabel('std_err_vz, met')
ax7.grid(True)
plt.legend()

plt.show()
