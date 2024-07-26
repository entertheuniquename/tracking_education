#!/usr/bin/python3

import numpy as np
import numpy.matlib as ml
import matplotlib.pyplot as plt
import math
from IPython.display import display, Math, Latex
from IPython.display import Markdown as md

import estimator as e
import math
import stand_10_4cross as stand

T = 6
process_var = 1
process_var_w = 0.000000001
meas_std = 300
velo_std = 30
acc_std = 3
w_std = 0.392

Rp = np.diag([pow(meas_std,2), pow(meas_std,2), pow(meas_std,2)])
Rv = np.diag([pow(velo_std,2), pow(velo_std,2), pow(velo_std,2)])
Ra = np.diag([pow(acc_std,2), pow(acc_std,2), pow(acc_std,2)])
Rw = np.diag([pow(w_std,2)])

Q0 = np.diag([process_var, process_var, process_var, process_var_w])

G = e.BindG_10

Q = G(T)@Q0@G(T).T

x01 = np.array([10., 200., 0., 10., 200., 0., 0., 0., 0., 0.])
x01 = x01[:, np.newaxis]

x02 = np.array([100000., -200., 0., 100000., -200., 0., 0., 0., 0., 0.])
x02 = x02[:, np.newaxis]

x03 = np.array([100000., -200., 0., 10., 200., 0., 0., 0., 0., 0.])
x03 = x03[:, np.newaxis]

x04 = np.array([10., 200., 0., 100000., -200., 0., 0., 0., 0., 0.])
x04 = x04[:, np.newaxis]

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

X1 = stand.make_true_data(x01,amount,e.BindFCV_10,T)
X2 = stand.make_true_data(x02,amount,e.BindFCV_10,T)
X3 = stand.make_true_data(x03,amount,e.BindFCV_10,T)
X4 = stand.make_true_data(x04,amount,e.BindFCV_10,T)

Xn1 = stand.add_process_noise(X1,Q)
Xn2 = stand.add_process_noise(X2,Q)
Xn3 = stand.add_process_noise(X3,Q)
Xn4 = stand.add_process_noise(X4,Q)

Zn1 = stand.make_meas(Xn1,Rp,e.BindHXX_10)
Zn2 = stand.make_meas(Xn2,Rp,e.BindHXX_10)
Zn3 = stand.make_meas(Xn3,Rp,e.BindHXX_10)
Zn4 = stand.make_meas(Xn4,Rp,e.BindHXX_10)

#ekf = stand.estimate(e.BindEKF_10_CV_XX,x01,x02,x03,x04,P0,Q0,Rp,Zn1,T)

#[ddd0, ekf_ct_std_err_2g] = stand.test(e.BindEKF_10_CT_XX,
#                                       x0_2g,P0,Q0,Rp,
#                                       e.BindFCT_10,
#                                       e.BindHXX_10,
#                                       e.BindG_10,
#                                       6,100,2000,0.098)

#[ddd1, ekf_ct_std_err_5g] = stand.test(e.BindEKF_10_CT_XX,
#                                       x0_5g,P0,Q0,Rp,
#                                       e.BindFCT_10,
#                                       e.BindHXX_10,
#                                       e.BindG_10,
#                                       6,100,2000,0.245)

#[ddd2, ekf_ct_std_err_8g] = stand.test(e.BindEKF_10_CT_XX,
#                                       x0_8g,P0,Q0,Rp,
#                                       e.BindFCT_10,
#                                       e.BindHXX_10,
#                                       e.BindG_10,
#                                       6,100,2000,0.392)

#[eee2, ekf_ca_std_err_2g] = stand.test(e.BindEKF_10_CA_XX,
#                                       x0_2g,P0,Q0,Rp,
#                                       e.BindFCT_10,
#                                       e.BindHXX_10,
#                                       e.BindG_10,
#                                       6,100,2000,0.098)

#[eee2, ekf_ca_std_err_5g] = stand.test(e.BindEKF_10_CA_XX,
#                                       x0_5g,P0,Q0,Rp,
#                                       e.BindFCT_10,
#                                       e.BindHXX_10,
#                                       e.BindG_10,
#                                       6,100,2000,0.245)

#[eee2, ekf_ca_std_err_8g] = stand.test(e.BindEKF_10_CA_XX,
#                                       x0_8g,P0,Q0,Rp,
#                                       e.BindFCT_10,
#                                       e.BindHXX_10,
#                                       e.BindG_10,
#                                       6,100,2000,0.392)

fig = plt.figure("Тест для исследования фильтров с моделью CT и перегрузками в 2G, 5G и 8G, состояниями типа [x,vx,ax,y,vy,ay,z,vz,az,w] и измерениями типа [x,y,z]",figsize=(21,11))

ax1 = fig.add_subplot(1,3,1)
ax1.plot(X1[3, :], X1[0, :], label='track 1', marker='', color='red')
ax1.plot(X2[3, :], X2[0, :], label='track 2', marker='', color='blue')
ax1.plot(X3[3, :], X3[0, :], label='track 3', marker='', color='green')
ax1.plot(X4[3, :], X4[0, :], label='track 4', marker='', color='orange')
ax1.plot(Zn1[1, :], Zn1[0, :], label='measurement 1', linestyle='', marker='x', color='red')
ax1.plot(Zn2[1, :], Zn2[0, :], label='measurement 2', linestyle='', marker='x', color='blue')
ax1.plot(Zn3[1, :], Zn3[0, :], label='measurement 3', linestyle='', marker='x', color='green')
ax1.plot(Zn4[1, :], Zn4[0, :], label='measurement 4', linestyle='', marker='x', color='orange')
#ax1.plot(ekf_ct_est_2g[3, :], ekf_ct_est_2g[0, :], label='ekf_ct(2G)', marker='', color='orange')
#ax1.plot(ekf_ct_est_5g[3, :], ekf_ct_est_5g[0, :], label='ekf_ct(5G)', marker='', color='orange')
#ax1.plot(ekf_ct_est_8g[3, :], ekf_ct_est_8g[0, :], label='ekf_ct(8G)', marker='', color='orange')
#ax1.plot(ekf_ca_est_2g[3, :], ekf_ca_est_2g[0, :], label='ekf_ca(2G)', marker='', color='brown')
#ax1.plot(ekf_ca_est_5g[3, :], ekf_ca_est_5g[0, :], label='ekf_ca(5G)', marker='', color='brown')
#ax1.plot(ekf_ca_est_8g[3, :], ekf_ca_est_8g[0, :], label='ekf_ca(8G)', marker='', color='brown')
ax1.set_title("X(Y)")
ax1.set_xlabel('y,met')
ax1.set_ylabel('x,met')
ax1.grid(True)
plt.legend()

#ax2 = fig.add_subplot(3,3,2)
#ax2.plot((np.arange(len(ekf_ct_std_err_2g[0, :]))+1)*T, ekf_ct_std_err_2g[0, :].T, label='ekf_ct(2G)', marker='', color='orange')
#ax2.plot((np.arange(len(ekf_ct_std_err_5g[0, :]))+1)*T, ekf_ct_std_err_5g[0, :].T, label='ekf_ct(5G)', marker='', color='orange')
#ax2.plot((np.arange(len(ekf_ct_std_err_8g[0, :]))+1)*T, ekf_ct_std_err_8g[0, :].T, label='ekf_ct(8G)', marker='', color='orange')
#ax2.plot((np.arange(len(ekf_ca_std_err_2g[0, :]))+1)*T, ekf_ca_std_err_2g[0, :].T, label='ekf_ca(2G)', marker='', color='brown')
#ax2.plot((np.arange(len(ekf_ca_std_err_5g[0, :]))+1)*T, ekf_ca_std_err_5g[0, :].T, label='ekf_ca(5G)', marker='', color='brown')
#ax2.plot((np.arange(len(ekf_ca_std_err_8g[0, :]))+1)*T, ekf_ca_std_err_8g[0, :].T, label='ekf_ca(8G)', marker='', color='brown')
#ax2.set_title("std_err_x(iteration)")
#ax2.set_xlabel('Time,s')
#ax2.set_ylabel('std_err_x, met')
#ax2.grid(True)
#plt.legend()

#ax3 = fig.add_subplot(3,3,3)
#ax3.plot((np.arange(len(ekf_ct_std_err_2g[1, :]))+1)*T, ekf_ct_std_err_2g[1, :].T, label='ekf_ct(2G)', marker='', color='orange')
#ax3.plot((np.arange(len(ekf_ct_std_err_5g[1, :]))+1)*T, ekf_ct_std_err_5g[1, :].T, label='ekf_ct(5G)', marker='', color='orange')
#ax3.plot((np.arange(len(ekf_ct_std_err_8g[1, :]))+1)*T, ekf_ct_std_err_8g[1, :].T, label='ekf_ct(8G)', marker='', color='orange')
#ax3.plot((np.arange(len(ekf_ca_std_err_2g[1, :]))+1)*T, ekf_ca_std_err_2g[1, :].T, label='ekf_ca(2G)', marker='', color='brown')
#ax3.plot((np.arange(len(ekf_ca_std_err_5g[1, :]))+1)*T, ekf_ca_std_err_5g[1, :].T, label='ekf_ca(5G)', marker='', color='brown')
#ax3.plot((np.arange(len(ekf_ca_std_err_8g[1, :]))+1)*T, ekf_ca_std_err_8g[1, :].T, label='ekf_ca(8G)', marker='', color='brown')
#ax3.set_title("err_vx(iteration)")
#ax3.set_xlabel('Time,s')
#ax3.set_ylabel('std_err_vx, met')
#ax3.grid(True)
#plt.legend()

#ax4 = fig.add_subplot(3,3,5)
#ax4.plot((np.arange(len(ekf_ct_std_err_2g[3, :]))+1)*T, ekf_ct_std_err_2g[3, :].T, label='ekf_ct(2G)', marker='', color='orange')
#ax4.plot((np.arange(len(ekf_ct_std_err_5g[3, :]))+1)*T, ekf_ct_std_err_5g[3, :].T, label='ekf_ct(5G)', marker='', color='orange')
#ax4.plot((np.arange(len(ekf_ct_std_err_8g[3, :]))+1)*T, ekf_ct_std_err_8g[3, :].T, label='ekf_ct(8G)', marker='', color='orange')
#ax4.plot((np.arange(len(ekf_ca_std_err_2g[3, :]))+1)*T, ekf_ca_std_err_2g[3, :].T, label='ekf_ca(2G)', marker='', color='brown')
#ax4.plot((np.arange(len(ekf_ca_std_err_5g[3, :]))+1)*T, ekf_ca_std_err_5g[3, :].T, label='ekf_ca(5G)', marker='', color='brown')
#ax4.plot((np.arange(len(ekf_ca_std_err_8g[3, :]))+1)*T, ekf_ca_std_err_8g[3, :].T, label='ekf_ca(8G)', marker='', color='brown')
#ax4.set_title("std_err_y(iteration)")
#ax4.set_xlabel('Time,s')
#ax4.set_ylabel('std_err_y, met')
#ax4.grid(True)
#plt.legend()

#ax5 = fig.add_subplot(3,3,6)
#ax5.plot((np.arange(len(ekf_ct_std_err_2g[4, :]))+1)*T, ekf_ct_std_err_2g[4, :].T, label='ekf_ct(2G)', marker='', color='orange')
#ax5.plot((np.arange(len(ekf_ct_std_err_5g[4, :]))+1)*T, ekf_ct_std_err_5g[4, :].T, label='ekf_ct(5G)', marker='', color='orange')
#ax5.plot((np.arange(len(ekf_ct_std_err_8g[4, :]))+1)*T, ekf_ct_std_err_8g[4, :].T, label='ekf_ct(8G)', marker='', color='orange')
#ax5.plot((np.arange(len(ekf_ca_std_err_2g[4, :]))+1)*T, ekf_ca_std_err_2g[4, :].T, label='ekf_ca(2G)', marker='', color='brown')
#ax5.plot((np.arange(len(ekf_ca_std_err_5g[4, :]))+1)*T, ekf_ca_std_err_5g[4, :].T, label='ekf_ca(5G)', marker='', color='brown')
#ax5.plot((np.arange(len(ekf_ca_std_err_8g[4, :]))+1)*T, ekf_ca_std_err_8g[4, :].T, label='ekf_ca(8G)', marker='', color='brown')
#ax5.set_title("err_vy(iteration)")
#ax5.set_xlabel('Time,s')
#ax5.set_ylabel('std_err_vy, met')
#ax5.grid(True)
#plt.legend()

#ax6 = fig.add_subplot(3,3,8)
#ax6.plot((np.arange(len(ekf_ct_std_err_2g[6, :]))+1)*T, ekf_ct_std_err_2g[6, :].T, label='ekf_ct(2G)', marker='', color='orange')
#ax6.plot((np.arange(len(ekf_ct_std_err_5g[6, :]))+1)*T, ekf_ct_std_err_5g[6, :].T, label='ekf_ct(5G)', marker='', color='orange')
#ax6.plot((np.arange(len(ekf_ct_std_err_8g[6, :]))+1)*T, ekf_ct_std_err_8g[6, :].T, label='ekf_ct(8G)', marker='', color='orange')
#ax6.plot((np.arange(len(ekf_ca_std_err_2g[6, :]))+1)*T, ekf_ca_std_err_2g[6, :].T, label='ekf_ca(2G)', marker='', color='brown')
#ax6.plot((np.arange(len(ekf_ca_std_err_5g[6, :]))+1)*T, ekf_ca_std_err_5g[6, :].T, label='ekf_ca(5G)', marker='', color='brown')
#ax6.plot((np.arange(len(ekf_ca_std_err_8g[6, :]))+1)*T, ekf_ca_std_err_8g[6, :].T, label='ekf_ca(8G)', marker='', color='brown')
#ax6.set_title("std_err_z(iteration)")
#ax6.set_xlabel('Time,s')
#ax6.set_ylabel('std_err_z, met')
#ax6.grid(True)
#plt.legend()

#ax7 = fig.add_subplot(3,3,9)
#ax7.plot((np.arange(len(ekf_ct_std_err_2g[7, :]))+1)*T, ekf_ct_std_err_2g[7, :].T, label='ekf_ct(2G)', marker='', color='orange')
#ax7.plot((np.arange(len(ekf_ct_std_err_5g[7, :]))+1)*T, ekf_ct_std_err_5g[7, :].T, label='ekf_ct(5G)', marker='', color='orange')
#ax7.plot((np.arange(len(ekf_ct_std_err_8g[7, :]))+1)*T, ekf_ct_std_err_8g[7, :].T, label='ekf_ct(8G)', marker='', color='orange')
#ax7.plot((np.arange(len(ekf_ca_std_err_2g[7, :]))+1)*T, ekf_ca_std_err_2g[7, :].T, label='ekf_ca(2G)', marker='', color='brown')
#ax7.plot((np.arange(len(ekf_ca_std_err_5g[7, :]))+1)*T, ekf_ca_std_err_5g[7, :].T, label='ekf_ca(5G)', marker='', color='brown')
#ax7.plot((np.arange(len(ekf_ca_std_err_8g[7, :]))+1)*T, ekf_ca_std_err_8g[7, :].T, label='ekf_ca(8G)', marker='', color='brown')
#ax7.set_title("err_vz(iteration)")
#ax7.set_xlabel('Time,s')
#ax7.set_ylabel('std_err_vz, met')
#ax7.grid(True)
#plt.legend()

plt.show()
