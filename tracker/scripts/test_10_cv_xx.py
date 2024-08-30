#!/usr/bin/python3

import numpy as np
import numpy.matlib as ml
import matplotlib.pyplot as plt
import math
from IPython.display import display, Math, Latex
from IPython.display import Markdown as md

import estimator as e
import math
import stand_10_cv_xx as stand

T = 6
process_var = 1
process_var_w = 0
meas_std = 300
velo_std = 30
acc_std = 3

Rp = np.diag([pow(meas_std,2), pow(meas_std,2), pow(meas_std,2)])
Rv = np.diag([pow(velo_std,2), pow(velo_std,2), pow(velo_std,2)])
Ra = np.diag([pow(acc_std,2), pow(acc_std,2), pow(acc_std,2)])

Q0 = np.diag([process_var, process_var, process_var, process_var_w])

G = e.BindG_10

Q = G(T)@Q0@G(T).T

x0 = np.array([200., 200., 0., 0., 0., 0., 0., 0., 0., 0.])
x0 = x0[:, np.newaxis]

Hp = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]])
Hv = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]])
Ha = np.array([[0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]])
P0  = Hp.T@Rp@Hp + Hv.T@Rv@Hv# + Ha.T@Ra@Ha

u = np.zeros((10,1))
B = np.zeros((10,10))

amount = 100

X = stand.make_true_data(x0,amount,e.BindFCV_10,T)
Xn = stand.add_process_noise(X,Q)
Zn = stand.make_meas(Xn,Rp,e.BindHXX_10)

kf_cv_est = stand.estimate(e.BindKF_10_CV_XX,x0,P0,Q0,Rp,Zn,T)
kf_ca_est = stand.estimate(e.BindKF_10_CA_XX,x0,P0,Q0,Rp,Zn,T)
ekf_cv_est = stand.estimate(e.BindEKF_10_CV_XX,x0,P0,Q0,Rp,Zn,T)
ekf_ct_est = stand.estimate(e.BindEKF_10_CT_XX,x0,P0,Q0,Rp,Zn,T)
ekf_ca_est = stand.estimate(e.BindEKF_10_CA_XX,x0,P0,Q0,Rp,Zn,T)

[aaa, kf_cv_std_err] = stand.test(e.BindKF_10_CV_XX,
                                  x0,P0,Q0,Rp,
                                  e.BindFCV_10,
                                  e.BindHXX_10,
                                  e.BindG_10,
                                  6,100,2000)

[bbb, kf_ca_std_err] = stand.test(e.BindKF_10_CA_XX,
                                  x0,P0,Q0,Rp,
                                  e.BindFCV_10,
                                  e.BindHXX_10,
                                  e.BindG_10,
                                  6,100,2000)

[ccc, ekf_cv_std_err] = stand.test(e.BindEKF_10_CV_XX,
                                   x0,P0,Q0,Rp,
                                   e.BindFCV_10,
                                   e.BindHXX_10,
                                   e.BindG_10,
                                   6,100,2000)

[ddd, ekf_ct_std_err] = stand.test(e.BindEKF_10_CT_XX,
                                   x0,P0,Q0,Rp,
                                   e.BindFCV_10,
                                   e.BindHXX_10,
                                   e.BindG_10,
                                   6,100,2000)

[ddd, ekf_ca_std_err] = stand.test(e.BindEKF_10_CA_XX,
                                   x0,P0,Q0,Rp,
                                   e.BindFCV_10,
                                   e.BindHXX_10,
                                   e.BindG_10,
                                   6,100,2000)


fig = plt.figure("Тест для исследования фильтров с моделью CV, состояниями типа [x,vx,ax,y,vy,ay,z,vz,az,w] и измерениями типа [x,y,z]",figsize=(21,11))

ax1 = fig.add_subplot(1,3,1)
ax1.plot(X[3, :], X[0, :], label='true', marker='', color='black')
ax1.plot(Zn[1, :], Zn[0, :], label='measurement', marker='x', color='grey')
ax1.plot(kf_cv_est[3, :], kf_cv_est[0, :], label='kf_cv', marker='', color='red')
ax1.plot(kf_ca_est[3, :], kf_ca_est[0, :], label='kf_ca', marker='', color='tomato')
ax1.plot(ekf_cv_est[3, :], ekf_cv_est[0, :], label='ekf_cv', marker='', color='orange')
ax1.plot(ekf_ct_est[3, :], ekf_ct_est[0, :], label='ekf_ct', marker='', color='gold')
ax1.plot(ekf_ca_est[3, :], ekf_ca_est[0, :], label='ekf_ca', marker='', color='yellow')
ax1.set_title("X(Y)")
ax1.set_xlabel('y,met')
ax1.set_ylabel('x,met')
ax1.grid(True)
plt.legend()

ax2 = fig.add_subplot(3,3,2)
ax2.plot((np.arange(len(kf_cv_std_err[0, :]))+1)*T, kf_cv_std_err[0, :].T, label='kf_cv', marker='', color='red')
ax2.plot((np.arange(len(kf_ca_std_err[0, :]))+1)*T, kf_ca_std_err[0, :].T, label='kf_ca', marker='', color='tomato')
ax2.plot((np.arange(len(ekf_cv_std_err[0, :]))+1)*T, ekf_cv_std_err[0, :].T, label='ekf_cv', marker='', color='orange')
ax2.plot((np.arange(len(ekf_ct_std_err[0, :]))+1)*T, ekf_ct_std_err[0, :].T, label='ekf_ct', marker='', color='gold')
ax2.plot((np.arange(len(ekf_ca_std_err[0, :]))+1)*T, ekf_ca_std_err[0, :].T, label='ekf_ca', marker='', color='yellow')
ax2.set_title("std_err_x(iteration)")
ax2.set_xlabel('Time,s')
ax2.set_ylabel('std_err_x, met')
ax2.grid(True)
plt.legend()

ax3 = fig.add_subplot(3,3,3)
ax3.plot((np.arange(len(kf_cv_std_err[1, :]))+1)*T, kf_cv_std_err[1, :].T, label='kf_cv', marker='', color='red')
ax3.plot((np.arange(len(kf_ca_std_err[1, :]))+1)*T, kf_ca_std_err[1, :].T, label='kf_ca', marker='', color='tomato')
ax3.plot((np.arange(len(ekf_cv_std_err[1, :]))+1)*T, ekf_cv_std_err[1, :].T, label='ekf_cv', marker='', color='orange')
ax3.plot((np.arange(len(ekf_ct_std_err[1, :]))+1)*T, ekf_ct_std_err[1, :].T, label='ekf_ct', marker='', color='gold')
ax3.plot((np.arange(len(ekf_ca_std_err[1, :]))+1)*T, ekf_ca_std_err[1, :].T, label='ekf_ca', marker='', color='yellow')
ax3.set_title("err_vx(iteration)")
ax3.set_xlabel('Time,s')
ax3.set_ylabel('std_err_vx, met')
ax3.grid(True)
plt.legend()

ax4 = fig.add_subplot(3,3,5)
ax4.plot((np.arange(len(kf_cv_std_err[3, :]))+1)*T, kf_cv_std_err[3, :].T, label='kf_cv', marker='', color='red')
ax4.plot((np.arange(len(kf_ca_std_err[3, :]))+1)*T, kf_ca_std_err[3, :].T, label='kf_ca', marker='', color='tomato')
ax4.plot((np.arange(len(ekf_cv_std_err[3, :]))+1)*T, ekf_cv_std_err[3, :].T, label='ekf_cv', marker='', color='orange')
ax4.plot((np.arange(len(ekf_ct_std_err[3, :]))+1)*T, ekf_ct_std_err[3, :].T, label='ekf_ct', marker='', color='gold')
ax4.plot((np.arange(len(ekf_ca_std_err[3, :]))+1)*T, ekf_ca_std_err[3, :].T, label='ekf_ca', marker='', color='yellow')
ax4.set_title("std_err_y(iteration)")
ax4.set_xlabel('Time,s')
ax4.set_ylabel('std_err_y, met')
ax4.grid(True)
plt.legend()

ax5 = fig.add_subplot(3,3,6)
ax5.plot((np.arange(len(kf_cv_std_err[4, :]))+1)*T, kf_cv_std_err[4, :].T, label='kf_cv', marker='', color='red')
ax5.plot((np.arange(len(kf_ca_std_err[4, :]))+1)*T, kf_ca_std_err[4, :].T, label='kf_ca', marker='', color='tomato')
ax5.plot((np.arange(len(ekf_cv_std_err[4, :]))+1)*T, ekf_cv_std_err[4, :].T, label='ekf_cv', marker='', color='orange')
ax5.plot((np.arange(len(ekf_ct_std_err[4, :]))+1)*T, ekf_ct_std_err[4, :].T, label='ekf_ct', marker='', color='gold')
ax5.plot((np.arange(len(ekf_ca_std_err[4, :]))+1)*T, ekf_ca_std_err[4, :].T, label='ekf_ca', marker='', color='yellow')
ax5.set_title("err_vy(iteration)")
ax5.set_xlabel('Time,s')
ax5.set_ylabel('std_err_vy, met')
ax5.grid(True)
plt.legend()

ax6 = fig.add_subplot(3,3,8)
ax6.plot((np.arange(len(kf_cv_std_err[6, :]))+1)*T, kf_cv_std_err[6, :].T, label='kf_cv', marker='', color='red')
ax6.plot((np.arange(len(kf_ca_std_err[6, :]))+1)*T, kf_ca_std_err[6, :].T, label='kf_ca', marker='', color='tomato')
ax6.plot((np.arange(len(ekf_cv_std_err[6, :]))+1)*T, ekf_cv_std_err[6, :].T, label='ekf_cv', marker='', color='orange')
ax6.plot((np.arange(len(ekf_ct_std_err[6, :]))+1)*T, ekf_ct_std_err[6, :].T, label='ekf_ct', marker='', color='gold')
ax6.plot((np.arange(len(ekf_ca_std_err[6, :]))+1)*T, ekf_ca_std_err[6, :].T, label='ekf_ca', marker='', color='yellow')
ax6.set_title("std_err_z(iteration)")
ax6.set_xlabel('Time,s')
ax6.set_ylabel('std_err_z, met')
ax6.grid(True)
plt.legend()

ax7 = fig.add_subplot(3,3,9)
ax7.plot((np.arange(len(kf_cv_std_err[7, :]))+1)*T, kf_cv_std_err[7, :].T, label='kf_cv', marker='', color='red')
ax7.plot((np.arange(len(kf_ca_std_err[7, :]))+1)*T, kf_ca_std_err[7, :].T, label='kf_ca', marker='', color='tomato')
ax7.plot((np.arange(len(ekf_cv_std_err[7, :]))+1)*T, ekf_cv_std_err[7, :].T, label='ekf_cv', marker='', color='orange')
ax7.plot((np.arange(len(ekf_ct_std_err[7, :]))+1)*T, ekf_ct_std_err[7, :].T, label='ekf_ct', marker='', color='gold')
ax7.plot((np.arange(len(ekf_ca_std_err[7, :]))+1)*T, ekf_ca_std_err[7, :].T, label='ekf_ca', marker='', color='yellow')
ax7.set_title("err_vz(iteration)")
ax7.set_xlabel('Time,s')
ax7.set_ylabel('std_err_vz, met')
ax7.grid(True)
plt.legend()

plt.show()
