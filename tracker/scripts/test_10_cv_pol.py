#!/usr/bin/python3

import numpy as np
import numpy.matlib as ml
import matplotlib.pyplot as plt
import math
from IPython.display import display, Math, Latex
from IPython.display import Markdown as md

import estimator as e
import math
import stand_10_cv_pol as stand

T = 0.2
process_var = 0.5
process_var_w = 0.001
meas_dec_std = 1.
velo_dec_std = 0.1
meas_pol_r_std = 1.
meas_pol_a_std = 1e-4
meas_pol_e_std = 1e-4
velo_pol_r_std = 0.1
velo_pol_a_std = 1e-5
velo_pol_e_std = 1e-5

Rdec_p = np.diag([pow(meas_dec_std,2), pow(meas_dec_std,2), pow(meas_dec_std,2)])
Rdec_v = np.diag([pow(velo_dec_std,2), pow(velo_dec_std,2), pow(velo_dec_std,2)])

Rpol_p = np.diag([pow(meas_pol_r_std,2), pow(meas_pol_a_std,2), pow(meas_pol_e_std,2)])
Rpol_v = np.diag([pow(velo_pol_r_std,2), pow(velo_pol_a_std,2), pow(velo_pol_e_std,2)])

Q0 = np.diag([process_var, process_var, process_var, process_var_w])

G = e.BindG_10_matrix

Q = G(T)@Q0@G(T).T

x0 = np.array([20000., 200., 0., 20000., 0., 0., 1000., 0., 0., 0.])
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

u = np.zeros((10,1))
B = np.zeros((10,10))

amount = 100

X = stand.make_true_data(x0,amount,e.BindFCV_10,T)
Xn = stand.add_process_noise(X,Q)

Zn_dec = stand.make_meas(Xn,Rdec_p,e.BindHXX_10)
Zn_pol = stand.make_meas(Xn,Rpol_p,e.BindHPOL_10)

P0_dec  = Hp.T@Rdec_p@Hp + Hv.T@Rdec_v@Hv
P0_pol = stand.make_cartcov(Zn_pol[:, 0][:, np.newaxis],Rpol_p)

ekf_cv_est_dec = stand.estimate(e.BindEKF_10_CV_XX,x0,P0_dec,Q0,Rdec_p,Zn_dec,T)
ekf_cv_est_pol = stand.estimate(e.BindEKF_10_CV_POL,x0,P0_pol,Q0,Rpol_p,Zn_pol,T)

[aaa, ekf_cv_std_err_dec] = stand.test(e.BindEKF_10_CV_XX,
                                      x0,P0_dec,Q0,Rdec_p,
                                      e.BindFCV_10,
                                      e.BindHXX_10,
                                      e.BindG_10_matrix,
                                      6,100,2000)

[bbb, ekf_cv_std_err_pol] = stand.test(e.BindEKF_10_CV_POL,
                                      x0,P0_pol,Q0,Rpol_p,
                                      e.BindFCV_10,
                                      e.BindHPOL_10,
                                      e.BindG_10_matrix,
                                      6,100,2000)




fig = plt.figure("Тест для исследования фильтров с моделью CV, состояниями типа [x,vx,ax,y,vy,ay,z,vz,az,w] и измерениями типа [x,y,z] и [r,a,e]",figsize=(21,11))

ax1 = fig.add_subplot(1,3,1)
ax1.plot(X[3, :], X[0, :], label='true', marker='', color='black')
ax1.plot(Xn[3, :], Xn[0, :], label='true+R', marker='+', color='blue')
ax1.plot(ekf_cv_est_dec[3, :], ekf_cv_est_dec[0, :], label='estimation(dec)', marker='*', color='red')
ax1.plot(ekf_cv_est_pol[3, :], ekf_cv_est_pol[0, :], label='estimation(pol)', marker='*', color='green')
ax1.set_title("X(Y)")
ax1.set_xlabel('y,met')
ax1.set_ylabel('x,met')
ax1.grid(True)
plt.legend()

ax2 = fig.add_subplot(3,3,2)
ax2.plot((np.arange(len(ekf_cv_std_err_dec[0, :]))+1)*T, ekf_cv_std_err_dec[0, :].T, label='ekf_cv_dec', marker='', color='red')
ax2.plot((np.arange(len(ekf_cv_std_err_pol[0, :]))+1)*T, ekf_cv_std_err_pol[0, :].T, label='ekf_cv_pol', marker='', color='green')
ax2.set_title("std_err_x(iteration)")
ax2.set_xlabel('Time,s')
ax2.set_ylabel('std_err_x, met')
ax2.grid(True)
plt.legend()

ax3 = fig.add_subplot(3,3,3)
ax3.plot((np.arange(len(ekf_cv_std_err_dec[1, :]))+1)*T, ekf_cv_std_err_dec[1, :].T, label='ekf_cv_dec', marker='', color='red')
ax3.plot((np.arange(len(ekf_cv_std_err_pol[1, :]))+1)*T, ekf_cv_std_err_pol[1, :].T, label='ekf_cv_pol', marker='', color='green')
ax3.set_title("err_vx(iteration)")
ax3.set_xlabel('Time,s')
ax3.set_ylabel('std_err_vx, met')
ax3.grid(True)
plt.legend()

ax4 = fig.add_subplot(3,3,5)
ax4.plot((np.arange(len(ekf_cv_std_err_dec[3, :]))+1)*T, ekf_cv_std_err_dec[3, :].T, label='ekf_cv_dec', marker='', color='red')
ax4.plot((np.arange(len(ekf_cv_std_err_pol[3, :]))+1)*T, ekf_cv_std_err_pol[3, :].T, label='ekf_cv_pol', marker='', color='green')
ax4.set_title("std_err_y(iteration)")
ax4.set_xlabel('Time,s')
ax4.set_ylabel('std_err_y, met')
ax4.grid(True)
plt.legend()

ax5 = fig.add_subplot(3,3,6)
ax5.plot((np.arange(len(ekf_cv_std_err_dec[4, :]))+1)*T, ekf_cv_std_err_dec[4, :].T, label='ekf_cv_dec', marker='', color='red')
ax5.plot((np.arange(len(ekf_cv_std_err_pol[4, :]))+1)*T, ekf_cv_std_err_pol[4, :].T, label='ekf_cv_pol', marker='', color='green')
ax5.set_title("err_vy(iteration)")
ax5.set_xlabel('Time,s')
ax5.set_ylabel('std_err_vy, met')
ax5.grid(True)
plt.legend()

ax6 = fig.add_subplot(3,3,8)
ax6.plot((np.arange(len(ekf_cv_std_err_dec[6, :]))+1)*T, ekf_cv_std_err_dec[6, :].T, label='ekf_cv_dec', marker='', color='red')
ax6.plot((np.arange(len(ekf_cv_std_err_pol[6, :]))+1)*T, ekf_cv_std_err_pol[6, :].T, label='ekf_cv_pol', marker='', color='green')
ax6.set_title("std_err_z(iteration)")
ax6.set_xlabel('Time,s')
ax6.set_ylabel('std_err_z, met')
ax6.grid(True)
plt.legend()

ax7 = fig.add_subplot(3,3,9)
ax7.plot((np.arange(len(ekf_cv_std_err_dec[7, :]))+1)*T, ekf_cv_std_err_dec[7, :].T, label='ekf_cv_dec', marker='', color='red')
ax7.plot((np.arange(len(ekf_cv_std_err_pol[7, :]))+1)*T, ekf_cv_std_err_pol[7, :].T, label='ekf_cv_pol', marker='', color='green')
ax7.set_title("err_vz(iteration)")
ax7.set_xlabel('Time,s')
ax7.set_ylabel('std_err_vz, met')
ax7.grid(True)
plt.legend()

plt.show()
