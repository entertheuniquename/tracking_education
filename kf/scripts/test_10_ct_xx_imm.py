#!/usr/bin/python3

import numpy as np
import numpy.matlib as ml
import matplotlib.pyplot as plt
import math
from IPython.display import display, Math, Latex
from IPython.display import Markdown as md

import estimator as e
import math
import stand_10_ct_xx as stand

T = 6.
process_var = 1.
process_var_w = 0.000000001
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

x0_2g = np.array([200., 200., 0., 0., 0., 0., 0., 0., 0., 0.])
x0_2g = x0_2g[:, np.newaxis]

x0_5g = np.array([10200., 200., 0., 0., 0., 0., 0., 0., 0., 0.])
x0_5g = x0_5g[:, np.newaxis]

x0_8g = np.array([20200., 200., 0., 0., 0., 0., 0., 0., 0., 0.])
x0_8g = x0_8g[:, np.newaxis]

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

X2G = stand.make_true_data(x0_2g,amount,e.BindFCT_10,T,0.098)
X5G = stand.make_true_data(x0_5g,amount,e.BindFCT_10,T,0.245)
X8G = stand.make_true_data(x0_8g,amount,e.BindFCT_10,T,0.392)

Xn2G = stand.add_process_noise(X2G,Q)
Xn5G = stand.add_process_noise(X5G,Q)
Xn8G = stand.add_process_noise(X8G,Q)

Zn2G = stand.make_meas(Xn2G,Rp,e.BindHXX_10)
Zn5G = stand.make_meas(Xn5G,Rp,e.BindHXX_10)
Zn8G = stand.make_meas(Xn8G,Rp,e.BindHXX_10)

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
imm_est_5g, mus_5g = stand.estimate(e.BindIMM_10_KFCV_EKFCT_KFCA,x0_5g,P0,Q0,Rp,Zn5G,T,MU=mu,TP=tp,imm_filters_amount=3)
imm_est_8g, mus_8g = stand.estimate(e.BindIMM_10_KFCV_EKFCT_KFCA,x0_8g,P0,Q0,Rp,Zn8G,T,MU=mu,TP=tp,imm_filters_amount=3)

imm4_est_2g, mus4_2g = stand.estimate(e.BindIMM_10_KFCV_EKFCT_KFCA_EKFCTv,x0_2g,P0,Q0,Rp,Zn2G,T,MU=mu4,TP=tp4,imm_filters_amount=4)
imm4_est_5g, mus4_5g = stand.estimate(e.BindIMM_10_KFCV_EKFCT_KFCA_EKFCTv,x0_5g,P0,Q0,Rp,Zn5G,T,MU=mu4,TP=tp4,imm_filters_amount=4)
imm4_est_8g, mus4_8g = stand.estimate(e.BindIMM_10_KFCV_EKFCT_KFCA_EKFCTv,x0_8g,P0,Q0,Rp,Zn8G,T,MU=mu4,TP=tp4,imm_filters_amount=4)

[fff0, imm_std_err_2g] = stand.test(e.BindIMM_10_KFCV_EKFCT_KFCA,
                                   x0_2g,P0,Q0,Rp,
                                   e.BindFCT_10,
                                   e.BindHXX_10,
                                   e.BindG_10,
                                   6,100,2000,0.098,MU=mu,TP=tp,imm_filters_amount=3)

[fff1, imm_std_err_5g] = stand.test(e.BindIMM_10_KFCV_EKFCT_KFCA,
                                   x0_5g,P0,Q0,Rp,
                                   e.BindFCT_10,
                                   e.BindHXX_10,
                                   e.BindG_10,
                                   6,100,2000,0.245,MU=mu,TP=tp,imm_filters_amount=3)

[fff2, imm_std_err_8g] = stand.test(e.BindIMM_10_KFCV_EKFCT_KFCA,
                                   x0_8g,P0,Q0,Rp,
                                   e.BindFCT_10,
                                   e.BindHXX_10,
                                   e.BindG_10,
                                   6,100,2000,0.392,MU=mu,TP=tp,imm_filters_amount=3)

[fff40, imm4_std_err_2g] = stand.test(e.BindIMM_10_KFCV_EKFCT_KFCA_EKFCTv,
                                      x0_2g,P0,Q0,Rp,
                                      e.BindFCT_10,
                                      e.BindHXX_10,
                                      e.BindG_10,
                                      6,100,2000,0.098,MU=mu4,TP=tp4,imm_filters_amount=4)

[fff41, imm4_std_err_5g] = stand.test(e.BindIMM_10_KFCV_EKFCT_KFCA_EKFCTv,
                                      x0_5g,P0,Q0,Rp,
                                      e.BindFCT_10,
                                      e.BindHXX_10,
                                      e.BindG_10,
                                      6,100,2000,0.098,MU=mu4,TP=tp4,imm_filters_amount=4)

[fff41, imm4_std_err_8g] = stand.test(e.BindIMM_10_KFCV_EKFCT_KFCA_EKFCTv,
                                      x0_8g,P0,Q0,Rp,
                                      e.BindFCT_10,
                                      e.BindHXX_10,
                                      e.BindG_10,
                                      6,100,2000,0.098,MU=mu4,TP=tp4,imm_filters_amount=4)

fig = plt.figure("Тест для исследования фильтра IMM и перегрузками в 2G, 5G и 8G, состояниями типа [x,vx,ax,y,vy,ay,z,vz,az,w] и измерениями типа [x,y,z]",figsize=(21,11))

ax1 = fig.add_subplot(1,4,1)
ax1.plot(X2G[3, :], X2G[0, :], label='true(2G)', marker='', color='black')
ax1.plot(X5G[3, :], X5G[0, :], label='true(5G)', marker='', color='black')
ax1.plot(X8G[3, :], X8G[0, :], label='true(8G)', marker='', color='black')
ax1.plot(Zn2G[1, :], Zn2G[0, :], label='measurement(2G)', marker='x', color='grey')
ax1.plot(Zn5G[1, :], Zn5G[0, :], label='measurement(5G)', marker='x', color='grey')
ax1.plot(Zn8G[1, :], Zn8G[0, :], label='measurement(8G)', marker='x', color='grey')
ax1.plot(imm_est_2g[3, :], imm_est_2g[0, :], label='imm[3](2G)', marker='', color='red')
ax1.plot(imm_est_5g[3, :], imm_est_5g[0, :], label='imm[3](5G)', marker='', color='blue')
ax1.plot(imm_est_8g[3, :], imm_est_8g[0, :], label='imm[3](8G)', marker='', color='green')
ax1.plot(imm4_est_2g[3, :], imm4_est_2g[0, :], label='imm[4](2G)', marker='', color='purple')
ax1.plot(imm4_est_5g[3, :], imm4_est_5g[0, :], label='imm[4](5G)', marker='', color='pink')
ax1.plot(imm4_est_8g[3, :], imm4_est_8g[0, :], label='imm[4](8G)', marker='', color='brown')
ax1.set_title("X(Y)")
ax1.set_xlabel('y,met')
ax1.set_ylabel('x,met')
ax1.grid(True)
plt.legend()

ax2 = fig.add_subplot(4,4,2)
ax2.plot((np.arange(len(imm_std_err_2g[0, :]))+1)*T, imm_std_err_2g[0, :].T, label='imm[3](2G)', marker='', color='red')
ax2.plot((np.arange(len(imm_std_err_5g[0, :]))+1)*T, imm_std_err_5g[0, :].T, label='imm[3](5G)', marker='', color='blue')
ax2.plot((np.arange(len(imm_std_err_8g[0, :]))+1)*T, imm_std_err_8g[0, :].T, label='imm[3](8G)', marker='', color='green')
ax2.plot((np.arange(len(imm4_std_err_2g[0, :]))+1)*T, imm4_std_err_2g[0, :].T, label='imm[4](2G)', marker='', color='purple')
ax2.plot((np.arange(len(imm4_std_err_5g[0, :]))+1)*T, imm4_std_err_5g[0, :].T, label='imm[4](5G)', marker='', color='pink')
ax2.plot((np.arange(len(imm4_std_err_8g[0, :]))+1)*T, imm4_std_err_8g[0, :].T, label='imm[4](8G)', marker='', color='brown')
ax2.set_title("std_err_x(iteration)")
ax2.set_xlabel('Time,s')
ax2.set_ylabel('std_err_x, met')
ax2.grid(True)
plt.legend()

ax3 = fig.add_subplot(4,4,6)
ax3.plot((np.arange(len(imm_std_err_2g[1, :]))+1)*T, imm_std_err_2g[1, :].T, label='imm[3](2G)', marker='', color='red')
ax3.plot((np.arange(len(imm_std_err_5g[1, :]))+1)*T, imm_std_err_5g[1, :].T, label='imm[3](5G)', marker='', color='blue')
ax3.plot((np.arange(len(imm_std_err_8g[1, :]))+1)*T, imm_std_err_8g[1, :].T, label='imm[3](8G)', marker='', color='green')
ax3.plot((np.arange(len(imm4_std_err_2g[1, :]))+1)*T, imm4_std_err_2g[1, :].T, label='imm[4](2G)', marker='', color='purple')
ax3.plot((np.arange(len(imm4_std_err_5g[1, :]))+1)*T, imm4_std_err_5g[1, :].T, label='imm[4](5G)', marker='', color='pink')
ax3.plot((np.arange(len(imm4_std_err_8g[1, :]))+1)*T, imm4_std_err_8g[1, :].T, label='imm[4](8G)', marker='', color='brown')
ax3.set_title("err_vx(iteration)")
ax3.set_xlabel('Time,s')
ax3.set_ylabel('std_err_vx, met')
ax3.grid(True)
plt.legend()

ax4 = fig.add_subplot(4,4,3)
ax4.plot((np.arange(len(imm_std_err_2g[3, :]))+1)*T, imm_std_err_2g[3, :].T, label='imm[3](2G)', marker='', color='red')
ax4.plot((np.arange(len(imm_std_err_5g[3, :]))+1)*T, imm_std_err_5g[3, :].T, label='imm[3](5G)', marker='', color='blue')
ax4.plot((np.arange(len(imm_std_err_8g[3, :]))+1)*T, imm_std_err_8g[3, :].T, label='imm[3](8G)', marker='', color='green')
ax4.plot((np.arange(len(imm4_std_err_2g[3, :]))+1)*T, imm4_std_err_2g[3, :].T, label='imm[4](2G)', marker='', color='purple')
ax4.plot((np.arange(len(imm4_std_err_5g[3, :]))+1)*T, imm4_std_err_5g[3, :].T, label='imm[4](5G)', marker='', color='pink')
ax4.plot((np.arange(len(imm4_std_err_8g[3, :]))+1)*T, imm4_std_err_8g[3, :].T, label='imm[4](8G)', marker='', color='brown')
ax4.set_title("std_err_y(iteration)")
ax4.set_xlabel('Time,s')
ax4.set_ylabel('std_err_y, met')
ax4.grid(True)
plt.legend()

ax5 = fig.add_subplot(4,4,7)
ax5.plot((np.arange(len(imm_std_err_2g[4, :]))+1)*T, imm_std_err_2g[4, :].T, label='imm[3](2G)', marker='', color='red')
ax5.plot((np.arange(len(imm_std_err_5g[4, :]))+1)*T, imm_std_err_5g[4, :].T, label='imm[3](5G)', marker='', color='blue')
ax5.plot((np.arange(len(imm_std_err_8g[4, :]))+1)*T, imm_std_err_8g[4, :].T, label='imm[3](8G)', marker='', color='green')
ax5.plot((np.arange(len(imm4_std_err_2g[4, :]))+1)*T, imm4_std_err_2g[4, :].T, label='imm[4](2G)', marker='', color='purple')
ax5.plot((np.arange(len(imm4_std_err_5g[4, :]))+1)*T, imm4_std_err_5g[4, :].T, label='imm[4](5G)', marker='', color='pink')
ax5.plot((np.arange(len(imm4_std_err_8g[4, :]))+1)*T, imm4_std_err_8g[4, :].T, label='imm[4](8G)', marker='', color='brown')
ax5.set_title("err_vy(iteration)")
ax5.set_xlabel('Time,s')
ax5.set_ylabel('std_err_vy, met')
ax5.grid(True)
plt.legend()

ax6 = fig.add_subplot(4,4,4)
ax6.plot((np.arange(len(imm_std_err_2g[6, :]))+1)*T, imm_std_err_2g[6, :].T, label='imm[3](2G)', marker='', color='red')
ax6.plot((np.arange(len(imm_std_err_5g[6, :]))+1)*T, imm_std_err_5g[6, :].T, label='imm[3](5G)', marker='', color='blue')
ax6.plot((np.arange(len(imm_std_err_8g[6, :]))+1)*T, imm_std_err_8g[6, :].T, label='imm[3](8G)', marker='', color='green')
ax6.plot((np.arange(len(imm4_std_err_2g[6, :]))+1)*T, imm4_std_err_2g[6, :].T, label='imm[4](2G)', marker='', color='purple')
ax6.plot((np.arange(len(imm4_std_err_5g[6, :]))+1)*T, imm4_std_err_5g[6, :].T, label='imm[4](5G)', marker='', color='pink')
ax6.plot((np.arange(len(imm4_std_err_8g[6, :]))+1)*T, imm4_std_err_8g[6, :].T, label='imm[4](8G)', marker='', color='brown')
ax6.set_title("std_err_z(iteration)")
ax6.set_xlabel('Time,s')
ax6.set_ylabel('std_err_z, met')
ax6.grid(True)
plt.legend()

ax7 = fig.add_subplot(4,4,8)
ax7.plot((np.arange(len(imm_std_err_2g[7, :]))+1)*T, imm_std_err_2g[7, :].T, label='imm[3](2G)', marker='', color='red')
ax7.plot((np.arange(len(imm_std_err_5g[7, :]))+1)*T, imm_std_err_5g[7, :].T, label='imm[3](5G)', marker='', color='blue')
ax7.plot((np.arange(len(imm_std_err_8g[7, :]))+1)*T, imm_std_err_8g[7, :].T, label='imm[3](8G)', marker='', color='green')
ax7.plot((np.arange(len(imm4_std_err_2g[7, :]))+1)*T, imm4_std_err_2g[7, :].T, label='imm[4](2G)', marker='', color='purple')
ax7.plot((np.arange(len(imm4_std_err_5g[7, :]))+1)*T, imm4_std_err_5g[7, :].T, label='imm[4](5G)', marker='', color='pink')
ax7.plot((np.arange(len(imm4_std_err_8g[7, :]))+1)*T, imm4_std_err_8g[7, :].T, label='imm[4](8G)', marker='', color='brown')
ax7.set_title("err_vz(iteration)")
ax7.set_xlabel('Time,s')
ax7.set_ylabel('std_err_vz, met')
ax7.grid(True)
plt.legend()

ax9 = fig.add_subplot(4,4,10)
ax9.plot(mus_2g[0, :], label='KF+CV', marker='', color='green')
ax9.plot(mus_2g[1, :], label='EKF+CT', marker='', color='orange')
ax9.plot(mus_2g[2, :], label='KF+CA', marker='', color='red')
ax9.set_title("mu[3](2G)")
ax9.set_xlabel('iteration')
ax9.set_ylabel('mu')
ax9.grid(True)
plt.legend()

ax10 = fig.add_subplot(4,4,11)
ax10.plot(mus_5g[0, :], label='KF+CV', marker='', color='green')
ax10.plot(mus_5g[1, :], label='EKF+CT', marker='', color='orange')
ax10.plot(mus_5g[2, :], label='KF+CA', marker='', color='red')
ax10.set_title("mu[3](5G)")
ax10.set_xlabel('iteration')
ax10.set_ylabel('mu')
ax10.grid(True)
plt.legend()

ax11 = fig.add_subplot(4,4,12)
ax11.plot(mus_8g[0, :], label='KF+CV', marker='', color='green')
ax11.plot(mus_8g[1, :], label='EKF+CT', marker='', color='orange')
ax11.plot(mus_8g[2, :], label='KF+CA', marker='', color='red')
ax11.set_title("mu[3](8G)")
ax11.set_xlabel('iteration')
ax11.set_ylabel('mu')
ax11.grid(True)
plt.legend()

ax12 = fig.add_subplot(4,4,14)
ax12.plot(mus4_2g[0, :], label='KF+CV', marker='', color='green')
ax12.plot(mus4_2g[1, :], label='EKF+CT', marker='', color='orange')
ax12.plot(mus4_2g[2, :], label='KF+CA', marker='', color='red')
ax12.plot(mus4_2g[3, :], label='EKF+CTv', marker='', color='purple')
ax12.set_title("mu[4](2G)")
ax12.set_xlabel('iteration')
ax12.set_ylabel('mu')
ax12.grid(True)
plt.legend()

ax13 = fig.add_subplot(4,4,15)
ax13.plot(mus4_5g[0, :], label='KF+CV', marker='', color='green')
ax13.plot(mus4_5g[1, :], label='EKF+CT', marker='', color='orange')
ax13.plot(mus4_5g[2, :], label='KF+CA', marker='', color='red')
ax13.plot(mus4_5g[3, :], label='EKF+CTv', marker='', color='purple')
ax13.set_title("mu[4](5G)")
ax13.set_xlabel('iteration')
ax13.set_ylabel('mu')
ax13.grid(True)
plt.legend()

ax14 = fig.add_subplot(4,4,16)
ax14.plot(mus4_8g[0, :], label='KF+CV', marker='', color='green')
ax14.plot(mus4_8g[1, :], label='EKF+CT', marker='', color='orange')
ax14.plot(mus4_8g[2, :], label='KF+CA', marker='', color='red')
ax14.plot(mus4_8g[3, :], label='EKF+CTv', marker='', color='purple')
ax14.set_title("mu[4](8G)")
ax14.set_xlabel('iteration')
ax14.set_ylabel('mu')
ax14.grid(True)
plt.legend()

plt.show()
