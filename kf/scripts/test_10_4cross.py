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

T = 6.
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

tracker = e.BindTracker_10_KF_CV_XX()

E1 = np.zeros((Zn1.shape[0],Zn1.shape[1]-1))
E2 = np.zeros((Zn2.shape[0],Zn2.shape[1]-1))
E3 = np.zeros((Zn3.shape[0],Zn3.shape[1]-1))
E4 = np.zeros((Zn4.shape[0],Zn4.shape[1]-1))

for i in range(Zn1.shape[1]-1):
    zs = np.array([Zn1[:,i],Zn2[:,i],Zn3[:,i],Zn4[:,i]])

    ests = tracker.step(zs,T)
    E1[:,i] = ests[0,:]
    E2[:,i] = ests[1,:]
    E3[:,i] = ests[2,:]
    E4[:,i] = ests[3,:]

fig = plt.figure("",figsize=(21,11))

ax1 = fig.add_subplot(1,1,1)
#ax1.plot(X1[0, :], X1[3, :], label='true-1', marker='', color='red')
#ax1.plot(X2[0, :], X2[3, :], label='true-2', marker='', color='blue')
#ax1.plot(X3[0, :], X3[3, :], label='true-3', marker='', color='green')
#ax1.plot(X4[0, :], X4[3, :], label='true-4', marker='', color='orange')
ax1.plot(Zn1[0, :], Zn1[1, :], label='measurement-1', linestyle='', marker='+', color='red')
ax1.plot(Zn2[0, :], Zn2[1, :], label='measurement-2', linestyle='', marker='+', color='blue')
ax1.plot(Zn3[0, :], Zn3[1, :], label='measurement-3', linestyle='', marker='+', color='green')
ax1.plot(Zn4[0, :], Zn4[1, :], label='measurement-4', linestyle='', marker='+', color='orange')
ax1.plot(E1[0, 1:], E1[1, 1:], label='estimation-1', marker='', color='tomato')
ax1.plot(E2[0, 1:], E2[1, 1:], label='estimation-2', marker='', color='lightblue')
ax1.plot(E3[0, 1:], E3[1, 1:], label='estimation-3', marker='', color='lime')
ax1.plot(E4[0, 1:], E4[1, 1:], label='estimation-4', marker='', color='yellow')
ax1.set_title("Y(X)")
ax1.set_xlabel('x,met')
ax1.set_ylabel('y,met')
ax1.grid(True)
plt.legend()

plt.show()
