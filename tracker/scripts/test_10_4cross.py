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

import random

T = 6.
process_var = 1
process_var_w = 0.001
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

#tracker_kf = e.BindTracker_10_KF_CV_XX()
tracker_imm = e.BindTracker_10_IMM3_XX()

def step(tracker, Z1, Z2, Z3, Z4):
    E1 = np.zeros((X1.shape[0]+1,X1.shape[1]-1))
    E2 = np.zeros((X2.shape[0]+1,X2.shape[1]-1))
    E3 = np.zeros((X3.shape[0]+1,X3.shape[1]-1))
    E4 = np.zeros((X4.shape[0]+1,X4.shape[1]-1))

    Pd = 0.95 #probability of detection
    #print("Pd: "+str(Pd))
    common_est_am = Z1.shape[1]-1 #estimation amount (100%)
    #print("common_est_am: "+str(common_est_am))
    true_detection_est_am = int(common_est_am*Pd) #estimation amount (95%)
    #print("true_detection_est_am: "+str(true_detection_est_am))
    false_alarm_est_am = common_est_am - true_detection_est_am #pass amount (5%)
    #print("false_alarm_est_am: "+str(false_alarm_est_am))

    rands1 = []
    rands2 = []
    rands3 = []
    rands4 = []

    for r in range(false_alarm_est_am):
        rands1.append(random.randint(0,common_est_am))
        rands2.append(random.randint(0,common_est_am))
        rands3.append(random.randint(0,common_est_am))
        rands4.append(random.randint(0,common_est_am))

    for i in range(common_est_am):
        zss = []
        am=0
        if not i in rands1:
            zss.append(Z1[:,i])
        if not i in rands2:
            zss.append(Z2[:,i])
        if not i in rands3:
            zss.append(Z3[:,i])
        if not i in rands4:
            zss.append(Z4[:,i])
        zs = np.array(zss)

        ests = tracker.step(zs,T)

        E1[:,i] = ests[0,:]
        E2[:,i] = ests[1,:]
        E3[:,i] = ests[2,:]
        E4[:,i] = ests[3,:]

    return E1, E2, E3, E4

#[EstKF1, EstKF2, EstKF3, EstKF4] = step(tracker_kf,Zn1,Zn2,Zn3,Zn4)
[EstIMM1, EstIMM2, EstIMM3, EstIMM4] = step(tracker_imm,Zn1,Zn2,Zn3,Zn4)

##############################################################################################################
def calc_err(x1,x2,x3,x4,Q,R,amount):
    xn1 = stand.add_process_noise(x1,Q)
    xn2 = stand.add_process_noise(x2,Q)
    xn3 = stand.add_process_noise(x3,Q)
    xn4 = stand.add_process_noise(x4,Q)
    zn1 = stand.make_meas(xn1,R,e.BindHXX_10)
    zn2 = stand.make_meas(xn2,R,e.BindHXX_10)
    zn3 = stand.make_meas(xn3,R,e.BindHXX_10)
    zn4 = stand.make_meas(xn4,R,e.BindHXX_10)

    tracker = e.BindTracker_10_IMM3_XX()

    [est1,est2,est3,est4] = step(tracker,zn1,zn2,zn3,zn4)

    err1 = est1[:-1,] - xn1[:, 1:]
    err2 = est2[:-1,] - xn2[:, 1:]
    err3 = est3[:-1,] - xn3[:, 1:]
    err4 = est4[:-1,] - xn4[:, 1:]

    return err1, err2, err3, err4

from tqdm import tqdm

def calc_std_err(x1,x2,x3,x4,Q,R,amount,num_iterations):
    var_err1 = np.zeros((x1.shape[0], x1.shape[1]-1))
    var_err2 = np.zeros((x2.shape[0], x2.shape[1]-1))
    var_err3 = np.zeros((x3.shape[0], x3.shape[1]-1))
    var_err4 = np.zeros((x4.shape[0], x4.shape[1]-1))

    for i in tqdm(range(num_iterations)):
        [err1, err2, err3, err4] = calc_err(x1,x2,x3,x4,Q,R,amount)
        var_err1 += err1 ** 2
        var_err2 += err2 ** 2
        var_err3 += err3 ** 2
        var_err4 += err4 ** 2

    var_err1 /= num_iterations
    var_err2 /= num_iterations
    var_err3 /= num_iterations
    var_err4 /= num_iterations
    return np.sqrt(var_err1), np.sqrt(var_err2), np.sqrt(var_err3), np.sqrt(var_err4)

[std_err_1, std_err_2, std_err_3, std_err_4] = calc_std_err(X1,X2,X3,X4,Q,Rp,100,20)
#####################################################################################
fig = plt.figure("",figsize=(21,11))

def xo_points(estimates):
    estimates_x = estimates.copy()
    estimates_o = estimates.copy()
    for i in range(estimates.shape[1]):
        if(estimates[-1,i]==1):
            estimates_x[:,i] = np.nan
        else:
            estimates_o[:,i] = np.nan
    return estimates_x, estimates_o

[EstIMM1_x, EstIMM1_o] = xo_points(EstIMM1)
[EstIMM2_x, EstIMM2_o] = xo_points(EstIMM2)
[EstIMM3_x, EstIMM3_o] = xo_points(EstIMM3)
[EstIMM4_x, EstIMM4_o] = xo_points(EstIMM4)

ax1 = fig.add_subplot(2,2,1)
ax1.plot(Xn1[0, :], Xn1[3, :], label='true-1', linestyle='--', marker='', color='red')
ax1.plot(Xn2[0, :], Xn2[3, :], label='true-2', linestyle='--', marker='', color='blue')
ax1.plot(Xn3[0, :], Xn3[3, :], label='true-3', linestyle='--', marker='', color='green')
ax1.plot(Xn4[0, :], Xn4[3, :], label='true-4', linestyle='--', marker='', color='orange')
ax1.plot(Zn1[0, :], Zn1[1, :], label='measurement-1', linestyle='', marker='+', color='red')
ax1.plot(Zn2[0, :], Zn2[1, :], label='measurement-2', linestyle='', marker='+', color='blue')
ax1.plot(Zn3[0, :], Zn3[1, :], label='measurement-3', linestyle='', marker='+', color='green')
ax1.plot(Zn4[0, :], Zn4[1, :], label='measurement-4', linestyle='', marker='+', color='orange')
ax1.plot(EstIMM1[0, 1:], EstIMM1[3, 1:], label='estimation(imm)-1', marker='', color='tomato')
ax1.plot(EstIMM1_x[0, 1:], EstIMM1_x[3, 1:], linestyle='', marker='X', color='red')
ax1.plot(EstIMM1_o[0, 1:], EstIMM1_o[3, 1:], linestyle='', marker='.', color='tomato')
ax1.plot(EstIMM2[0, 1:], EstIMM2[3, 1:], label='estimation(imm)-2', marker='', color='lightblue')
ax1.plot(EstIMM2_x[0, 1:], EstIMM2_x[3, 1:], linestyle='', marker='X', color='red')
ax1.plot(EstIMM2_o[0, 1:], EstIMM2_o[3, 1:], linestyle='', marker='.', color='lightblue')
ax1.plot(EstIMM3[0, 1:], EstIMM3[3, 1:], label='estimation(imm)-3', marker='', color='lime')
ax1.plot(EstIMM3_x[0, 1:], EstIMM3_x[3, 1:], linestyle='', marker='X', color='red')
ax1.plot(EstIMM3_o[0, 1:], EstIMM3_o[3, 1:], linestyle='', marker='.', color='lime')
ax1.plot(EstIMM4[0, 1:], EstIMM4[3, 1:], label='estimation(imm)-4', marker='', color='yellow')
ax1.plot(EstIMM4_x[0, 1:], EstIMM4_x[3, 1:], linestyle='', marker='X', color='red')
ax1.plot(EstIMM4_o[0, 1:], EstIMM4_o[3, 1:], linestyle='', marker='.', color='yellow')
ax1.set_title("IMM - Y(X)")
ax1.set_xlabel('x,met')
ax1.set_ylabel('y,met')
ax1.grid(True)
plt.legend()

#ax2 = fig.add_subplot(2,2,3)
#ax2.plot(X1[0, :], X1[3, :], label='true-1', marker='', color='grey')
#ax2.plot(X2[0, :], X2[3, :], label='true-2', marker='', color='grey')
#ax2.plot(X3[0, :], X3[3, :], label='true-3', marker='', color='grey')
#ax2.plot(X4[0, :], X4[3, :], label='true-4', marker='', color='grey')
#ax2.plot(Zn1[0, :], Zn1[1, :], label='measurement-1', linestyle='', marker='+', color='red')
#ax2.plot(Zn2[0, :], Zn2[1, :], label='measurement-2', linestyle='', marker='+', color='blue')
#ax2.plot(Zn3[0, :], Zn3[1, :], label='measurement-3', linestyle='', marker='+', color='green')
#ax2.plot(Zn4[0, :], Zn4[1, :], label='measurement-4', linestyle='', marker='+', color='orange')
#ax2.plot(EstKF1[0, 1:], EstKF1[1, 1:], label='estimation(kf)-1', marker='', color='tomato')
#ax2.plot(EstKF2[0, 1:], EstKF2[1, 1:], label='estimation(kf)-2', marker='', color='lightblue')
#ax2.plot(EstKF3[0, 1:], EstKF3[1, 1:], label='estimation(kf)-3', marker='', color='lime')
#ax2.plot(EstKF4[0, 1:], EstKF4[1, 1:], label='estimation(kf)-4', marker='', color='yellow')
#ax2.set_title("KF - Y(X)")
#ax2.set_xlabel('x,met')
#ax2.set_ylabel('y,met')
#ax2.grid(True)
#plt.legend()

ax0x = fig.add_subplot(4,2,2)
ax0x.plot((np.arange(len(std_err_1[0, :]))+1)*T, std_err_1[0, :].T,color='red')
ax0x.plot((np.arange(len(std_err_2[0, :]))+1)*T, std_err_2[0, :].T,color='blue')
ax0x.plot((np.arange(len(std_err_3[0, :]))+1)*T, std_err_3[0, :].T,color='green')
ax0x.plot((np.arange(len(std_err_4[0, :]))+1)*T, std_err_4[0, :].T,color='orange')
ax0x.grid(True)
ax0x.set_xlabel('Time,s')
ax0x.set_ylabel('std_x, met')

ax0vx = fig.add_subplot(4,2,4)
ax0vx.plot((np.arange(len(std_err_1[1, :]))+1)*T, std_err_1[1, :].T,color='red')
ax0vx.plot((np.arange(len(std_err_2[1, :]))+1)*T, std_err_2[1, :].T,color='blue')
ax0vx.plot((np.arange(len(std_err_3[1, :]))+1)*T, std_err_3[1, :].T,color='green')
ax0vx.plot((np.arange(len(std_err_4[1, :]))+1)*T, std_err_4[1, :].T,color='orange')
ax0vx.grid(True)
ax0vx.set_xlabel('Time,s')
ax0vx.set_ylabel('std_vx, met')

ax0y = fig.add_subplot(4,2,6)
ax0y.plot((np.arange(len(std_err_1[3, :]))+1)*T, std_err_1[3, :].T,color='red')
ax0y.plot((np.arange(len(std_err_2[3, :]))+1)*T, std_err_2[3, :].T,color='blue')
ax0y.plot((np.arange(len(std_err_3[3, :]))+1)*T, std_err_3[3, :].T,color='green')
ax0y.plot((np.arange(len(std_err_4[3, :]))+1)*T, std_err_4[3, :].T,color='orange')
ax0y.grid(True)
ax0y.set_xlabel('Time,s')
ax0y.set_ylabel('std_y, met')

ax0vy = fig.add_subplot(4,2,8)
ax0vy.plot((np.arange(len(std_err_1[4, :]))+1)*T, std_err_1[4, :].T,color='red')
ax0vy.plot((np.arange(len(std_err_2[4, :]))+1)*T, std_err_2[4, :].T,color='blue')
ax0vy.plot((np.arange(len(std_err_3[4, :]))+1)*T, std_err_3[4, :].T,color='green')
ax0vy.plot((np.arange(len(std_err_4[4, :]))+1)*T, std_err_4[4, :].T,color='orange')
ax0vy.grid(True)
ax0vy.set_xlabel('Time,s')
ax0vy.set_ylabel('std_vy, met')

ax0z = fig.add_subplot(4,2,5)
ax0z.plot((np.arange(len(std_err_1[6, :]))+1)*T, std_err_1[6, :].T,color='red')
ax0z.plot((np.arange(len(std_err_2[6, :]))+1)*T, std_err_2[6, :].T,color='blue')
ax0z.plot((np.arange(len(std_err_3[6, :]))+1)*T, std_err_3[6, :].T,color='green')
ax0z.plot((np.arange(len(std_err_4[6, :]))+1)*T, std_err_4[6, :].T,color='orange')
ax0z.grid(True)
ax0z.set_xlabel('Time,s')
ax0z.set_ylabel('std_z, met')

ax0vz = fig.add_subplot(4,2,7)
ax0vz.plot((np.arange(len(std_err_1[7, :]))+1)*T, std_err_1[7, :].T,color='red')
ax0vz.plot((np.arange(len(std_err_2[7, :]))+1)*T, std_err_2[7, :].T,color='blue')
ax0vz.plot((np.arange(len(std_err_3[7, :]))+1)*T, std_err_3[7, :].T,color='green')
ax0vz.plot((np.arange(len(std_err_4[7, :]))+1)*T, std_err_4[7, :].T,color='orange')
ax0vz.grid(True)
ax0vz.set_xlabel('Time,s')
ax0vz.set_ylabel('std_vz, met')

plt.show()
