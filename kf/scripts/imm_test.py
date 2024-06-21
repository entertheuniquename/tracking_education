#!/usr/bin/python3

import math
import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.kalman import ExtendedKalmanFilter
from filterpy.kalman import IMMEstimator
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
initialState = np.array([30000., -200., 0., 0., 0., 0., 0.098])#radian

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

kf1 = KalmanFilter(dim_x=7,dim_z=3)

kf1.x = initialState
kf1.F = np.array([[1.,T ,0.,0.,0.,0.,0.],
                  [0.,1.,0.,0.,0.,0.,0.],
                  [0.,0.,1.,T ,0.,0.,0.],
                  [0.,0.,0.,1.,0.,0.,0.],
                  [0.,0.,0.,0.,1.,T ,0.],
                  [0.,0.,0.,0.,0.,1.,0.],
                  [0.,0.,0.,0.,0.,0.,1.]])
kf1.R = R
kf1.Q = Q
kf1.P = P0

print("kf1 - DONE!")

kf2 = KalmanFilter(dim_x=7,dim_z=3)

kf2.x = initialState
kf2.F = np.array([[1.,T ,0.,0.,0.,0.,0.],
                  [0.,1.,0.,0.,0.,0.,0.],
                  [0.,0.,1.,T ,0.,0.,0.],
                  [0.,0.,0.,1.,0.,0.,0.],
                  [0.,0.,0.,0.,1.,T ,0.],
                  [0.,0.,0.,0.,0.,1.,0.],
                  [0.,0.,0.,0.,0.,0.,1.]])
kf2.R = R
kf2.Q = Q
kf2.P = P0

print("kf2 - DONE!")

ekf = ExtendedKalmanFilter(dim_x=7,dim_z=3)

ekf.x = initialState
ekf.F = np.array([[1.,T ,0.,0.,0.,0.,0.],
                  [0.,1.,0.,0.,0.,0.,0.],
                  [0.,0.,1.,T ,0.,0.,0.],
                  [0.,0.,0.,1.,0.,0.,0.],
                  [0.,0.,0.,0.,1.,T ,0.],
                  [0.,0.,0.,0.,0.,1.,0.],
                  [0.,0.,0.,0.,0.,0.,1.]])
ekf.R = R
ekf.Q = Q
ekf.P = P0
ekf.B = 0

print("ekf.x:")
print(ekf.x)
print("ekf.F:")
print(ekf.F)

print("ekf - DONE!")

filters = [kf1, kf2]
mu = [0.5, 0.5]
trans = np.array([[0.97, 0.03], [0.03, 0.97]])

imm = IMMEstimator(filters, mu, trans)

print("imm - DONE!")

est = []
print("[start]x:")
print(imm.x)
for i in range(Zn.shape[1]-1):
    print("step: "+str(i))
    imm.predict(0)

    Z1 = Zn.transpose()
    z = Z1[i+1,:]

    imm.update(z)
    #print("z:")
    #print(z)
    #print("x:")
    #print(imm.x)
    est.append(imm.x)

print("[finish]x:")
print(imm.x)

E = np.asarray(est)

print("E:")
print(E)

print("estimation - DONE!")

fig = plt.figure(figsize=(18,25))

ax1 = fig.add_subplot(1,1,1)
ax1.plot(X[0,:], X[2,:], label='true', marker='', color='black')
ax1.plot(Zn[0,:], Zn[1,:], label='measurement', marker='x', color='grey')
#ax1.plot(E[0,:], E[2,:], label='estimation', marker='', color='red')
ax1.set_title("x(y)")
plt.legend()
ax1.set_xlabel('x.m')
ax1.set_ylabel('y.m')
ax1.grid(True)

plt.show()
