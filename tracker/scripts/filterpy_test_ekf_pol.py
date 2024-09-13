#!/usr/bin/python3
import filterpy
from filterpy.kalman import KalmanFilter
from filterpy.kalman import ExtendedKalmanFilter
from numpy import array
import numpy as np
import math
import estimator as e

print("filterpy: "+filterpy.__file__)

T = 0.2

Q0 = np.array([[1,0],
               [0,1]])

G = np.array([[T**2/2,           0],
              [     T,           0],
              [     0,      T**2/2],
              [     0,           T]])

F = np.array([[1,T,0,0],
              [0,1,0,0],
              [0,0,1,T],
              [0,0,0,1]])

H =  np.array([[1, 0, 0, 0],
               [0, 0, 1, 0]])

def Fx(x):
    return np.dot(f,x)

def Fj(x):
    return F

def Hx(x):
    return np.dot(H,x)

def Hj(x):
    return H

def Hx_pol(x):
    x_ = x[0];
    y_ = x[2];
    a = math.atan(y_/x_);
    r = math.sqrt(x_**2+y_**2);
    return np.array([r,a])

def Hj_pol(x):
    x_ = x[0];
    y_ = x[2];
    j = np.zeros((2,4))
    j[0,0] = x_/math.sqrt((x_**2)+(y_**2))
    j[0,1] = 0
    j[0,2] = y_/math.sqrt((x_**2)+(y_**2))
    j[0,3] = 0

    j[1,0] = -y_/(x_**2*(1+((y_**2)/(x_**2))))
    j[1,1] = 0
    j[1,2] = 1/(x_*(1+((y_**2)/(x_**2))))
    j[1,3] = 0
    print("j:")
    print(j)
    return j

Q = G@Q0@G.T

# Матрица ошибок измерения
R = np.array([[1,0],
              [0,1]])

H = np.array([[1, 0, 0, 0],
              [0, 0, 1, 0]])

P0  = np.array([[1,0,0,0],
                [0,1,0,0],
                [0,0,1,0],
                [0,0,0,1]])

x0 = np.array([20.,2.,20.,2.])

z1_dec = np.array([25,25])
z1_pol = np.array([35.355,0.785])#45])

z2_dec = np.array([30,20])
z2_pol = np.array([36.056,0.588])#33.69])


#ekf = ExtendedKalmanFilter(dim_x=4, dim_z=2)

print("=== ekf_dec ===")
ekf_dec = ExtendedKalmanFilter(dim_x=4, dim_z=2)
print("--------------- init ----------------")
ekf_dec.x = x0
ekf_dec.P = P0
ekf_dec.F = F
ekf_dec.H = H
ekf_dec.R = R
ekf_dec.Q = Q
print("ekf_dec.x:")
print(ekf_dec.x)
print("ekf_dec.P:")
print(ekf_dec.P)
print("ekf_dec.F:")
print(ekf_dec.F)
print("ekf_dec.H:")
print(ekf_dec.H)
print("ekf_dec.R:")
print(ekf_dec.R)
print("ekf_dec.Q:")
print(ekf_dec.Q)
print("")
print("[step-1]")
print("-------------- predict --------------")
print("[.] pred.ekf_dec.x:")
print(ekf_dec.x)
print("[.] pred.ekf_dec.P:")
print(ekf_dec.P)
print("[.] pred.ekf_dec.F:")
print(ekf_dec.F)
print("[.] pred.ekf_dec.Q:")
print(ekf_dec.Q)
print("[.] pred.ekf_dec.B:")
print(ekf_dec.B)
print("-------------------------------------")
ekf_dec.predict()
print("[1] pred.ekf_dec.x:")
print(ekf_dec.x)
print("[1] pred.ekf_dec.P:")
print(ekf_dec.P)
print("-------------- correct --------------")
print("[.] pred.ekf_dec.x:")
print(ekf_dec.x)
print("[.] pred.ekf_dec.P:")
print(ekf_dec.P)
print("[.] pred.ekf_dec.R:")
print(ekf_dec.R)
print("-------------------------------------")
ekf_dec.update(z1_dec,Hj,Hx)
print("[1] corr.ekf_dec.x:")
print(ekf_dec.x)
print("[1] corr.ekf_dec.P:")
print(ekf_dec.P)
print("[1] corr.ekf_dec.S:")
print(ekf_dec.S)
print("[1] corr.ekf_dec.K:")
print(ekf_dec.K)
print("[1] corr.ekf_dec.z:")
print(ekf_dec.z)
print("")
print("[step-2]")
print("-------------- predict --------------")
ekf_dec.predict()
print("[2] pred.ekf_dec.x:")
print(ekf_dec.x)
print("[2] pred.ekf_dec.P:")
print(ekf_dec.P)
print("-------------- correct --------------")
ekf_dec.update(z2_dec,Hj,Hx)
print("[2] corr.ekf_dec.x:")
print(ekf_dec.x)
print("[2] corr.ekf_dec.P:")
print(ekf_dec.P)
print("[2] corr.ekf_dec.S:")
print(ekf_dec.S)
print("[2] corr.ekf_dec.K:")
print(ekf_dec.K)
print("[2] corr.ekf_dec.z:")
print(ekf_dec.z)

print("")

print("=== ekf_pol ===")
ekf_pol = ExtendedKalmanFilter(dim_x=4, dim_z=2)
print("--------------- init ----------------")
ekf_pol.x = x0
ekf_pol.P = P0
ekf_pol.F = F
ekf_pol.H = H
ekf_pol.R = R
ekf_pol.Q = Q
print("ekf_pol.x:")
print(ekf_pol.x)
print("ekf_pol.P:")
print(ekf_pol.P)
print("ekf_pol.F:")
print(ekf_pol.F)
print("ekf_pol.H:")
print(ekf_pol.H)
print("ekf_pol.R:")
print(ekf_pol.R)
print("ekf_pol.Q:")
print(ekf_pol.Q)
print("")
print("[step-1]")
print("-------------- predict --------------")
ekf_pol.predict()
print("[1] pred.ekf_pol.x:")
print(ekf_pol.x)
print("[1] pred.ekf_pol.P:")
print(ekf_pol.P)
print("-------------- correct --------------")
ekf_pol.update(z1_pol,Hj_pol,Hx_pol)
print("[1] corr.ekf_pol.x:")
print(ekf_pol.x)
print("[1] corr.ekf_pol.P:")
print(ekf_pol.P)
print("[1] corr.ekf_pol.S:")
print(ekf_pol.S)
print("[1] corr.ekf_pol.K:")
print(ekf_pol.K)
print("[1] corr.ekf_pol.z:")
print(ekf_pol.z)
print("")
print("[step-2]")
print("-------------- predict --------------")
ekf_pol.predict()
print("[2] pred.ekf_pol.x:")
print(ekf_pol.x)
print("[2] pred.ekf_pol.P:")
print(ekf_pol.P)
print("-------------- correct --------------")
ekf_pol.update(z2_pol,Hj_pol,Hx_pol)
print("[2] corr.ekf_pol.x:")
print(ekf_pol.x)
print("[2] corr.ekf_pol.P:")
print(ekf_pol.P)
print("[2] corr.ekf_pol.S:")
print(ekf_pol.S)
print("[2] corr.ekf_pol.K:")
print(ekf_pol.K)
print("[2] corr.ekf_pol.z:")
print(ekf_pol.z)

print("")
