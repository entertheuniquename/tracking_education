#!/usr/bin/python3
from filterpy.kalman import ExtendedKalmanFilter
from numpy import array
import numpy as np
import math
import estimator as e

T = 0.2

Q0 = np.array([[1,0,0],
               [0,1,0],
               [0,0,1]])
G = np.array([[T**2/2,           0,0],
              [     T,           0,0],
              [     0,      T**2/2,0],
              [     0,           T,0],
              [     0,           0,1]])

F = np.array([[1,T,0,0],
              [0,1,0,0],
              [0,0,1,T],
              [0,0,0,1]])



def Fj(x,t):
    w = x[4]
    vx = x[1]
    vy = x[3]
    return np.array([[1., math.sin(w*t)/w     , (math.cos(w*t)-1.)/w, 0., (t*vx*math.cos(w*t)/w) - (t*vy*math.sin(w*t)/w) - (vx*math.sin(w*t)/w**2) - (vy*(math.cos(w*t)-1.)/w**2)],
                     [0., math.cos(w*t)       , -math.sin(w*t)      , 0., -t*vx*math.sin(w*t) - t*vy*math.cos(w*t)                                                               ],
                     [0., (1.-math.cos(w*t))/w, math.sin(w*t)/w     , t , (t*vx*math.sin(w*t)/w) + (t*vy*math.cos(w*t)/w) - (vx*(1.-math.cos(w*t))/w**2) - (vy*math.sin(w*t)/w**2)],
                     [0., math.sin(w*t)       , math.cos(w*t)       , 1., t*vx*math.cos(w*t) - t*vy*math.sin(w*t)                                                                ],
                     [0., 0.                  , 0.                  , 0., 1.                                                                                                     ]])

def Fx(x,t):
    w = x[4]
    return np.array([[1., math.sin(w*t)/w     , 0., (math.cos(w*t)-1.)/w, 0.],
                     [0., math.cos(w*t)       , 0., -math.sin(w*t)      , 0.],
                     [0., (1.-math.cos(w*t))/w, 1., math.sin(w*t)/w     , 0.],
                     [0., math.sin(w*t)       , 0., math.cos(w*t)       , 0.],
                     [0., 0.                  , 0., 0.                  , 1.]])

def Hj():
    return H

def Hj(x):
    return H

def Hx(x):
    return np.dot(H,x)

Q = G@Q0@G.T

# Матрица ошибок измерения
R = np.array([[1,0],
              [0,1]])

H = np.array([[1, 0, 0, 0, 0],
              [0, 0, 1, 0, 0]])

P0  = np.array([[1,0,0,0,0],
                [0,1,0,0,0],
                [0,0,1,0,0],
                [0,0,0,1,0],
                [0,0,0,0,1]])

z1 = np.array([10,20])
z2 = np.array([0,40])
z3 = np.array([-20,50])

x0 = np.array([1,2,3,4,0.0001])

ekf = ExtendedKalmanFilter(dim_x=5, dim_z=2)

ekf.x = x0
ekf.P = P0
ekf.H = H
ekf.R = R
ekf.Q = Q

print("== step 1 ==")
ekf.F = Fx(ekf.x,T)
ekf.predict_x()

ekf.F = Fj(ekf.x,T)
ekf.predict()

print("[1] pred.ekf.x:")
print(ekf.x)
print("[1] pred.ekf.P:")
print(ekf.P)
print("[1] pred.ekf.S:")
print(ekf.S)

ekf.update(z1,Hj,Hx)

print("[1] corr.ekf.x:")
print(ekf.x)
print("[1] corr.ekf.P:")
print(ekf.P)
print("[1] corr.ekf.S:")
print(ekf.S)

#step 2
ekf.F = Fx(ekf.x,T)
ekf.predict_x()

ekf.F = Fj(ekf.x,T)
ekf.predict()

print("[2] pred.ekf.x:")
print(ekf.x)
print("[2] pred.ekf.P:")
print(ekf.P)
print("[2] pred.ekf.S:")
print(ekf.S)

ekf.update(z2,Hj,Hx)

print("[2] corr.ekf.x:")
print(ekf.x)
print("[2] corr.ekf.P:")
print(ekf.P)
print("[2] corr.ekf.S:")
print(ekf.S)

#step 3
ekf.F = Fx(ekf.x,T)
ekf.predict_x()

ekf.F = Fj(ekf.x,T)
ekf.predict()

print("[3] pred.ekf.x:")
print(ekf.x)
print("[3] pred.ekf.P:")
print(ekf.P)
print("[3] pred.ekf.S:")
print(ekf.S)

ekf.update(z3,Hj,Hx)

print("[3] corr.ekf.x:")
print(ekf.x)
print("[3] corr.ekf.P:")
print(ekf.P)
print("[3] corr.ekf.S:")
print(ekf.S)
