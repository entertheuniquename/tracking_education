#!/usr/bin/python3
from filterpy.kalman import ExtendedKalmanFilter
from numpy import array
import numpy as np
import math
import estimator as e

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

def h():
    return H

def hx(x):
    return np.dot(h(),x)

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

x0 = np.array([1,2,3,4])
z = np.array([10,20])

def h(x):
    return H

def hx(x):
    return np.dot(H,x)

def f(x):
    return F

def fx(x):
    return np.dot(F,x)

ekf = ExtendedKalmanFilter(dim_x=4, dim_z=2)

ekf.x = x0
ekf.P = P0
ekf.H = H
ekf.F = F
ekf.R = R
ekf.Q = Q

print("ekf.x:")
print(ekf.x)
print("ekf.P:")
print(ekf.P)
print("ekf.H:")
print(ekf.H)
print("ekf.F:")
print(ekf.F)
print("ekf.R:")
print(ekf.R)
print("ekf.Q:")
print(ekf.Q)

ekf.predict()

print("pred.ekf.x:")
print(ekf.x)
print("pred.ekf.P:")
print(ekf.P)
print("pred.ekf.S:")
print(ekf.S)

ekf.update(z,h,hx)

print("corr.ekf.x:")
print(ekf.x)
print("corr.ekf.P:")
print(ekf.P)
print("corr.ekf.S:")
print(ekf.S)
