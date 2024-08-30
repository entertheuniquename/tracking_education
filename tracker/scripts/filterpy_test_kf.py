#!/usr/bin/python3
from filterpy.kalman import KalmanFilter
from numpy import array
import numpy as np
import math

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

kf = KalmanFilter(dim_x=4, dim_z=2)

kf.x = x0
kf.P = P0
kf.H = H
kf.F = F
kf.R = R
kf.Q = Q

print("kf.x:")
print(kf.x)
print("kf.P:")
print(kf.P)
print("kf.H:")
print(kf.H)
print("kf.F:")
print(kf.F)
print("kf.R:")
print(kf.R)
print("kf.Q:")
print(kf.Q)

kf.predict()

print("pred.kf.x:")
print(kf.x)
print("pred.kf.P:")
print(kf.P)

kf.update(z)

print("corr.kf.x:")
print(kf.x)
print("corr.kf.P:")
print(kf.P)
