#!/usr/bin/python3
from filterpy.kalman import KalmanFilter
from filterpy.kalman import ExtendedKalmanFilter
from filterpy.kalman import IMMEstimator
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


kf1 = KalmanFilter(4, 2)
kf1.x = x0
kf1.P = P0
kf1.H = H
kf1.F = F
kf1.R = R
kf1.Q = Q

kf2 = KalmanFilter(4, 2)
kf2.x = x0
kf2.P = P0
kf2.H = H
kf2.F = F
kf2.R = R
kf2.Q *= 0   # no prediction error in second filter

filters = [kf1, kf2]
mu = [0.5, 0.5]  # each filter is equally likely at the start
trans = np.array([[0.97, 0.03], [0.03, 0.97]])
imm = IMMEstimator(filters, mu, trans)

imm.predict()
print("pred.imm.x:")
print(imm.x)
print("pred.imm.P:")
print(imm.P)

imm.update(z)
print("corr.imm.x:")
print(imm.x)
print("corr.imm.P:")
print(imm.P)

print("imm.mu:")
print(imm.mu)

print("imm.M:")
print(imm.M)

