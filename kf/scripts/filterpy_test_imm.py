#!/usr/bin/python3
from filterpy.kalman import KalmanFilter
from filterpy.kalman import ExtendedKalmanFilter
from filterpy.kalman import IMMEstimator
from numpy import array
import numpy as np
import math
import estimator as e

T = 0.2

Q01 = np.array([[1.,0.],
                [0.,1.]])

Q02 = np.array([[0.,0.],
                [0.,0.]])

Q03 = np.array([[10.,0.],
                [0.,10.]])

G = np.array([[T**2/2.,           0.],
              [     T ,           0.],
              [     0.,      T**2/2.],
              [     0.,           T ]])

F = np.array([[1.,T ,0.,0.],
              [0.,1.,0.,0.],
              [0.,0.,1.,T ],
              [0.,0.,0.,1.]])

Q1 = G@Q01@G.T
Q2 = G@Q02@G.T
Q3 = G@Q03@G.T

# Матрица ошибок измерения
R = np.array([[1.,0.],
              [0.,1.]])

H = np.array([[1., 0., 0., 0.],
              [0., 0., 1., 0.]])

P0  = np.array([[1.,0.,0.,0.],
                [0.,1.,0.,0.],
                [0.,0.,1.,0.],
                [0.,0.,0.,1.]])

x0 = np.array([1.,2.,3.,4.])
z = np.array([10.,20.])


kf1 = KalmanFilter(4, 2)
kf1.x = x0
kf1.P = P0
kf1.H = H
kf1.F = F
kf1.R = R
kf1.Q = Q1

kf2 = KalmanFilter(4, 2)
kf2.x = x0
kf2.P = P0
kf2.H = H
kf2.F = F
kf2.R = R
kf2.Q = Q2   # no prediction error in second filter

kf3 = KalmanFilter(4, 2)
kf3.x = x0
kf3.P = P0
kf3.H = H
kf3.F = F
kf3.R = R
kf3.Q = Q3

filters = [kf1, kf2, kf3]
mu = [0.3333, 0.3333, 0.3333]  # each filter is equally likely at the start
trans = np.array([[0.95, 0.025, 0.025], [0.025, 0.95, 0.025], [0.025, 0.025, 0.95]])
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

print("imm.likelihood:")
print(imm.likelihood)

print("kf1.Se:")
print(kf1.S)
print("kf2.Se:")
print(kf2.S)
print("kf3.Se:")
print(kf3.S)

print("cbar:")
print(imm.cbar)

print("imm.mu:")
print(imm.mu)

print("imm.omega:")
print(imm.omega)

print("imm.M:")
print(imm.M)

