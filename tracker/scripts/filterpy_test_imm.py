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
mu = [0.33333333, 0.33333333, 0.33333333]  # each filter is equally likely at the start
trans = np.array([[0.95, 0.025, 0.025],
                  [0.025, 0.95, 0.025],
                  [0.025, 0.025, 0.95]])
imm = IMMEstimator(filters, mu, trans)

print(" input data ----- ----- ----- -----")
print("imm.x0:")
print(imm.x)
print("imm.P0:")
print(imm.P)
print("kf1.x0:")
print(kf1.x)
print("kf1.Q:")
print(kf1.Q)
print("kf1.R:")
print(kf1.R)
print("kf1.P0:")
print(kf1.P)
print("kf2.x0:")
print(kf2.x)
print("kf2.P0:")
print(kf2.P)
print("kf2.Q:")
print(kf2.Q)
print("kf2.R:")
print(kf2.R)
print("kf3.x0:")
print(kf3.x)
print("kf3.P0:")
print(kf3.P)
print("kf3.Q:")
print(kf3.Q)
print("kf3.R:")
print(kf3.R)
print("imm.mu:")
print(imm.mu)
print("imm.M:")
print(imm.M)
print(" predict data ----- ----- ----- -----")
imm.predict()
print("pred.imm.x:")
print(imm.x)
print("pred.imm.P:")
print(imm.P)
print("kf1.x:")
print(kf1.x)
print("kf1.P:")
print(kf1.P)
print("kf2.x:")
print(kf2.x)
print("kf2.P:")
print(kf2.P)
print("kf3.x:")
print(kf3.x)
print("kf3.P:")
print(kf3.P)
print(" correct data ----- ----- ----- -----")
imm.update(z)
print("corr.imm.x:")
print(imm.x)
print("corr.imm.P:")
print(imm.P)
print("kf1.x:")
print(kf1.x)
print("kf1.P:")
print(kf1.P)
print("kf2.x:")
print(kf2.x)
print("kf2.P:")
print(kf2.P)
print("kf3.x:")
print(kf3.x)
print("kf3.P:")
print(kf3.P)
print(" result data ----- ----- ----- -----")
print("imm.likelihood:")
print(imm.likelihood)
print(" ----- ----- ----- ----- -----")
print("kf1.Se:")
print(kf1.S)
print("kf2.Se:")
print(kf2.S)
print("kf3.Se:")
print(kf3.S)
print(" ----- ----- ----- ----- -----")
print("cbar:")
print(imm.cbar)
print(" ----- ----- ----- ----- -----")
print("imm.mu:")
print(imm.mu)
print(" ----- ----- ----- ----- -----")
print("imm.omega:")
print(imm.omega)
print(" ----- ----- ----- ----- -----")
print("imm.M:")
print(imm.M)

#############################################################################
print("#################################################################################")
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
kf2.Q = Q1

kf3 = KalmanFilter(4, 2)
kf3.x = x0
kf3.P = P0
kf3.H = H
kf3.F = F
kf3.R = R
kf3.Q = Q1

filters = [kf1, kf2, kf3]
mu = [0.33333333, 0.33333333, 0.33333333]  # each filter is equally likely at the start
trans = np.array([[0.95, 0.025, 0.025],
                  [0.025, 0.95, 0.025],
                  [0.025, 0.025, 0.95]])
imm = IMMEstimator(filters, mu, trans)

print(" input data ----- ----- ----- -----")
print("imm.x0:")
print(imm.x)
print("imm.P0:")
print(imm.P)
print("kf1.x0:")
print(kf1.x)
print("kf1.Q:")
print(kf1.Q)
print("kf1.R:")
print(kf1.R)
print("kf1.P0:")
print(kf1.P)
print("kf2.x0:")
print(kf2.x)
print("kf2.P0:")
print(kf2.P)
print("kf2.Q:")
print(kf2.Q)
print("kf2.R:")
print(kf2.R)
print("kf3.x0:")
print(kf3.x)
print("kf3.P0:")
print(kf3.P)
print("kf3.Q:")
print(kf3.Q)
print("kf3.R:")
print(kf3.R)
print("imm.mu:")
print(imm.mu)
print("imm.M:")
print(imm.M)
print(" predict data ----- ----- ----- -----")
imm.predict()
print("pred.imm.x:")
print(imm.x)
print("pred.imm.P:")
print(imm.P)
print("kf1.x:")
print(kf1.x)
print("kf1.P:")
print(kf1.P)
print("kf2.x:")
print(kf2.x)
print("kf2.P:")
print(kf2.P)
print("kf3.x:")
print(kf3.x)
print("kf3.P:")
print(kf3.P)
print(" correct data ----- ----- ----- -----")
imm.update(z)
print("corr.imm.x:")
print(imm.x)
print("corr.imm.P:")
print(imm.P)
print("kf1.x:")
print(kf1.x)
print("kf1.P:")
print(kf1.P)
print("kf2.x:")
print(kf2.x)
print("kf2.P:")
print(kf2.P)
print("kf3.x:")
print(kf3.x)
print("kf3.P:")
print(kf3.P)
print(" result data ----- ----- ----- -----")
print("imm.likelihood:")
print(imm.likelihood)
print(" ----- ----- ----- ----- -----")
print("kf1.Se:")
print(kf1.S)
print("kf2.Se:")
print(kf2.S)
print("kf3.Se:")
print(kf3.S)
print(" ----- ----- ----- ----- -----")
print("cbar:")
print(imm.cbar)
print(" ----- ----- ----- ----- -----")
print("imm.mu:")
print(imm.mu)
print(" ----- ----- ----- ----- -----")
print("imm.omega:")
print(imm.omega)
print(" ----- ----- ----- ----- -----")
print("imm.M:")
print(imm.M)
