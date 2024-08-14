#!/usr/bin/python3

import numpy as np
import numpy.matlib as ml
import matplotlib.pyplot as plt
import math
from IPython.display import display, Math, Latex
from IPython.display import Markdown as md
#from arr2ltx import convert2latex, to_latex

import estimator as e
import math

### Входные данные ###########################################################################################
# Период поступления данных
#CAUSE
##T = 1
T = 6
#~

# Ошибки процесса
#CAUSE
##process_var = 1
##process_var_w = 0.01
process_var = 0.01
process_var_w = 0.000001
#~

# Ошибки измерений
#CAUSE
##meas_std = 100
meas_std = 30
#~

#CAUSE
# Угловая скорость на развороте в радианах
w2g_r_ = 0.098
w5g_r_ = 0.245
w8g_r_ = 0.392

# Угловая скорость на развороте в градусах
w2g_d = w2g_r_*180/math.pi
w5g_d = w5g_r_*180/math.pi
w8g_d = w8g_r_*180/math.pi

w2g_r = w2g_d
w5g_r = w5g_d
w8g_r = w8g_d

# Угловая скорость на развороте в радианах
#w2g_r = 0.098
#w5g_r = 0.245
#w8g_r = 0.392

# Угловая скорость на развороте в градусах
#w2g_d = w2g_r*180/math.pi
#w5g_d = w5g_r*180/math.pi
#w8g_d = w8g_r*180/math.pi
#~

# Матрица ошибок процесса
Q0 = np.diag([process_var, process_var, process_var, process_var_w])
G = np.array([[T**2/2, 0,      0     , 0],
              [T,      0,      0     , 0],
              [0,      T**2/2, 0     , 0],
              [0,      T,      0     , 0],
              [0,      0,      T**2/2, 0],
              [0,      0,      T     , 0],
              [0,      0,      0     , T]])
#CAUSE
Q = Q0
#Q = G@Q0@G.T
#~

# Матрица ошибок измерения
R = np.diag([meas_std*meas_std, meas_std*meas_std, meas_std*meas_std])

#CAUSE
# Векторы входных данных
initialState2gr = np.array([30000., -200., 30000., 0., 10000., 0., w2g_r])#radian
initialState5gr = np.array([30000., -200., 30000., 0., 10000., 0., w5g_r])#radian
initialState8gr = np.array([30000., -200., 30000., 0., 10000., 0., w8g_r])#radian

#initialState2gr = np.array([30000., -200., 0., 0., 0., 0., w2g_r])#radian
#initialState5gr = np.array([30000., -200., 0., 0., 0., 0., w5g_r])#radian
#initialState8gr = np.array([30000., -200., 0., 0., 0., 0., w8g_r])#radian

initialState2gr_rad = np.array([30000., -200., 30000., 0., 10000., 0., w2g_r_])
initialState5gr_rad = np.array([30000., -200., 30000., 0., 10000., 0., w5g_r_])
initialState8gr_rad = np.array([30000., -200., 30000., 0., 10000., 0., w8g_r_])
#~

### Создание наборов данных ##################################################################################

# Функция создания набора данных: по кругу
def make_true_data_round(x0, am, dt):
    # Создание обнулённой матрицы нужного размера
    X = np.zeros((x0.shape[0], am))
    # Запись первого значения
    X[:, 0] = x0.T
    # Цикл создания участка разворота
    for i in range(am-1):
        xx = e.stateModel_CTx(np.copy(X[:, i]),dt) #TUT
        X[:, i+1] = xx.flatten()
    return X

X2g0r=make_true_data_round(initialState2gr, 200, T)
X5g0r=make_true_data_round(initialState5gr, 200, T)
X8g0r=make_true_data_round(initialState8gr, 200, T)

def make_true_data_round_rad(x0, am, dt):
    # Создание обнулённой матрицы нужного размера
    X = np.zeros((x0.shape[0], am))
    # Запись первого значения
    X[:, 0] = x0.T
    # Цикл создания участка разворота
    for i in range(am-1):
        xx = e.stateModel_CTx_rad(np.copy(X[:, i]),dt) #TUT
        X[:, i+1] = xx.flatten()
    return X

X2g0r_rad=make_true_data_round_rad(initialState2gr_rad, 200, T)
X5g0r_rad=make_true_data_round_rad(initialState5gr_rad, 200, T)
X8g0r_rad=make_true_data_round_rad(initialState8gr_rad, 200, T)

### Добавление к наборам данных ошибок процесса ##############################################################
def add_process_noise(X,Var):
    #CAUSE
    ##Xn = X ##+ np.sqrt(Var) @ np.random.normal(loc=0, scale=1.0, size=(X.shape[0], X.shape[1]))
    Xn = X + np.sqrt(Var) @ np.random.normal(loc=0, scale=1.0, size=(X.shape[0], X.shape[1]))
    #~
    return Xn

Xn2g0r = add_process_noise(X2g0r,G@Q0@G.T)
Xn5g0r = add_process_noise(X5g0r,G@Q0@G.T)
Xn8g0r = add_process_noise(X8g0r,G@Q0@G.T)

Xn2g0r_rad = add_process_noise(X2g0r_rad,G@Q0@G.T)
Xn5g0r_rad = add_process_noise(X5g0r_rad,G@Q0@G.T)
Xn8g0r_rad = add_process_noise(X8g0r_rad,G@Q0@G.T)

### Получение из наборов данных измерений и добавление к ним шцмов ###########################################
# Функция получения измерений
def make_meas(X, R):
    # Получение обнуленного набора измерений
    Z = np.zeros((R.shape[0], X.shape[1]))
    # Цикл по заполнению набора измерений зашумлёнными значениями
    for i in range(Z.shape[1]):
        # Получение очередного значения набора данных
        zz = e.measureModel_XwXx(np.copy(X[:, i]))
        Z[:, i] = zz.flatten()
    # Добавления шумов к набору измерений
    Zn = Z + np.sqrt(R) @ np.random.normal(loc=0, scale=math.sqrt(1.0), size=(Z.shape[0], Z.shape[1]))
    return Zn

Zn2g0r = make_meas(Xn2g0r, R)
Zn5g0r = make_meas(Xn5g0r, R)
Zn8g0r = make_meas(Xn8g0r, R)

Zn2g0r_rad = make_meas(Xn2g0r_rad, R)
Zn5g0r_rad = make_meas(Xn5g0r_rad, R)
Zn8g0r_rad = make_meas(Xn8g0r_rad, R)

### Фильтрация XYZ_EKFE_CT ###################################################################################

# Имитация отметки
def make_tu(TP,M):
    tu = (TP,M)
    return tu

# Функция фильтрации набора данных
def step(Zn, dt):
    time = 0
    track = e.BindTrackEKFE_xyz_ct(make_tu(time,Zn[:, 0][:, np.newaxis]))
    track_ekf = e.BindTrackEKF_xyz_ct(make_tu(time,Zn[:, 0][:, np.newaxis]))
    track_ekf_rad = e.BindTrackEKF_xyz_ct_rad(make_tu(time,Zn[:, 0][:, np.newaxis]))

    P0 = np.zeros((1,3))
    c = track.getCov()
    P0[0][0] = np.sqrt(c[0][0])
    P0[0][1] = np.sqrt(c[2][2])
    P0[0][2] = np.sqrt(c[4][4])

    P0_ekf = np.zeros((1,3))
    c_ekf = track_ekf.getCov()
    P0_ekf[0][0] = np.sqrt(c_ekf[0][0])
    P0_ekf[0][1] = np.sqrt(c_ekf[2][2])
    P0_ekf[0][2] = np.sqrt(c_ekf[4][4])

    P0_ekf_rad = np.zeros((1,3))
    c_ekf_rad = track_ekf_rad.getCov()
    P0_ekf_rad[0][0] = np.sqrt(c_ekf_rad[0][0])
    P0_ekf_rad[0][1] = np.sqrt(c_ekf_rad[2][2])
    P0_ekf_rad[0][2] = np.sqrt(c_ekf_rad[4][4])

    est = np.zeros((7, Zn.shape[1]-1))#7 - bad! not dt
    est_ekf = np.zeros((7, Zn.shape[1]-1))
    est_ekf_rad = np.zeros((7, Zn.shape[1]-1))


    est_cov = np.zeros((3, Zn.shape[1]-1))#3 - bad! not dt
    est_cov_ekf = np.zeros((3, Zn.shape[1]-1))
    est_cov_ekf_rad = np.zeros((3, Zn.shape[1]-1))

    for col in range(Zn.shape[1]-1):
        time = time+dt

        ee = track.step(make_tu(time,np.copy(Zn[:, col+1])))
        ee_ekf = track_ekf.step(make_tu(time,np.copy(Zn[:, col+1])))
        ee_ekf_rad = track_ekf_rad.step(make_tu(time,np.copy(Zn[:, col+1])))

        ee_cov = np.zeros((1,3))
        ee_cov[0][0] = np.sqrt(ee[1][0][0])
        ee_cov[0][1] = np.sqrt(ee[1][2][2])
        ee_cov[0][2] = np.sqrt(ee[1][4][4])

        ee_cov_ekf = np.zeros((1,3))
        ee_cov_ekf[0][0] = np.sqrt(ee_ekf[1][0][0])
        ee_cov_ekf[0][1] = np.sqrt(ee_ekf[1][2][2])
        ee_cov_ekf[0][2] = np.sqrt(ee_ekf[1][4][4])

        ee_cov_ekf_rad = np.zeros((1,3))
        ee_cov_ekf_rad[0][0] = np.sqrt(ee_ekf_rad[1][0][0])
        ee_cov_ekf_rad[0][1] = np.sqrt(ee_ekf_rad[1][2][2])
        ee_cov_ekf_rad[0][2] = np.sqrt(ee_ekf_rad[1][4][4])

        est[:, col] = np.squeeze(ee[0][:])
        est_ekf[:, col] = np.squeeze(ee_ekf[0][:])
        est_ekf_rad[:, col] = np.squeeze(ee_ekf_rad[0][:])

        est_cov[:, col] = np.squeeze(ee_cov[0][:])
        est_cov_ekf[:, col] = np.squeeze(ee_cov_ekf[0][:])
        est_cov_ekf_rad[:, col] = np.squeeze(ee_cov_ekf_rad[0][:])
    return est, est_cov, P0, est_ekf, est_cov_ekf, P0_ekf, est_ekf_rad, est_cov_ekf_rad, P0_ekf_rad



[E2g0r, E2g0r_cov, E2g0r_cov0, E2g0r_ekf, E2g0r_cov_ekf, E2g0r_cov0_ekf, E2g0r_ekf_rad, E2g0r_cov_ekf_rad, E2g0r_cov0_ekf_rad] = step(Zn2g0r, T)
[E5g0r, E5g0r_cov, E5g0r_cov0, E5g0r_ekf, E5g0r_cov_ekf, E5g0r_cov0_ekf, E5g0r_ekf_rad, E5g0r_cov_ekf_rad, E5g0r_cov0_ekf_rad] = step(Zn5g0r, T)
[E8g0r, E8g0r_cov, E8g0r_cov0, E8g0r_ekf, E8g0r_cov_ekf, E8g0r_cov0_ekf, E8g0r_ekf_rad, E8g0r_cov_ekf_rad, E8g0r_cov0_ekf_rad] = step(Zn8g0r, T)

### Отрисовка графиков для сглаживания #######################################################################
fig1 = plt.figure(figsize=(18,25))

ax11 = fig1.add_subplot(2,2,2)
ax11.plot(X2g0r[0, :], X2g0r[2, :], label='true', color='black')
#ax11.plot(Xn2g0r[0, :], Xn2g0r[2, :], linestyle="", marker='x', label='true+noize', color='yellow')
ax11.plot(Zn2g0r[0, :], Zn2g0r[1, :], label='measurements', linestyle="", marker='x', color='grey')
ax11.plot(E2g0r[0, :], E2g0r[2, :], label='ekf_eigen3', color='red')
ax11.plot(E2g0r_ekf[0, :], E2g0r_ekf[2, :], label='ekf', color='blue')
ax11.plot(X2g0r_rad[0, :], X2g0r_rad[2, :], label='true(rad)', color='purple')
ax11.plot(Zn2g0r_rad[0, :], Zn2g0r_rad[1, :], label='measurements(rad)', linestyle="", marker='x', color='pink')
ax11.plot(E2g0r_ekf_rad[0, :], E2g0r_ekf_rad[2, :], label='ekf(rad)', color='brown')
ax11.set_title("2G - Y(X)")
ax11.set_xlabel('x,met')
ax11.set_ylabel('y,met')
ax11.grid(True)

ax21 = fig1.add_subplot(4,2,1)
ax21.plot(X2g0r[0, 1:], label='true', color='black')
#ax21.plot(Xn2g0r[0, 1:], label='true+noise')
ax21.plot(Zn2g0r[0, 1:], label='measurements', linestyle='', marker='x', color='grey')
ax21.plot(E2g0r[0, :], label='ekf_eigen3', color='red')
ax21.plot(E2g0r_ekf[0, :], label='ekf', color='blue')
ax21.plot(E2g0r_ekf_rad[0, :], label='ekf(rad)', color='brown')
ax21.set_title("2G - X")
ax21.set_xlabel('step')
ax21.set_ylabel('x,met.')
ax21.grid(True)

ax31 = fig1.add_subplot(4,2,3)
ax31.plot(X2g0r[2, 1:], label='true',color='black')
#ax31.plot(Xn2g0r[2, 1:], label='Xn2gr - true+noise')
ax31.plot(Zn2g0r[1, 1:], label='measurements', linestyle='', marker='x', color='grey')
ax31.plot(E2g0r[2, :], label='ekf_eigen3',color='red')
ax31.plot(E2g0r_ekf[2, :], label='ekf', color='blue')
ax31.plot(E2g0r_ekf_rad[2, :], label='ekf(rad)', color='brown')
ax31.set_title("2G - Y")
ax31.set_xlabel('step')
ax31.set_ylabel('y,met')
ax31.grid(True)

ax41 = fig1.add_subplot(4,2,5)
ax41.plot(X2g0r[4, 1:], label='true',color='black')
#ax41.plot(Xn2g0r[4, 1:], label='Xn2g0r - true+noise')
ax41.plot(Zn2g0r[2, 1:], label='measurements', linestyle='', marker='x',color='grey')
ax41.plot(E2g0r[4, :], label='ekf_eigen3',color='red')
ax41.plot(E2g0r_ekf[4, :], label='ekf', color='blue')
ax41.plot(E2g0r_ekf_rad[4, :], label='ekf(rad)', color='brown')
ax41.set_title("2G - Z")
ax41.set_xlabel('step')
ax41.set_ylabel('z,met')
ax41.grid(True)

ax51 = fig1.add_subplot(4,4,13)
ax51.plot(X2g0r[6, 1:], label='true',color='black')
#ax51.plot(Xn2g0r[6, 1:], label='true+noise')
ax51.plot(E2g0r[6, :], label='ekf_eigen3',color='red')
ax51.plot(E2g0r_ekf[6, :], label='ekf',color='blue')
ax51.set_title("2G - W")
ax51.set_xlabel('step')
ax51.set_ylabel('w,deg.')
ax51.grid(True)

ax510 = fig1.add_subplot(4,4,14)
ax510.plot(X2g0r_rad[6, 1:], label='true',color='black')
#ax510.plot(Xn2g0r[6, 1:], label='true+noise')
ax510.plot(E2g0r_ekf_rad[6, :], label='ekf(rad)', color='brown')
ax510.set_title("2G - W")
ax510.set_xlabel('step')
ax510.set_ylabel('w,deg.')
ax510.grid(True)

ax61 = fig1.add_subplot(4,2,6)
ax61.plot(E2g0r_cov0[0,0], label='ekf_eigen3: X0 - covariations', linestyle='', marker='o')
ax61.plot(E2g0r_cov0[0,1], label='ekf_eigen3: Y0 - covariations', linestyle='', marker='o')
ax61.plot(E2g0r_cov0[0,2], label='ekf_eigen3: Z0 - covariations', linestyle='', marker='o')
ax61.plot(E2g0r_cov0_ekf[0,0], label='ekf: X0 - covariations', linestyle='', marker='o')
ax61.plot(E2g0r_cov0_ekf[0,1], label='ekf: Y0 - covariations', linestyle='', marker='o')
ax61.plot(E2g0r_cov0_ekf[0,2], label='ekf: Z0 - covariations', linestyle='', marker='o')
ax61.plot(E2g0r_cov[0, :], label='ekf_eigen3: X - covariations')
ax61.plot(E2g0r_cov[1, :], label='ekf_eigen3: Y - covariations')
ax61.plot(E2g0r_cov[2, :], label='ekf_eigen3: Z - covariations')
ax61.plot(E2g0r_cov_ekf[0, :], label='ekf: X - covariations')
ax61.plot(E2g0r_cov_ekf[1, :], label='ekf: Y - covariations')
ax61.plot(E2g0r_cov_ekf[2, :], label='ekf: Z - covariations')
ax61.set_title("2G - cov")
ax61.set_xlabel('step')
ax61.set_ylabel('cov')
ax61.grid(True)

ax610 = fig1.add_subplot(4,2,8)
ax610.plot(E8g0r_cov0_ekf_rad[0,0], label='ekf(rad): X0 - covariations', linestyle='', marker='o')
ax610.plot(E8g0r_cov0_ekf_rad[0,1], label='ekf(rad): Y0 - covariations', linestyle='', marker='o')
ax610.plot(E8g0r_cov0_ekf_rad[0,2], label='ekf(rad): Z0 - covariations', linestyle='', marker='o')
ax610.plot(E8g0r_cov_ekf_rad[0, :], label='ekf(rad): X - covariations')
ax610.plot(E8g0r_cov_ekf_rad[1, :], label='ekf(rad): Y - covariations')
ax610.plot(E8g0r_cov_ekf_rad[2, :], label='ekf(rad): Z - covariations')
ax610.set_title("2G - cov")
ax610.set_xlabel('step')
ax610.set_ylabel('cov')
ax610.grid(True)

#plt.show()
##############################################################################################################
fig2 = plt.figure(figsize=(18,25))

ax12 = fig2.add_subplot(2,2,2)
ax12.plot(X5g0r[0, :], X5g0r[2, :], label='true', color='black')
#ax12.plot(Xn5g0r[0, :], Xn5g0r[2, :], linestyle="", marker='x', label='true+noize', color='yellow')
ax12.plot(Zn5g0r[0, :], Zn5g0r[1, :], label='measurements', linestyle="", marker='x', color='grey')
ax12.plot(E5g0r[0, :], E5g0r[2, :], label='ekf_eigen3', color='red')
ax12.plot(E5g0r_ekf[0, :], E5g0r_ekf[2, :], label='ekf', color='blue')
ax12.plot(X5g0r_rad[0, :], X5g0r_rad[2, :], label='true(rad)', color='purple')
ax12.plot(Zn5g0r_rad[0, :], Zn5g0r_rad[1, :], label='measurements(rad)', linestyle="", marker='x', color='pink')
ax12.plot(E5g0r_ekf_rad[0, :], E5g0r_ekf_rad[2, :], label='ekf(rad)', color='brown')
ax12.set_title("5G")
ax12.set_xlabel('x,met.')
ax12.set_ylabel('y,met.')
ax12.grid(True)

ax22 = fig2.add_subplot(4,2,1)
ax22.plot(X5g0r[0, 1:], label='true', color='black')
#ax22.plot(Xn5g0r[0, 1:], label='true+noise')
ax22.plot(Zn5g0r[0, 1:], label='measurements', linestyle='', marker='x', color='grey')
ax22.plot(E5g0r[0, :], label='ekf_eigen3', color='red')
ax22.plot(E5g0r_ekf[0, :], label='ekf', color='blue')
ax22.plot(E5g0r_ekf_rad[0, :], label='ekf(rad)', color='brown')
ax22.set_title("5G - X")
ax22.set_xlabel('step')
ax22.set_ylabel('x,met.')
ax22.grid(True)

ax32 = fig2.add_subplot(4,2,3)
ax32.plot(X5g0r[2, 1:], label='true',color='black')
#ax32.plot(Xn5g0r[2, 1:], label='true+noise')
ax32.plot(Zn5g0r[1, 1:], label='measurements', linestyle='', marker='x', color='grey')
ax32.plot(E5g0r[2, :], label='ekf_eigen3',color='red')
ax32.plot(E5g0r_ekf[2, :], label='ekf', color='blue')
ax32.plot(E5g0r_ekf_rad[2, :], label='ekf(rad)', color='brown')
ax32.set_title("5G - Y")
ax32.set_xlabel('step')
ax32.set_ylabel('y,met')
ax32.grid(True)

ax42 = fig2.add_subplot(4,2,5)
ax42.plot(X5g0r[4, 1:], label='true',color='black')
#ax42.plot(Xn5g0r[4, 1:], label='true+noise')
ax42.plot(Zn5g0r[2, 1:], label='measurements', linestyle='', marker='x',color='grey')
ax42.plot(E5g0r[4, :], label='ekf_eigen3',color='red')
ax42.plot(E5g0r_ekf[4, :], label='ekf', color='blue')
ax42.plot(E5g0r_ekf_rad[4, :], label='ekf(rad)', color='brown')
ax42.set_title("5G - Z")
ax42.set_xlabel('step')
ax42.set_ylabel('z,met')
ax42.grid(True)

ax52 = fig2.add_subplot(4,4,13)
ax52.plot(X5g0r[6, 1:], label='true',color='black')
#ax52.plot(Xn5g0r[6, 1:], label='true+noise')
ax52.plot(E5g0r[6, :], label='ekf_eigen3',color='red')
ax52.plot(E5g0r_ekf[6, :], label='ekf',color='blue')
ax52.set_title("5G - W")
ax52.set_xlabel('step')
ax52.set_ylabel('w,deg.')
ax52.grid(True)

ax520 = fig2.add_subplot(4,4,14)
ax520.plot(X5g0r_rad[6, 1:], label='true',color='black')
#ax520.plot(Xn5g0r[6, 1:], label='true+noise')
ax520.plot(E5g0r_ekf_rad[6, :], label='ekf(rad)', color='brown')
ax520.set_title("5G - W")
ax520.set_xlabel('step')
ax520.set_ylabel('w,deg.')
ax520.grid(True)

ax62 = fig2.add_subplot(4,2,6)
ax62.plot(E5g0r_cov0[0,0], label='ekf_eigen3: X0 - covariations', linestyle='', marker='o')
ax62.plot(E5g0r_cov0[0,1], label='ekf_eigen3: Y0 - covariations', linestyle='', marker='o')
ax62.plot(E5g0r_cov0[0,2], label='ekf_eigen3: Z0 - covariations', linestyle='', marker='o')
ax62.plot(E5g0r_cov0_ekf[0,0], label='ekf: X0 - covariations', linestyle='', marker='o')
ax62.plot(E5g0r_cov0_ekf[0,1], label='ekf: Y0 - covariations', linestyle='', marker='o')
ax62.plot(E5g0r_cov0_ekf[0,2], label='ekf: Z0 - covariations', linestyle='', marker='o')
ax62.plot(E5g0r_cov[0, :], label='ekf_eigen3: X - covariations')
ax62.plot(E5g0r_cov[1, :], label='ekf_eigen3: Y - covariations')
ax62.plot(E5g0r_cov[2, :], label='ekf_eigen3: Z - covariations')
ax62.plot(E5g0r_cov_ekf[0, :], label='ekf: X - covariations')
ax62.plot(E5g0r_cov_ekf[1, :], label='ekf: Y - covariations')
ax62.plot(E5g0r_cov_ekf[2, :], label='ekf: Z - covariations')
ax62.set_title("5G - cov")
ax62.set_xlabel('step')
ax62.set_ylabel('cov')
ax62.grid(True)

ax620 = fig2.add_subplot(4,2,8)
ax620.plot(E5g0r_cov0_ekf_rad[0,0], label='ekf(rad): X0 - covariations', linestyle='', marker='o')
ax620.plot(E5g0r_cov0_ekf_rad[0,1], label='ekf(rad): Y0 - covariations', linestyle='', marker='o')
ax620.plot(E5g0r_cov0_ekf_rad[0,2], label='ekf(rad): Z0 - covariations', linestyle='', marker='o')
ax620.plot(E5g0r_cov_ekf_rad[0, :], label='ekf(rad): X - covariations')
ax620.plot(E5g0r_cov_ekf_rad[1, :], label='ekf(rad): Y - covariations')
ax620.plot(E5g0r_cov_ekf_rad[2, :], label='ekf(rad): Z - covariations')
ax620.set_title("5G - cov")
ax620.set_xlabel('step')
ax620.set_ylabel('cov')
ax620.grid(True)

##plt.show()
###############################################################################################################
fig3 = plt.figure(figsize=(18,25))

ax13 = fig3.add_subplot(2,2,2)
ax13.plot(X8g0r[0, :], X8g0r[2, :], label='true', color='black')
#ax13.plot(Xn8g0r[0, :], Xn8g0r[2, :], linestyle="", marker='x', label='true+noize', color='yellow')
ax13.plot(Zn8g0r[0, :], Zn8g0r[1, :], label='measurements', linestyle="", marker='x', color='grey')
ax13.plot(E8g0r[0, :], E8g0r[2, :], label='ekf_eigen3', color='red')
ax13.plot(E8g0r_ekf[0, :], E8g0r_ekf[2, :], label='ekf', color='blue')
ax13.plot(X8g0r_rad[0, :], X8g0r_rad[2, :], label='true(rad)', color='purple')
ax13.plot(Zn8g0r_rad[0, :], Zn8g0r_rad[1, :], label='measurements(rad)', linestyle="", marker='x', color='pink')
ax13.plot(E8g0r_ekf_rad[0, :], E8g0r_ekf_rad[2, :], label='ekf(rad)', color='brown')
ax13.set_title("8G")
ax13.set_xlabel('x,met.')
ax13.set_ylabel('y,met.')
ax13.grid(True)

ax23 = fig3.add_subplot(4,2,1)
ax23.plot(X8g0r[0, 1:], label='true', color='black')
#ax23.plot(Xn8g0r[0, 1:], label='true+noise')
ax23.plot(Zn8g0r[0, 1:], label='measurements', linestyle='', marker='x', color='grey')
ax23.plot(E8g0r[0, :], label='ekf_eigen3', color='red')
ax23.plot(E8g0r_ekf[0, :], label='ekf', color='blue')
ax23.plot(E8g0r_ekf_rad[0, :], label='ekf(rad)', color='brown')
ax23.set_title("8G - X")
ax23.set_xlabel('step')
ax23.set_ylabel('x,met.')
ax23.grid(True)

ax33 = fig3.add_subplot(4,2,3)
ax33.plot(X8g0r[2, 1:], label='true',color='black')
#ax33.plot(Xn8g0r[2, 1:], label='true+noise')
ax33.plot(Zn8g0r[1, 1:], label='measurements', linestyle='', marker='x', color='grey')
ax33.plot(E8g0r[2, :], label='ekf_eigen3',color='red')
ax33.plot(E8g0r_ekf[2, :], label='ekf', color='blue')
ax33.plot(E8g0r_ekf_rad[2, :], label='ekf(rad)', color='brown')
ax33.set_title("8G - Y")
ax33.set_xlabel('step')
ax33.set_ylabel('y,met')
ax33.grid(True)

ax43 = fig3.add_subplot(4,2,5)
ax43.plot(X8g0r[4, 1:], label='true',color='black')
#ax43.plot(Xn8g0r[4, 1:], label='true+noise')
ax43.plot(Zn8g0r[2, 1:], label='measurements', linestyle='', marker='x',color='grey')
ax43.plot(E8g0r[4, :], label='ekf_eigen3',color='red')
ax43.plot(E8g0r_ekf[4, :], label='ekf', color='blue')
ax43.plot(E8g0r_ekf_rad[4, :], label='ekf(rad)', color='brown')
ax43.set_title("5G - Z")
ax43.set_xlabel('step')
ax43.set_ylabel('z,met')
ax43.grid(True)

ax53 = fig3.add_subplot(4,4,13)
ax53.plot(X8g0r[6, 1:], label='true',color='black')
#ax53.plot(Xn8g0r[6, 1:], label='true+noise')
ax53.plot(E8g0r[6, :], label='ekf_eigen3',color='red')
ax53.plot(E8g0r_ekf[6, :], label='ekf',color='blue')
ax53.set_title("8G - W")
ax53.set_xlabel('step')
ax53.set_ylabel('w,deg.')
ax53.grid(True)

ax530 = fig3.add_subplot(4,4,14)
ax530.plot(X8g0r_rad[6, 1:], label='true',color='black')
#ax530.plot(Xn8g0r[6, 1:], label='true+noise')
ax530.plot(E8g0r_ekf_rad[6, :], label='ekf(rad)', color='brown')
ax530.set_title("8G - W")
ax530.set_xlabel('step')
ax530.set_ylabel('w,deg.')
ax530.grid(True)

ax63 = fig3.add_subplot(4,2,6)
ax63.plot(E8g0r_cov0[0,0], label='ekf_eigen3: X0 - covariations', linestyle='', marker='o')
ax63.plot(E8g0r_cov0[0,1], label='ekf_eigen3: Y0 - covariations', linestyle='', marker='o')
ax63.plot(E8g0r_cov0[0,2], label='ekf_eigen3: Z0 - covariations', linestyle='', marker='o')
ax63.plot(E8g0r_cov0_ekf[0,0], label='ekf: X0 - covariations', linestyle='', marker='o')
ax63.plot(E8g0r_cov0_ekf[0,1], label='ekf: Y0 - covariations', linestyle='', marker='o')
ax63.plot(E8g0r_cov0_ekf[0,2], label='ekf: Z0 - covariations', linestyle='', marker='o')
ax63.plot(E8g0r_cov[0, :], label='ekf_eigen3: X - covariations')
ax63.plot(E8g0r_cov[1, :], label='ekf_eigen3: Y - covariations')
ax63.plot(E8g0r_cov[2, :], label='ekf_eigen3: Z - covariations')
ax63.plot(E8g0r_cov_ekf[0, :], label='ekf: X - covariations')
ax63.plot(E8g0r_cov_ekf[1, :], label='ekf: Y - covariations')
ax63.plot(E8g0r_cov_ekf[2, :], label='ekf: Z - covariations')
ax63.set_title("8G - cov")
ax63.set_xlabel('step')
ax63.set_ylabel('cov')
ax63.grid(True)

ax630 = fig3.add_subplot(4,2,8)
ax630.plot(E8g0r_cov0_ekf_rad[0,0], label='ekf(rad): X0 - covariations', linestyle='', marker='o')
ax630.plot(E8g0r_cov0_ekf_rad[0,1], label='ekf(rad): Y0 - covariations', linestyle='', marker='o')
ax630.plot(E8g0r_cov0_ekf_rad[0,2], label='ekf(rad): Z0 - covariations', linestyle='', marker='o')
ax630.plot(E8g0r_cov_ekf_rad[0, :], label='ekf(rad): X - covariations')
ax630.plot(E8g0r_cov_ekf_rad[1, :], label='ekf(rad): Y - covariations')
ax630.plot(E8g0r_cov_ekf_rad[2, :], label='ekf(rad): Z - covariations')
ax630.set_title("8G - cov")
ax630.set_xlabel('step')
ax630.set_ylabel('cov')
ax630.grid(True)

#plt.show()
##############################################################################################################
def calc_err(X,dt):
    Xn = add_process_noise(X,G@Q@G.T)
    Zn = make_meas(Xn,R)
    [est,est_cov,est_cov0,est1,est_cov1,est_cov01,est1r,est_cov1r,est_cov01r] = step(Zn,dt)
    err = est - Xn[:, 1:]
    err1 = est1 - Xn[:, 1:]
    err1r = est1r - Xn[:, 1:]
    return err, err1, err1r

from tqdm import tqdm

def calc_std_err(X,dt):
    num_iterations = 2000
    var_err = np.zeros((X.shape[0], X.shape[1]-1))
    var_err1 = np.zeros((X.shape[0], X.shape[1]-1))
    var_err1r = np.zeros((X.shape[0], X.shape[1]-1))

    for i in tqdm(range(num_iterations)):
        [err, err1, err1r] = calc_err(X,dt)
        var_err += err ** 2
        var_err1 += err1 ** 2
        var_err1r += err1r ** 2

    var_err /= num_iterations
    var_err1 /= num_iterations
    var_err1r /= num_iterations
    return np.sqrt(var_err), np.sqrt(var_err1), np.sqrt(var_err1r)

[std_err_2g0r, std_err_2g0r1, std_err_2g0r1r] = calc_std_err(X2g0r,T)
[std_err_5g0r, std_err_5g0r1, std_err_5g0r1r] = calc_std_err(X5g0r,T)
[std_err_8g0r, std_err_8g0r1, std_err_8g0r1r] = calc_std_err(X8g0r,T)
#####################################################################################
plt.figure(figsize=(20,45))
# x
plt.subplot(8, 3, 1)
plt.plot((np.arange(len(std_err_2g0r[0, :]))+1)*T, std_err_2g0r[0, :].T,color='red')
plt.plot((np.arange(len(std_err_2g0r1[0, :]))+1)*T, std_err_2g0r1[0, :].T,color='blue')
plt.plot((np.arange(len(std_err_2g0r1r[0, :]))+1)*T, std_err_2g0r1r[0, :].T,color='brown')
plt.grid(True)
plt.xlabel('Time,s')
plt.ylabel('std_x(2G), met')

plt.subplot(8, 3, 2)
plt.plot((np.arange(len(std_err_5g0r[0, :]))+1)*T, std_err_5g0r[0, :].T,color='red')
plt.plot((np.arange(len(std_err_5g0r1[0, :]))+1)*T, std_err_5g0r1[0, :].T,color='blue')
plt.plot((np.arange(len(std_err_5g0r1r[0, :]))+1)*T, std_err_5g0r1r[0, :].T,color='brown')
plt.grid(True)
plt.xlabel('Time,s')
plt.ylabel('std_x(5G), met')

plt.subplot(8, 3, 3)
plt.plot((np.arange(len(std_err_8g0r[0, :]))+1)*T, std_err_8g0r[0, :].T,color='red')
plt.plot((np.arange(len(std_err_8g0r1[0, :]))+1)*T, std_err_8g0r1[0, :].T,color='blue')
plt.plot((np.arange(len(std_err_8g0r1r[0, :]))+1)*T, std_err_8g0r1r[0, :].T,color='brown')
plt.grid(True)
plt.xlabel('Time,s')
plt.ylabel('std_x(8G), met')

# y
plt.subplot(8, 3, 4)
plt.plot((np.arange(len(std_err_2g0r[2, :]))+1)*T, std_err_2g0r[2, :].T,color='red')
plt.plot((np.arange(len(std_err_2g0r1[2, :]))+1)*T, std_err_2g0r1[2, :].T,color='blue')
plt.plot((np.arange(len(std_err_2g0r1r[2, :]))+1)*T, std_err_2g0r1r[2, :].T,color='brown')
plt.grid(True)
plt.xlabel('Time,s')
plt.ylabel('std_y(2G), met')

plt.subplot(8, 3, 5)
plt.plot((np.arange(len(std_err_5g0r[2, :]))+1)*T, std_err_5g0r[2, :].T,color='red')
plt.plot((np.arange(len(std_err_5g0r1[2, :]))+1)*T, std_err_5g0r1[2, :].T,color='blue')
plt.plot((np.arange(len(std_err_5g0r1r[2, :]))+1)*T, std_err_5g0r1r[2, :].T,color='brown')
plt.grid(True)
plt.xlabel('Time,s')
plt.ylabel('std_y(5G), met')

plt.subplot(8, 3, 6)
plt.plot((np.arange(len(std_err_8g0r[2, :]))+1)*T, std_err_8g0r[2, :].T,color='red')
plt.plot((np.arange(len(std_err_8g0r1[2, :]))+1)*T, std_err_8g0r1[2, :].T,color='blue')
plt.plot((np.arange(len(std_err_8g0r1r[2, :]))+1)*T, std_err_8g0r1r[2, :].T,color='brown')
plt.grid(True)
plt.xlabel('Time,s')
plt.ylabel('std_y(8G), met')

# z
plt.subplot(8, 3, 7)
plt.plot((np.arange(len(std_err_2g0r[4, :]))+1)*T, std_err_2g0r[4, :].T,color='red')
plt.plot((np.arange(len(std_err_2g0r1[4, :]))+1)*T, std_err_2g0r1[4, :].T,color='blue')
plt.plot((np.arange(len(std_err_2g0r1r[4, :]))+1)*T, std_err_2g0r1r[4, :].T,color='brown')
plt.grid(True)
plt.xlabel('Time,s')
plt.ylabel('std_z(2G), met')

plt.subplot(8, 3, 8)
plt.plot((np.arange(len(std_err_5g0r[4, :]))+1)*T, std_err_5g0r[4, :].T,color='red')
plt.plot((np.arange(len(std_err_5g0r1[4, :]))+1)*T, std_err_5g0r1[4, :].T,color='blue')
plt.plot((np.arange(len(std_err_5g0r1r[4, :]))+1)*T, std_err_5g0r1r[4, :].T,color='brown')
plt.grid(True)
plt.xlabel('Time,s')
plt.ylabel('std_z(5G), met')

plt.subplot(8, 3, 9)
plt.plot((np.arange(len(std_err_8g0r[4, :]))+1)*T, std_err_8g0r[4, :].T,color='red')
plt.plot((np.arange(len(std_err_8g0r1[4, :]))+1)*T, std_err_8g0r1[4, :].T,color='blue')
plt.plot((np.arange(len(std_err_8g0r1r[4, :]))+1)*T, std_err_8g0r1r[4, :].T,color='brown')
plt.grid(True)
plt.xlabel('Time,s')
plt.ylabel('std_z(8G), met')
# vx
plt.subplot(8, 3, 10)
plt.plot((np.arange(len(std_err_2g0r[1, :]))+1)*T, std_err_2g0r[1, :].T,color='red')
plt.plot((np.arange(len(std_err_2g0r1[1, :]))+1)*T, std_err_2g0r1[1, :].T,color='blue')
plt.plot((np.arange(len(std_err_2g0r1r[1, :]))+1)*T, std_err_2g0r1r[1, :].T,color='brown')
plt.grid(True)
plt.xlabel('Time,s')
plt.ylabel('std_vx(2G), met')

plt.subplot(8, 3, 11)
plt.plot((np.arange(len(std_err_5g0r[1, :]))+1)*T, std_err_5g0r[1, :].T,color='red')
plt.plot((np.arange(len(std_err_5g0r1[1, :]))+1)*T, std_err_5g0r1[1, :].T,color='blue')
plt.plot((np.arange(len(std_err_5g0r1r[1, :]))+1)*T, std_err_5g0r1r[1, :].T,color='brown')
plt.grid(True)
plt.xlabel('Time,s')
plt.ylabel('std_vx(5G), met')

plt.subplot(8, 3, 12)
plt.plot((np.arange(len(std_err_8g0r[1, :]))+1)*T, std_err_8g0r[1, :].T,color='red')
plt.plot((np.arange(len(std_err_8g0r1[1, :]))+1)*T, std_err_8g0r1[1, :].T,color='blue')
plt.plot((np.arange(len(std_err_8g0r1r[1, :]))+1)*T, std_err_8g0r1r[1, :].T,color='brown')
plt.grid(True)
plt.xlabel('Time,s')
plt.ylabel('std_vx(8G), met')

# vy
plt.subplot(8, 3, 13)
plt.plot((np.arange(len(std_err_2g0r[3, :]))+1)*T, std_err_2g0r[3, :].T,color='red')
plt.plot((np.arange(len(std_err_2g0r1[3, :]))+1)*T, std_err_2g0r1[3, :].T,color='blue')
plt.plot((np.arange(len(std_err_2g0r1r[3, :]))+1)*T, std_err_2g0r1r[3, :].T,color='brown')
plt.grid(True)
plt.xlabel('Time,s')
plt.ylabel('std_vy(2G), met')

plt.subplot(8, 3, 14)
plt.plot((np.arange(len(std_err_5g0r[3, :]))+1)*T, std_err_5g0r[3, :].T,color='red')
plt.plot((np.arange(len(std_err_5g0r1[3, :]))+1)*T, std_err_5g0r1[3, :].T,color='blue')
plt.plot((np.arange(len(std_err_5g0r1r[3, :]))+1)*T, std_err_5g0r1r[3, :].T,color='brown')
plt.grid(True)
plt.xlabel('Time,s')
plt.ylabel('std_vy(5G), met')

plt.subplot(8, 3, 15)
plt.plot((np.arange(len(std_err_8g0r[3, :]))+1)*T, std_err_8g0r[3, :].T,color='red')
plt.plot((np.arange(len(std_err_8g0r1[3, :]))+1)*T, std_err_8g0r1[3, :].T,color='blue')
plt.plot((np.arange(len(std_err_8g0r1r[3, :]))+1)*T, std_err_8g0r1r[3, :].T,color='brown')
plt.grid(True)
plt.xlabel('Time,s')
plt.ylabel('stdvy(8G), met')

# vz
plt.subplot(8, 3, 16)
plt.plot((np.arange(len(std_err_2g0r[5, :]))+1)*T, std_err_2g0r[5, :].T,color='red')
plt.plot((np.arange(len(std_err_2g0r1[5, :]))+1)*T, std_err_2g0r1[5, :].T,color='blue')
plt.plot((np.arange(len(std_err_2g0r1r[5, :]))+1)*T, std_err_2g0r1r[5, :].T,color='brown')
plt.grid(True)
plt.xlabel('Time,s')
plt.ylabel('std_vz(2G), met')

plt.subplot(8, 3, 17)
plt.plot((np.arange(len(std_err_5g0r[5, :]))+1)*T, std_err_5g0r[5, :].T,color='red')
plt.plot((np.arange(len(std_err_5g0r1[5, :]))+1)*T, std_err_5g0r1[5, :].T,color='blue')
plt.plot((np.arange(len(std_err_5g0r1r[5, :]))+1)*T, std_err_5g0r1r[5, :].T,color='brown')
plt.grid(True)
plt.xlabel('Time,s')
plt.ylabel('std_vz(5G), met')

plt.subplot(8, 3, 18)
plt.plot((np.arange(len(std_err_8g0r[5, :]))+1)*T, std_err_8g0r[5, :].T,color='red')
plt.plot((np.arange(len(std_err_8g0r1[5, :]))+1)*T, std_err_8g0r1[5, :].T,color='blue')
plt.plot((np.arange(len(std_err_8g0r1r[5, :]))+1)*T, std_err_8g0r1r[5, :].T,color='brown')
plt.grid(True)
plt.xlabel('Time,s')
plt.ylabel('std_vz(8G), met')
# w
plt.subplot(8, 3, 19)
plt.plot((np.arange(len(std_err_2g0r[6, :]))+1)*T, std_err_2g0r[6, :].T,color='red')
plt.plot((np.arange(len(std_err_2g0r1[6, :]))+1)*T, std_err_2g0r1[6, :].T,color='blue')
plt.grid(True)
plt.xlabel('Time,s')
plt.ylabel('std_w(2G), met')

plt.subplot(8, 3, 20)
plt.plot((np.arange(len(std_err_5g0r[6, :]))+1)*T, std_err_5g0r[6, :].T,color='red')
plt.plot((np.arange(len(std_err_5g0r1[6, :]))+1)*T, std_err_5g0r1[6, :].T,color='blue')
plt.grid(True)
plt.xlabel('Time,s')
plt.ylabel('std_w(5G), met')

plt.subplot(8, 3, 21)
plt.plot((np.arange(len(std_err_8g0r[6, :]))+1)*T, std_err_8g0r[6, :].T,color='red')
plt.plot((np.arange(len(std_err_8g0r1[6, :]))+1)*T, std_err_8g0r1[6, :].T,color='blue')
plt.grid(True)
plt.xlabel('Time,s')
plt.ylabel('std_w(8G), met')

plt.subplot(8, 3, 22)
plt.plot((np.arange(len(std_err_2g0r1r[6, :]))+1)*T, std_err_2g0r1r[6, :].T,color='brown')
plt.grid(True)
plt.xlabel('Time,s')
plt.ylabel('std_w(2G), met')

plt.subplot(8, 3, 23)
plt.plot((np.arange(len(std_err_5g0r1r[6, :]))+1)*T, std_err_5g0r1r[6, :].T,color='brown')
plt.grid(True)
plt.xlabel('Time,s')
plt.ylabel('std_w(5G), met')

plt.subplot(8, 3, 24)
plt.plot((np.arange(len(std_err_8g0r1r[6, :]))+1)*T, std_err_8g0r1r[6, :].T,color='brown')
plt.grid(True)
plt.xlabel('Time,s')
plt.ylabel('std_w(8G), met')

plt.show()
