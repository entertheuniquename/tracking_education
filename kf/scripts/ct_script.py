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
T = 6

# Ошибки процесса
process_var = 0.01
process_var_w = 0.000001

# Ошибки измерений
meas_std = 30

# Угловая скорость на развороте в радианах
w2g_r = 0.098
w5g_r = 0.245
w8g_r = 0.392

# Угловая скорость на развороте в градусах
w2g_d = w2g_r*180/math.pi
w5g_d = w5g_r*180/math.pi
w8g_d = w8g_r*180/math.pi
print("w2g_d: "+str(w2g_d))
print("w5g_d: "+str(w5g_d))
print("w8g_d: "+str(w8g_d))

# Матрица ошибок процесса
Q0 = np.diag([process_var, process_var, process_var, process_var_w])
G = np.array([[T**2/2, 0,      0     , 0],
              [T,      0,      0     , 0],
              [0,      T**2/2, 0     , 0],
              [0,      T,      0     , 0],
              [0,      0,      T**2/2, 0],
              [0,      0,      T     , 0],
              [0,      0,      0     , T]])

#print("Q0:")
#print(Q0)

Q = G@Q0@G.T

#print("Q:")
#print(Q)

# Матрица ошибок измерения
R = np.diag([meas_std*meas_std, meas_std*meas_std, meas_std*meas_std])

# Векторы входных данных
initialState2gr = np.array([30000., -200., 0., 0., 0., 0., w2g_r])#radian
initialState5gr = np.array([30000., -200., 0., 0., 0., 0., w5g_r])#radian
initialState8gr = np.array([30000., -200., 0., 0., 0., 0., w8g_r])#radian

### Создание наборов данных ##################################################################################

# Функция создания набора данных: по кругу
def make_true_data_round(x0, am, dt):
    # Создание обнулённой матрицы нужного размера
    X = np.zeros((x0.shape[0], am))
    # Запись первого значения
    X[:, 0] = x0.T
    # Цикл создания участка разворота
    for i in range(am-1):
        xx = e.stateModel_CTx(np.copy(X[:, i]),dt)
        X[:, i+1] = xx.flatten()
    return X

X2g0r=make_true_data_round(initialState2gr, 200, T)
X5g0r=make_true_data_round(initialState5gr, 200, T)
X8g0r=make_true_data_round(initialState8gr, 200, T)

### Добавление к наборам данных ошибок процесса ##############################################################
def add_process_noise(X,Var):
    Xn = X + np.sqrt(Var) @ np.random.normal(loc=0, scale=1.0, size=(X.shape[0], X.shape[1]))
    return Xn

Xn2g0r = add_process_noise(X2g0r,Q)
Xn5g0r = add_process_noise(X5g0r,Q)
Xn8g0r = add_process_noise(X8g0r,Q)

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

### Фильтрация XYZ_EKFE_CT ###################################################################################

# Имитация отметки
def make_tu(TP,M):
    tu = (TP,M)
    return tu

# Функция фильтрации набора данных
def stepEKFE_xyz_ct(Zn, dt):
    time = 0
    track = e.BindTrackEKFE_xyz_ct(make_tu(time,Zn[:, 0][:, np.newaxis]))

    P0 = np.zeros((1,3))
    c = track.getCov()
    P0[0][0] = np.sqrt(c[0][0])
    P0[0][1] = np.sqrt(c[2][2])
    P0[0][2] = np.sqrt(c[4][4])

    est = np.zeros((7, Zn.shape[1]-1))#7 - bad! not dt
    print("Zn("+str(Zn.shape[0])+","+str(Zn.shape[1])+"):")
    print(Zn)
    print("est("+str(est.shape[0])+","+str(est.shape[1])+"):")
    print(est)
    est_cov = np.zeros((3, Zn.shape[1]-1))#3 - bad! not dt
    for col in range(Zn.shape[1]-1):
        time = time+dt
        ee = track.step(make_tu(time,np.copy(Zn[:, col+1])))
        print("Zn["+str(col+1)+"]:")
        print(Zn[:, col+1])
        print("ee["+str(col+1)+"]:")
        print(ee[0])
        print(ee[1])
        ee_cov = np.zeros((1,3))
        ee_cov[0][0] = np.sqrt(ee[1][0][0])
        ee_cov[0][1] = np.sqrt(ee[1][2][2])
        ee_cov[0][2] = np.sqrt(ee[1][4][4])

        est[:, col] = np.squeeze(ee[0][:])
        print("est["+str(col)+"]:")
        print(est[:, col])
        est_cov[:, col] = np.squeeze(ee_cov[0][:])
        print("est_cov["+str(col)+"]:")
        print(est_cov[:, col])
    return est, est_cov, P0



[E2g0r, E2g0r_cov, E2g0r_cov0] = stepEKFE_xyz_ct(Zn2g0r, T)
[E5g0r, E5g0r_cov, E5g0r_cov0] = stepEKFE_xyz_ct(Zn5g0r, T)
[E8g0r, E8g0r_cov, E8g0r_cov0] = stepEKFE_xyz_ct(Zn8g0r, T)

### Отрисовка графиков для сглаживания #######################################################################
fig1 = plt.figure(figsize=(18,25))

ax11 = fig1.add_subplot(2,2,2)
ax11.plot(X2g0r[0, :], X2g0r[2, :], label='X2g0r - true')
ax11.plot(Xn2g0r[0, :], Xn2g0r[2, :], label='Xn2g0r - true+noize')
ax11.plot(Zn2g0r[0, :], Zn2g0r[1, :], label='Zn2g0r - measurements')
ax11.plot(E2g0r[0, :], E2g0r[2, :], label='E2g0r - estimations')
ax11.set_title("2G - [x,y] - w-rad")
ax11.set_xlabel('x,met.')
ax11.set_ylabel('y,met.')
ax11.grid(True)

ax21 = fig1.add_subplot(4,2,1)
ax21.plot(X2g0r[0, 1:], label='X2gr - true')
ax21.plot(Xn2g0r[0, 1:], label='Xn2gr - true+noise')
ax21.plot(Zn2g0r[0, 1:], label='Zn2gr - measurements')
ax21.plot(Zn2g0r[0, 1:], label='Zn2gr - measurements', linestyle='', marker='x')
ax21.plot(E2g0r[0, :], label='E2g0r - estimations')
ax21.set_title("2G - X")
ax21.set_xlabel('t.')
ax21.set_ylabel('x.')
ax21.grid(True)

ax31 = fig1.add_subplot(4,2,3)
ax31.plot(X2g0r[2, 1:], label='X2gr - true')
ax31.plot(Xn2g0r[2, 1:], label='Xn2gr - true+noise')
ax31.plot(Zn2g0r[1, 1:], label='Zn2gr - measurements')
ax31.plot(Zn2g0r[1, 1:], label='Zn2gr - measurements', linestyle='', marker='x')
ax31.plot(E2g0r[2, :], label='E2g0r - estimations')
ax31.set_title("2G - Y")
ax31.set_xlabel('t.')
ax31.set_ylabel('y.')
ax31.grid(True)

ax41 = fig1.add_subplot(4,2,5)
ax41.plot(X2g0r[4, 1:], label='X2g0r - true')
ax41.plot(Xn2g0r[4, 1:], label='Xn2g0r - true+noise')
ax41.plot(Zn2g0r[2, 1:], label='Zn2g0r - measurements')
ax41.plot(Zn2g0r[2, 1:], label='Zn2g0r - measurements', linestyle='', marker='x')
ax41.plot(E2g0r[4, :], label='E2g0r - estimations')
ax41.set_title("2G - Z")
ax41.set_xlabel('t.')
ax41.set_ylabel('z.')
ax41.grid(True)

ax51 = fig1.add_subplot(4,2,7)
ax51.plot(X2g0r[6, 1:], label='X2g0r - true')
ax51.plot(Xn2g0r[6, 1:], label='Xn2g0r - true+noise')
ax51.plot(0, label='Zn2g0r - measurements', linestyle='', marker='+')
ax51.plot(E2g0r[6, :], label='E2g0r - estimations')
ax51.set_title("2G - W")
ax51.set_xlabel('t.')
ax51.set_ylabel('w.')
ax51.grid(True)

ax61 = fig1.add_subplot(2,2,4)
ax61.plot(E2g0r_cov0[0,0], label='E2g0r0_cov - X0 - covariations', linestyle='', marker='o')
ax61.plot(E2g0r_cov0[0,1], label='E2g0r0_cov - Y0 - covariations', linestyle='', marker='o')
ax61.plot(E2g0r_cov0[0,2], label='E2g0r0_cov - Z0 - covariations', linestyle='', marker='o')
ax61.plot(E2g0r_cov[0, :], label='E2g0r_cov - X - covariations')
ax61.plot(E2g0r_cov[1, :], label='E2g0r_cov - Y - covariations')
ax61.plot(E2g0r_cov[2, :], label='E2g0r_cov - Z - covariations')
ax61.set_title("2G - cov")
ax61.set_xlabel('t,met.')
ax61.set_ylabel('z,met.')
ax61.grid(True)

#plt.show()
##############################################################################################################
fig2 = plt.figure(figsize=(18,25))

ax12 = fig2.add_subplot(2,2,2)
ax12.plot(X5g0r[0, :], X5g0r[2, :], label='X5g0r - true')
ax12.plot(Xn5g0r[0, :], Xn5g0r[2, :], label='Xn5g0r - true+noize')
ax12.plot(Zn5g0r[0, :], Zn5g0r[1, :], label='Zn5g0r - measurements')
ax12.plot(E5g0r[0, :], E5g0r[2, :], label='E5g0r - estimations')
ax12.set_title("5G - [x,y] - w-rad")
ax12.set_xlabel('x,met.')
ax12.set_ylabel('y,met.')
ax12.grid(True)

ax22 = fig2.add_subplot(4,2,1)
ax22.plot(X5g0r[0, 1:], label='X5gr - true')
ax22.plot(Xn5g0r[0, 1:], label='Xn5gr - true+noise')
ax22.plot(Zn5g0r[0, 1:], label='Zn5gr - measurements')
ax22.plot(Zn5g0r[0, 1:], label='Zn5gr - measurements', linestyle='', marker='x')
ax22.plot(E5g0r[0, :], label='E5g0r - estimations')
ax22.set_title("5G - X")
ax22.set_xlabel('t.')
ax22.set_ylabel('x.')
ax22.grid(True)

ax32 = fig2.add_subplot(4,2,3)
ax32.plot(X5g0r[2, 1:], label='X5gr - true')
ax32.plot(Xn5g0r[2, 1:], label='Xn5gr - true+noise')
ax32.plot(Zn5g0r[1, 1:], label='Zn5gr - measurements')
ax32.plot(Zn5g0r[1, 1:], label='Zn5gr - measurements', linestyle='', marker='x')
ax32.plot(E5g0r[2, :], label='E5g0r - estimations')
ax32.set_title("5G - Y")
ax32.set_xlabel('t.')
ax32.set_ylabel('y.')
ax32.grid(True)

ax42 = fig2.add_subplot(4,2,5)
ax42.plot(X5g0r[4, 1:], label='X5g0r - true')
ax42.plot(Xn5g0r[4, 1:], label='Xn5g0r - true+noise')
ax42.plot(Zn5g0r[2, 1:], label='Zn5g0r - measurements')
ax42.plot(Zn5g0r[2, 1:], label='Zn5g0r - measurements', linestyle='', marker='x')
ax42.plot(E5g0r[4, :], label='E5g0r - estimations')
ax42.set_title("5G - Z")
ax42.set_xlabel('t.')
ax42.set_ylabel('z.')
ax42.grid(True)

ax52 = fig2.add_subplot(4,2,7)
ax52.plot(X5g0r[6, 1:], label='X5g0r - true')
ax52.plot(Xn5g0r[6, 1:], label='Xn5g0r - true+noise')
ax52.plot(0, label='Zn5g0r - measurements', linestyle='', marker='+')
ax52.plot(E5g0r[6, :], label='E5g0r - estimations')
ax52.set_title("5G - W")
ax52.set_xlabel('t.')
ax52.set_ylabel('w.')
ax52.grid(True)

ax62 = fig2.add_subplot(2,2,4)
ax62.plot(E5g0r_cov0[0,0], label='E5g0r0_cov - X0 - covariations', linestyle='', marker='o')
ax62.plot(E5g0r_cov0[0,1], label='E5g0r0_cov - Y0 - covariations', linestyle='', marker='o')
ax62.plot(E5g0r_cov0[0,2], label='E5g0r0_cov - Z0 - covariations', linestyle='', marker='o')
ax62.plot(E5g0r_cov[0, :], label='E5g0r_cov - X - covariations')
ax62.plot(E5g0r_cov[1, :], label='E5g0r_cov - Y - covariations')
ax62.plot(E5g0r_cov[2, :], label='E5g0r_cov - Z - covariations')
ax62.set_title("5G - cov")
ax62.set_xlabel('t,met.')
ax62.set_ylabel('z,met.')
ax62.grid(True)

#plt.show()
##############################################################################################################
fig3 = plt.figure(figsize=(18,25))

ax13 = fig3.add_subplot(2,2,2)
ax13.plot(X8g0r[0, :], X8g0r[2, :], label='X8g0r - true')
ax13.plot(Xn8g0r[0, :], Xn8g0r[2, :], label='Xn8g0r - true+noize')
ax13.plot(Zn8g0r[0, :], Zn8g0r[1, :], label='Zn8g0r - measurements')
ax13.plot(E8g0r[0, :], E8g0r[2, :], label='E8g0r - estimations')
ax13.set_title("8G - [x,y] - w-rad")
ax13.set_xlabel('x,met.')
ax13.set_ylabel('y,met.')
ax13.grid(True)

ax23 = fig3.add_subplot(4,2,1)
ax23.plot(X8g0r[0, 1:], label='X8gr - true')
ax23.plot(Xn8g0r[0, 1:], label='Xn8gr - true+noise')
ax23.plot(Zn8g0r[0, 1:], label='Zn8gr - measurements')
ax23.plot(Zn8g0r[0, 1:], label='Zn8gr - measurements', linestyle='', marker='x')
ax23.plot(E8g0r[0, :], label='E8g0r - estimations')
ax23.set_title("8G - X")
ax23.set_xlabel('t.')
ax23.set_ylabel('x.')
ax23.grid(True)

ax33 = fig3.add_subplot(4,2,3)
ax33.plot(X8g0r[2, 1:], label='X8gr - true')
ax33.plot(Xn8g0r[2, 1:], label='Xn8gr - true+noise')
ax33.plot(Zn8g0r[1, 1:], label='Zn8gr - measurements')
ax33.plot(Zn8g0r[1, 1:], label='Zn8gr - measurements', linestyle='', marker='x')
ax33.plot(E8g0r[2, :], label='E8g0r - estimations')
ax33.set_title("8G - Y")
ax33.set_xlabel('t.')
ax33.set_ylabel('y.')
ax33.grid(True)

ax43 = fig3.add_subplot(4,2,5)
ax43.plot(X8g0r[4, 1:], label='X8g0r - true')
ax43.plot(Xn8g0r[4, 1:], label='Xn8g0r - true+noise')
ax43.plot(Zn8g0r[2, 1:], label='Zn8g0r - measurements')
ax43.plot(Zn8g0r[2, 1:], label='Zn8g0r - measurements', linestyle='', marker='x')
ax43.plot(E8g0r[4, :], label='E8g0r - estimations')
ax43.set_title("58 - Z")
ax43.set_xlabel('t.')
ax43.set_ylabel('z.')
ax43.grid(True)

ax53 = fig3.add_subplot(4,2,7)
ax53.plot(X8g0r[6, 1:], label='X8g0r - true')
ax53.plot(Xn8g0r[6, 1:], label='Xn8g0r - true+noise')
ax53.plot(0, label='Zn8g0r - measurements', linestyle='', marker='+')
ax53.plot(E8g0r[6, :], label='E8g0r - estimations')
ax53.set_title("8G - W")
ax53.set_xlabel('t.')
ax53.set_ylabel('w.')
ax53.grid(True)

ax63 = fig3.add_subplot(2,2,4)
ax63.plot(E8g0r_cov0[0,0], label='E8g0r0_cov - X0 - covariations', linestyle='', marker='o')
ax63.plot(E8g0r_cov0[0,1], label='E8g0r0_cov - Y0 - covariations', linestyle='', marker='o')
ax63.plot(E8g0r_cov0[0,2], label='E8g0r0_cov - Z0 - covariations', linestyle='', marker='o')
ax63.plot(E8g0r_cov[0, :], label='E8g0r_cov - X - covariations')
ax63.plot(E8g0r_cov[1, :], label='E8g0r_cov - Y - covariations')
ax63.plot(E8g0r_cov[2, :], label='E8g0r_cov - Z - covariations')
ax63.set_title("8G - cov")
ax63.set_xlabel('t,met.')
ax63.set_ylabel('z,met.')
ax63.grid(True)

plt.show()
##############################################################################################################
#def calc_err_EKFE_xyz_ct(X,dt):
#    Xn = add_process_noise(X,Q)
#    Zn = make_meas(Xn,R)
#    est = stepEKFE_xyz_ct(Zn,dt)
#    err = est - Xn[:, 1:]
#    return err

#from tqdm import tqdm

#def calc_std_err_EKFE_xyz_ct(X,dt):
#    num_iterations = 2000
#    var_err = np.zeros((X.shape[0], X.shape[1]-1))

#    for i in tqdm(range(num_iterations)):
#        err = calc_err_EKFE_xyz_ct(X,dt)
#        var_err += err ** 2

#    var_err /= num_iterations
#    return np.sqrt(var_err)

#std_err_2g0r = calc_std_err_EKFE_xyz_ct(X2g0r,T)
##std_err_5g0 = calc_std_err_EKFE_xyz_ct(X5g0,T,Ptd)
##std_err_8g0 = calc_std_err_EKFE_xyz_ct(X8g0,T,Ptd)
#####################################################################################
#plt.figure(figsize=(20,45))

##plt.title("[x,vx,y,vy,z,vz,w]")

##plt.subplot(7, 2, 1)
##plt.plot((np.arange(len(std_err_2g[0, :]))+1)*T, std_err_2g[0, :].T)
##plt.plot((np.arange(len(std_err_5g[0, :]))+1)*T, std_err_5g[0, :].T)
##plt.plot((np.arange(len(std_err_8g[0, :]))+1)*T, std_err_8g[0, :].T)
##plt.grid(True)
##plt.xlabel('Time,s')
##plt.ylabel('std_x, met')

#plt.subplot(7, 2, 2)
#plt.plot((np.arange(len(std_err_2g0r[0, :]))+1)*T, std_err_2g0r[0, :].T)
##plt.plot((np.arange(len(std_err_5g0[0, :]))+1)*T, std_err_5g0[0, :].T)
##plt.plot((np.arange(len(std_err_8g0[0, :]))+1)*T, std_err_8g0[0, :].T)
#plt.grid(True)
#plt.xlabel('Time,s')
#plt.ylabel('std_x, met')

##plt.subplot(7, 2, 3)
##plt.plot((np.arange(len(std_err_2g[1, :]))+1)*T, std_err_2g[1, :].T)
##plt.plot((np.arange(len(std_err_5g[1, :]))+1)*T, std_err_5g[1, :].T)
##plt.plot((np.arange(len(std_err_8g[1, :]))+1)*T, std_err_8g[1, :].T)
##plt.grid(True)
##plt.xlabel('Time,s')
##plt.ylabel('std_vx, m/s')

#plt.subplot(7, 2, 4)
#plt.plot((np.arange(len(std_err_2g0r[1, :]))+1)*T, std_err_2g0r[1, :].T)
##plt.plot((np.arange(len(std_err_5g0[1, :]))+1)*T, std_err_5g0[1, :].T)
##plt.plot((np.arange(len(std_err_8g0[1, :]))+1)*T, std_err_8g0[1, :].T)
#plt.grid(True)
#plt.xlabel('Time,s')
#plt.ylabel('std_vx, m/s')

##plt.subplot(7, 2, 5)
##plt.plot((np.arange(len(std_err_2g[2, :]))+1)*T, std_err_2g[2, :].T)
##plt.plot((np.arange(len(std_err_5g[2, :]))+1)*T, std_err_5g[2, :].T)
##plt.plot((np.arange(len(std_err_8g[2, :]))+1)*T, std_err_8g[2, :].T)
##plt.grid(True)
##plt.xlabel('Time,s')
##plt.ylabel('std_y, met')

#plt.subplot(7, 2, 6)
#plt.plot((np.arange(len(std_err_2g0r[2, :]))+1)*T, std_err_2g0r[2, :].T)
##plt.plot((np.arange(len(std_err_5g0[2, :]))+1)*T, std_err_5g0[2, :].T)
##plt.plot((np.arange(len(std_err_8g0[2, :]))+1)*T, std_err_8g0[2, :].T)
#plt.grid(True)
#plt.xlabel('Time,s')
#plt.ylabel('std_y, met')

##plt.subplot(7, 2, 7)
##plt.plot((np.arange(len(std_err_2g[3, :]))+1)*T, std_err_2g[3, :].T)
##plt.grid(True)
##plt.xlabel('Time,s')
##plt.ylabel('std_z, met')

#plt.subplot(7, 2, 10)
#plt.plot((np.arange(len(std_err_2g0r[4, :]))+1)*T, std_err_2g0r[4, :].T)
##plt.plot((np.arange(len(std_err_5g0[4, :]))+1)*T, std_err_5g0[4, :].T)
##plt.plot((np.arange(len(std_err_8g0[4, :]))+1)*T, std_err_8g0[4, :].T)
#plt.grid(True)
#plt.xlabel('Time,s')
#plt.ylabel('std_z, met')

##plt.subplot(7, 2, 11)
##plt.plot((np.arange(len(std_err_2g[5, :]))+1)*T, std_err_2g[5, :].T)
##plt.plot((np.arange(len(std_err_5g[5, :]))+1)*T, std_err_5g[5, :].T)
##plt.plot((np.arange(len(std_err_8g[5, :]))+1)*T, std_err_8g[5, :].T)
##plt.grid(True)
##plt.xlabel('Time,s')
##plt.ylabel('std_vz, m/s')

#plt.subplot(7, 2, 12)
#plt.plot((np.arange(len(std_err_2g0r[5, :]))+1)*T, std_err_2g0r[5, :].T)
##plt.plot((np.arange(len(std_err_5g0[5, :]))+1)*T, std_err_5g0[5, :].T)
##plt.plot((np.arange(len(std_err_8g0[5, :]))+1)*T, std_err_8g0[5, :].T)
#plt.grid(True)
#plt.xlabel('Time,s')
#plt.ylabel('std_vz, m/s')

##plt.subplot(7, 2, 13)
##plt.plot((np.arange(len(std_err_2g[6, :]))+1)*T, std_err_2g[6, :].T)
##plt.plot((np.arange(len(std_err_5g[6, :]))+1)*T, std_err_5g[6, :].T)
##plt.plot((np.arange(len(std_err_8g[6, :]))+1)*T, std_err_8g[6, :].T)
##plt.grid(True)
##plt.xlabel('Time,s')
##plt.ylabel('std_w, m/s')

#plt.subplot(7, 2, 14)
#plt.plot((np.arange(len(std_err_2g0r[6, :]))+1)*T, std_err_2g0r[6, :].T)
##plt.plot((np.arange(len(std_err_5g0[6, :]))+1)*T, std_err_5g0[6, :].T)
##plt.plot((np.arange(len(std_err_8g0[6, :]))+1)*T, std_err_8g0[6, :].T)
#plt.grid(True)
#plt.xlabel('Time,s')
#plt.ylabel('std_w, m/s')

##plt.subplots_adjust(wspace=8.0, hspace=0.7)
#plt.show()
#t.plot((np.arange(len(std_err_5g[3, :]))+1)*T, std_err_5g[3, :].T)
##plt.plot((np.arange(len(std_err_8g[3, :]))+1)*T, std_err_8g[3, :].T)
##plt.grid(True)
##plt.xlabel('Time,s')
##plt.ylabel('std_vy, m/s')

#plt.subplot(7, 2, 8)
#plt.plot((np.arange(len(std_err_2g0r[3, :]))+1)*T, std_err_2g0r[3, :].T)
##plt.plot((np.arange(len(std_err_5g0[3, :]))+1)*T, std_err_5g0[3, :].T)
##plt.plot((np.arange(len(std_err_8g0[3, :]))+1)*T, std_err_8g0[3, :].T)
#plt.grid(True)
#plt.xlabel('Time,s')
#plt.ylabel('std_vy, m/s')

##plt.subplot(7, 2, 9)
##plt.plot((np.arange(len(std_err_2g[4, :]))+1)*T, std_err_2g[4, :].T)
##plt.plot((np.arange(len(std_err_5g[4, :]))+1)*T, std_err_5g[4, :].T)
##plt.plot((np.arange(len(std_err_8g[4, :]))+1)*T, std_err_8g[4, :].T)
##plt.grid(True)
##plt.xlabel('Time,s')
##plt.ylabel('std_z, met')

#plt.subplot(7, 2, 10)
#plt.plot((np.arange(len(std_err_2g0r[4, :]))+1)*T, std_err_2g0r[4, :].T)
##plt.plot((np.arange(len(std_err_5g0[4, :]))+1)*T, std_err_5g0[4, :].T)
##plt.plot((np.arange(len(std_err_8g0[4, :]))+1)*T, std_err_8g0[4, :].T)
#plt.grid(True)
#plt.xlabel('Time,s')
#plt.ylabel('std_z, met')

##plt.subplot(7, 2, 11)
##plt.plot((np.arange(len(std_err_2g[5, :]))+1)*T, std_err_2g[5, :].T)
##plt.plot((np.arange(len(std_err_5g[5, :]))+1)*T, std_err_5g[5, :].T)
##plt.plot((np.arange(len(std_err_8g[5, :]))+1)*T, std_err_8g[5, :].T)
##plt.grid(True)
##plt.xlabel('Time,s')
##plt.ylabel('std_vz, m/s')

#plt.subplot(7, 2, 12)
#plt.plot((np.arange(len(std_err_2g0r[5, :]))+1)*T, std_err_2g0r[5, :].T)
##plt.plot((np.arange(len(std_err_5g0[5, :]))+1)*T, std_err_5g0[5, :].T)
##plt.plot((np.arange(len(std_err_8g0[5, :]))+1)*T, std_err_8g0[5, :].T)
#plt.grid(True)
#plt.xlabel('Time,s')
#plt.ylabel('std_vz, m/s')

##plt.subplot(7, 2, 13)
##plt.plot((np.arange(len(std_err_2g[6, :]))+1)*T, std_err_2g[6, :].T)
##plt.plot((np.arange(len(std_err_5g[6, :]))+1)*T, std_err_5g[6, :].T)
##plt.plot((np.arange(len(std_err_8g[6, :]))+1)*T, std_err_8g[6, :].T)
##plt.grid(True)
##plt.xlabel('Time,s')
##plt.ylabel('std_w, m/s')

#plt.subplot(7, 2, 14)
#plt.plot((np.arange(len(std_err_2g0r[6, :]))+1)*T, std_err_2g0r[6, :].T)
##plt.plot((np.arange(len(std_err_5g0[6, :]))+1)*T, std_err_5g0[6, :].T)
##plt.plot((np.arange(len(std_err_8g0[6, :]))+1)*T, std_err_8g0[6, :].T)
#plt.grid(True)
#plt.xlabel('Time,s')
#plt.ylabel('std_w, m/s')

##plt.subplots_adjust(wspace=8.0, hspace=0.7)
#plt.show()
