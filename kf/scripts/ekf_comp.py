#!/usr/bin/python3

import numpy as np
import math
import filterpy.kalman
import matplotlib.pyplot as plt
import estimator as e

### Входные данные ###########################################################################################
# Период поступления данных
T = 6

# Ошибки процесса
process_var = 0.01
process_var_w = 0.000001

# Ошибки измерений
meas_std = 30
velo_std = 3

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

R = np.diag([meas_std*meas_std, meas_std*meas_std, meas_std*meas_std])
Rvel = np.diag([velo_std*velo_std, velo_std*velo_std, velo_std*velo_std])

# Матрица ошибок измерения
R = np.diag([meas_std*meas_std, meas_std*meas_std, meas_std*meas_std])

# Векторы входных данных
initialState2gr = np.array([30000., -200., 0., 0., 0., 0., w2g_r])#radian
initialState5gr = np.array([30000., -200., 0., 0., 0., 0., w5g_r])#radian
initialState8gr = np.array([30000., -200., 0., 0., 0., 0., w8g_r])#radian

Hp = np.array([[1, 0, 0, 0, 0, 0, 0],
               [0, 0, 1, 0, 0, 0, 0],
               [0, 0, 0, 0, 1, 0, 0]])
Hv = np.array([[0, 1, 0, 0, 0, 0, 0],
               [0, 0, 0, 1, 0, 0, 0],
               [0, 0, 0, 0, 0, 1, 0]])

P0  = Hp.T@R@Hp + Hv.T@Rvel@Hv;

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

### Фильтрация filterpy #####################################################################################

dim_x = initialState2gr.shape[0];
dim_z = Zn2g0r.shape[0];

def fx(x, dt):
    return e.stateModel_CTx(x,dt)

print("dim_x="+str(dim_x)+" dim_z="+str(dim_z))

ekf0 = filterpy.kalman.ExtendedKalmanFilter(dim_x,dim_z)
ekf0.R=R
ekf0.Q=Q
ekf0.H=Hp
ekf0.P=P0

ekf1 = e.BindEKFE_xyz_ct(initialState2gr, P0, Q, R)

est0 = np.zeros((initialState2gr.shape[0], Zn2g0r.shape[1]-1))
E2g0r = np.zeros((initialState2gr.shape[0], Zn2g0r.shape[1]-1))

for col in range(E2g0r.shape[1]):
    z = Zn2g0r[:, col + 1]
    xp = ekf1.predict(T)
    m1 = np.array([z[0], z[1], z[2]])
    xc = ekf1.correct(m1.T)
    E2g0r[:, col] = np.squeeze(xc[:])

    ekf0.predict_update(Zn2g0r[:, col + 1],e.measureModel_XwXx,e.measureModel_XwXx)

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

plt.show()


