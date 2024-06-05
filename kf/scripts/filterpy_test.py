#!/usr/bin/python3
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import ExtendedKalmanFilter
from numpy import eye, array, asarray
import numpy as np
from numpy.random import randn
import math
from math import sqrt
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
vel_std = 3

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
Rvel = np.diag([vel_std*vel_std, vel_std*vel_std, vel_std*vel_std])

Hp = np.array([[1, 0, 0, 0, 0, 0, 0],
               [0, 0, 1, 0, 0, 0, 0],
               [0, 0, 0, 0, 1, 0, 0]])
Hv = np.array([[0, 1, 0, 0, 0, 0, 0],
               [0, 0, 0, 1, 0, 0, 0],
               [0, 0, 0, 0, 0, 1, 0]])

P0  = Hp.T@R@Hp + Hv.T@Rvel@Hv;

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

### Фильтр ###################################################################################################

def HJacobian_at(x):
    return e.measureModel_XwXx(x)

def HJacobian_at2(xxx):
    print("xxx:")
    print(xxx)
    x = xxx[0]
    vx = xxx[1]
    y = xxx[2]
    vy = xxx[3]
    z = xxx[4]
    vz = xxx[5]
    w = xxx[6]

    t = 6

    if w==0:
        w=0.0000001;

    J00 = 1.
    J01 = math.sin(w*t)/w
    J02 = 0.;
    J03 = (math.cos(w*t)-1)/w;
    J04 = 0.;
    J05 = 0.;
    J06 = (t*vx*math.cos(w*t)/w) - (t*vy*math.sin(w*t)/w) - (vx*math.sin(w*t)/math.pow(w,2)) - (vy*(math.cos(w*t)-1)/math.pow(w,2));

    J10 = 0.;
    J11 = math.cos(w*t);
    J12 = 0.;
    J13 = -math.sin(w*t);
    J14 = 0.;
    J15 = 0.;
    J16 = -t*vx*math.sin(w*t) - t*vy*math.cos(w*t);

    J20 = 0.;
    J21 = (1-math.cos(w*t))/w;
    J22 = 1.;
    J23 = math.sin(w*t)/w;
    J24 = 0.;
    J25 = 0.;
    J26 = (t*vx*math.sin(w*t)/w) + (t*vy*math.cos(w*t)/w) - (vx*(1-math.cos(w*t))/math.pow(w,2)) - (vy*math.sin(w*t)/math.pow(w,2));

    J30 = 0.;
    J31 = math.sin(w*t);
    J32 = 0.;
    J33 = math.cos(w*t);
    J34 = 0.;
    J35 = 0.;
    J36 = t*vx*math.cos(w*t) - t*vy*math.sin(w*t);

    J40 = 0.;
    J41 = 0.;
    J42 = 0.;
    J43 = 0.;
    J44 = 1.;
    J45 = t;
    J46 = 0.;

    J50 = 0.;
    J51 = 0.;
    J52 = 0.;
    J53 = 0.;
    J54 = 0.;
    J55 = 1.;
    J56 = 0.;

    J60 = 0.;
    J61 = 0.;
    J62 = 0.;
    J63 = 0.;
    J64 = 0.;
    J65 = 0.;
    J66 = 1.;

    J = np.array([[J00, J01, J02, J03, J04, J05, J06],
                  [J10, J11, J12, J13, J14, J15, J16],
                  [J20, J21, J22, J23, J24, J25, J26],
                  [J30, J31, J32, J33, J34, J35, J36],
                  [J40, J41, J42, J43, J44, J45, J46],
                  [J50, J51, J52, J53, J54, J55, J56],
                  [J60, J61, J62, J63, J64, J65, J66]])
    return J

JJ = HJacobian_at2(X2g0r[:,0])

print("JJ:")
print(JJ)

def hx(x):
    return e.measureModel_XwXx(x)

def fx(x):
    return e.stateModel_CTx(x)

rk = ExtendedKalmanFilter(dim_x=7, dim_z=3)

rk.x = X2g0r[:,0]

print("rk.x:")
print(rk.x)
print("HJacobian_at(rk.x):")
print(HJacobian_at(rk.x))

rk.F = HJacobian_at2

print("rk.F:")
print(rk.F)

range_std = 5. # meters

print("range_std: "+str(range_std))

rk.R = np.diag([range_std**2])

print("rk.R:")
print(rk.R)

rk.Q = Q

print("rk.Q:")
print(rk.Q)

rk.P = P0

print("rk.P:")
print(rk.P)

xs, track = [], []

print("xs:")
print(xs)
print("track:")
print(track)

dt=T

print("<"+str(int(20/dt))+">")
for i in range(int(20/dt)):
    print("["+str(i)+"]")
    z = Zn2g0r[:,i]
    print("    z:"+str(z))
    track.append(z)
    print("    track:")
    rk.update(array([z]), HJacobian_at, hx)
    print("    rk:")
    xs.append(rk.x)
    print("    ->xc:"+str(rk.x))
    rk.predict()
print("<~>")

xs = asarray(xs)
track = asarray(track)

print("xs:")
print(xs)
print("track:")
print(track)

time = np.arange(0, len(xs)*dt, dt)
#ekf_internal.plot_radar(xs, track, time)

fig1 = plt.figure(figsize=(18,25))

ax11 = fig1.add_subplot(2,2,2)
ax11.plot(track[:, 0], label='track[pos]')
ax11.plot(xs[:, 0], label='xs[pos]')
#ax11.plot(Zn2g0r[0, :], Zn2g0r[1, :], label='Zn2g0r - measurements')
#ax11.plot(E2g0r[0, :], E2g0r[2, :], label='E2g0r - estimations')
#ax11.set_title("2G - [x,y] - w-rad")
#ax11.set_xlabel('x,met.')
#ax11.set_ylabel('y,met.')
ax11.grid(True)

plt.show()
