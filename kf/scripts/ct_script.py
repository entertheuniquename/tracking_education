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

### Входные данные ###################################################
# Период поступления данных
T = 0.2#6
#print("T: "+str(T))

# Колличество измерений в наборе данных
amount = 162#100
#print("amount: "+str(amount))

#Вероятность правильного обнаружения
Ptd = 0.9
#print("Ptd: "+str(Ptd))

#Колличество пропусков в наборе
pass_am = round(amount*(1-Ptd))
#print("pass_am: "+str(pass_am))

# Угловая скорость на развороте в радианах
w2g_r = 0.098
#print("w2g_r: "+str(w2g_r))

# Угловая скорость на развороте в градусах
w2g_d = w2g_r*180/math.pi
#print("w2g_d: "+str(w2g_d))

# Матрица ошибок процесса
process_var = 0.5#1
Q0 = np.diag([process_var, process_var, process_var])
G = np.array([[T**2/2, 0,      0     ],
              [T,      0,      0     ],
              [0,      T**2/2, 0     ],
              [0,      T,      0     ],
              [0,      0,      T**2/2],
              [0,      0,      T     ],
              [0,      0,      0     ]])

Q = G@Q0@G.T
#print("Q:")
#print(Q)

# Матрица ошибок измерения
meas_var = 1#300
velo_var = 0.1#30

R = np.diag([meas_var*meas_var, meas_var*meas_var, meas_var*meas_var])
Rvel = np.diag([velo_var*velo_var, velo_var*velo_var, velo_var*velo_var])

Rpol = np.diag([1e-4, 1e-4, 1.0])

# Вектор входных данных 1
initialState = np.array([40., 200., 0., 0., 0., 0., 0.])
initialState = initialState[:, np.newaxis]
#print("initialState:")
#print(initialState)

# Вектор входных данных 2
initialState2g = np.array([40., 200., 0., 0., 0., 0., 0.])
initialState2g = initialState2g[:, np.newaxis]
#print("initialState2g:")
#print(initialState2g)

### test #############################################################

#print("initialState2g:")
#print(initialState2g)

initialState2g_ = initialState2g[:-1,:]

#print("initialState2g_:")
#print(initialState2g_)

### Создание наборов данных ##########################################

# Функция создания набора данных
def make_true_data(x0, am, dt, w):
    # Рассчёт колличества измерений на участке разворота
    pointAm2g = round(180/(w*dt));
    print("pointAm2g: "+str(pointAm2g))

    # Рассчёт колличества измерений для каждрого из прямых участков
    pointAm2g_ = round((am-pointAm2g)/2)
    print("pointAm2g_: "+str(pointAm2g_))

    print("check common points amount:"+str(pointAm2g+2*pointAm2g_))

    # Создание обнулённой матрицы нужного размера
    X = np.zeros((x0.shape[0], amount))
    #print("X:")
    #print(X)

    # Запись первого значения
    X[:, 0] = x0.T
    #print("X:")
    #print(X)

    # Цикл создания первого прямолинейного участка
    for i in range(pointAm2g_):
        #print(">>>>>>>>>> "+str(i)+"->"+str(i+1)+" >>>>>>>>>>[1]")
        # Копирование очередного столбца данных
        xa = np.copy(X[:, i])
        #print("xa:")
        #print(xa)

        # Обнуление угловой скорости
        xa[-1] = 0
        #print("xa':")
        #print(xa)

        # Экстраполяция данных на один шаг
        xx = e.stateModel_CTx(xa,dt)
        #print("xx:")
        #print(xx)

        # Превращение строки в столбец и добавление в набор
        X[:, i+1] = xx.flatten()
        #print("X[:, "+str(i+1)+"]:")
        #print(X[:, i+1])

    # Цикл создания участка разворота
    for i in range(pointAm2g):
        #print(">>>>>>>>>> "+str(pointAm2g_+i)+"->"+str(pointAm2g_+i+1)+" >>>>>>>>>>[2]")
        xx = e.stateModel_CTx(np.copy(X[:, pointAm2g_+i]),dt)
        xx[-1] = w*math.pi/180
        X[:, pointAm2g_+i+1] = xx.flatten()
        #print("X[:, "+str(pointAm2g_+i+1)+"]:")
        #print(X[:, pointAm2g_+i+1])

    # Цикл создания первого прямолинейного участка
    for i in range(pointAm2g_-1):
        #print(">>>>>>>>>> "+str(pointAm2g_+pointAm2g+i)+"->"+str(pointAm2g_+pointAm2g+i+1)+" >>>>>>>>>>[3]")
        # Копирование очередного столбца данных
        xa = np.copy(X[:, pointAm2g_+pointAm2g+i])
        #print("xa:")
        #print(xa)

        xa[-1] = 0

        # Экстраполяция данных на один шаг
        xx = e.stateModel_CTx(xa,dt)
        #print("xx:")
        #print(xx)

        # Превращение строки в столбец и добавление в набор
        X[:, pointAm2g_+pointAm2g+i+1] = xx.flatten()
        #print("X[:, "+str(pointAm2g_+pointAm2g+i+1)+"]:")
        #print(X[:, pointAm2g_+pointAm2g+i+1])
    return X

# Создание набора данных 1
#X=make_true_data(initialState, amount, T, w2g_d)
#print("X:")
#print(X)

# Создание набора данных 2
X2g=make_true_data(initialState2g, amount, T, w2g_d)
#print("X2g:")
#print(X2g)

### test #############################################################

#print("X:"+str(X[0,0])+"->"+str(X[0,1])+"->"+str(X[0,2])+"->"+str(X[0,3])+"->"+str(X[0,4])+"->...")

### Добавление к наборам данных ошибок процесса ######################

def add_process_noise(X,Var):
    Xn = X + np.sqrt(Var) @ np.random.normal(loc=0, scale=1.0, size=(X.shape[0], X.shape[1]))
    return Xn

Xn2g = add_process_noise(X2g,Q)

### Получение из наборов данных измерений и добавление к ним шцмов ###
# Функция получения измерений
def make_meas(X, R):
    # Получение обнуленного набора измерений
    Z = np.zeros((R.shape[0], X.shape[1]))
    #print("Z:")
    #print(Z)

    # Цикл по заполнению набора измерений зашумлёнными значениями
    for i in range(Z.shape[1]):
        #print(".......... "+str(i)+" ..........")
        # Получение очередного значения набора данных
        zz0 = np.copy(X[:, i])
        #print("zz0:")
        #print(zz0)

        # Обрезание угловой скорости для использования модени измерения
        zz1 = zz0[:-1]
        #print("zz1:")
        #print(zz1)

        # Получение очередного измерения
        zz = e.measureModel_XXx(zz1)
        #print("zz:")
        #print(zz)

        # Запись полученного измерения в массив измерений
        Z[:, i] = zz.flatten()
        #print("Z[:, "+str(i)+"]:")
        #print(Z[:, i])

    # Добавления шумов к набору измерений
    Zn = Z + np.sqrt(R) @ np.random.normal(loc=0, scale=math.sqrt(1.0), size=(Z.shape[0], Z.shape[1]))
    return Zn

Zn2g = make_meas(Xn2g, R)

# Имитация отметки
def make_tu(TP,M):
    tu = (TP,M)
    return tu

### Фильтрация XYZ_KFE #######################################################

# Функция фильтрации набора данных
def stepKFE(Zn, dt):
    rnd = np.random.randint(0,98,pass_am)
    time = 0
    track = e.BindTrackKFE(make_tu(time,Zn[:, 0][:, np.newaxis]))
    est = np.zeros((6, Zn.shape[1]-1))#6 - bad! not dt
    for col in range(Zn.shape[1]-1):
        time = time+dt
        z = Zn[:, col+1]
        if col in rnd:
            ee = track.step(time)
        else:
            ee = track.step(make_tu(time,z))
        est[:, col] = np.squeeze(ee[:])
    return est

#est2g=stepKFE(Zn2g, T)

### Фильтрация XYZ_KFE_CT #######################################################

# Функция фильтрации набора данных
def stepKFE_CT(Zn, dt):
    rnd = np.random.randint(0,98,pass_am)
    time = 0
    track = e.BindTrackKFE_CT(make_tu(time,Zn[:, 0][:, np.newaxis]))
    est = np.zeros((7, Zn.shape[1]-1))#7 - bad! not dt
    for col in range(Zn.shape[1]-1):
        time = time+dt
        z = Zn[:, col+1]
        if col in rnd:
            ee = track.step(time)
        else:
            ee = track.step(make_tu(time,z))
        est[:, col] = np.squeeze(ee[:])
    return est

#est2g_ct=stepKFE_CT(Zn2g, T)
#print("est2g_ct:")
#print(est2g_ct)

### Фильтрация XYZ_EKFE_CV #######################################################

# Функция фильтрации набора данных
def stepEKFE_xyz_cv(Zn, dt):
    rnd = np.random.randint(0,98,pass_am)
    time = 0
    track = e.BindTrackEKFE_xyz_cv(make_tu(time,Zn[:, 0][:, np.newaxis]))
    est = np.zeros((6, Zn.shape[1]-1))#6 - bad! not dt
    for col in range(Zn.shape[1]-1):
        time = time+dt
        z = Zn[:, col+1]
        if col in rnd:
            ee = track.step(time)
        else:
            ee = track.step(make_tu(time,z))
        est[:, col] = np.squeeze(ee[:])
    return est

#est_ekfe_xyz_cv_2g=stepEKFE_xyz_cv(Zn2g, T)

### Фильтрация XYZ_EKFE_CT #######################################################

# Функция фильтрации набора данных
def stepEKFE_xyz_ct(Zn, dt):
    rnd = np.random.randint(0,98,pass_am)
    time = 0
    track = e.BindTrackEKFE_xyz_ct(make_tu(time,Zn[:, 0][:, np.newaxis]))
    est = np.zeros((7, Zn.shape[1]-1))#6 - bad! not dt
    for col in range(Zn.shape[1]-1):
        time = time+dt
        z = Zn[:, col+1]
        if col in rnd:
            ee = track.step(time)
        else:
            ee = track.step(make_tu(time,z))
        est[:, col] = np.squeeze(ee[:])
    return est

est_ekfe_xyz_ct_2g=stepEKFE_xyz_ct(Zn2g, T)
#print("est_ekfe_xyz_ct_2g:")
#print(est_ekfe_xyz_ct_2g)

### Отрисовка графиков для сглаживания ###############################

fig = plt.figure(figsize=(9,4))
ax1 = fig.add_subplot(1,1,1)

ax1.plot(initialState[0], initialState[2], label='First point', linestyle='', marker='o')
#ax1.plot(120000, 0.001, label='.', linestyle='', marker='.')
#ax1.plot(X[0, :], X[2, :], label='X')
ax1.plot(X2g[0, :], X2g[2, :], label='X2g - true')
ax1.plot(Xn2g[0, :], Xn2g[2, :], label='X2g - true+noise', linestyle='', marker='.')
ax1.plot(Zn2g[0, :], Zn2g[1, :], label='X2g - measurement', linestyle='', marker='+')
#ax1.plot(est2g[0, :], est2g[2, :], label='X2g - estimates')
#ax1.plot(est2g_ct[0, :], est2g_ct[2, :], label='X2g_ct - estimates')
#ax1.plot(est_ekfe_xyz_cv_2g[0, :], est_ekfe_xyz_cv_2g[2, :], label='est_ekfe_xyz_cv_2g - estimates')
ax1.plot(est_ekfe_xyz_ct_2g[0, :], est_ekfe_xyz_ct_2g[2, :], label='est_ekfe_xyz_ct_2g - estimates')
ax1.set_title("[x,vx,y,vy,z,vz,w]")
ax1.set_xlabel('x,met.')
ax1.set_ylabel('y,met.')
ax1.grid(True)

plt.show()
