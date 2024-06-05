#!/usr/bin/python3

import math
import numpy as np
import filterpy
from filterpy.kalman import ExtendedKalmanFilter
from filterpy.common import Q_discrete_white_noise
from numpy.random import randn
from numpy import eye, array, asarray
import matplotlib.pyplot as plt
import estimator as e

def jacobian_ct(x_in,t_in):
    """ compute Jacobian of F matrix at x_in """

    x = x_in[0]
    vx = x_in[1]
    y = x_in[2]
    vy = x_in[3]
    z = x_in[4]
    vz = x_in[5]
    w = x_in[6]
    t = t_in

    if w==0:
        w=0.000001

    J00 = 1.;
    J01 = math.sin(w*t)/w;
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

    J = array([[J00, J01, J02, J03, J04, J05, J06],
               [J10, J11, J12, J13, J14, J15, J16],
               [J20, J21, J22, J23, J24, J25, J26],
               [J30, J31, J32, J33, J34, J35, J36],
               [J40, J41, J42, J43, J44, J45, J46],
               [J50, J51, J52, J53, J54, J55, J56],
               [J60, J61, J62, J63, J64, J65, J66]])
    return J

def F_Jacobian(x,t):
    """ compute Jacobian of H matrix at x """

    ret = jacobian_ct(x,t)

    return ret

def H_Jacobian(x):
    """ compute Jacobian of H matrix at x """

    ret = np.array([[1,0,0,0,0,0,0],
                    [0,0,1,0,0,0,0],
                    [0,0,0,0,1,0,0]])

    return ret

def hx(x):
    """ compute measurement for slant range that
    would correspond to state x.
    """
    #print("hx0")
    #print("x")
    #print(x)
    fun = np.array([[1,0,0,0,0,0,0],
                    [0,0,1,0,0,0,0],
                    [0,0,0,0,1,0,0]])

    ret = np.dot(fun,x)
    return ret

#################################################################################################################
#print("filterpy: ExtendedKalmanFilter")
#################################################################################################################
#print("=== make data(oldstyle) ===")
# Период поступления данных
T = 6

# Ошибки процесса
process_var = 0.01
process_var_w = 0.000001

# Ошибки измерений
meas_std = 300
velo_std = 30

# Угловая скорость на развороте в радианах
w2g_r = 0.098
w5g_r = 0.245
w8g_r = 0.392

# Угловая скорость на развороте в градусах
w2g_d = w2g_r*180/math.pi
w5g_d = w5g_r*180/math.pi
w8g_d = w8g_r*180/math.pi
#print("w2g_d: "+str(w2g_d))
#print("w5g_d: "+str(w5g_d))
#print("w8g_d: "+str(w8g_d))

# Матрица ошибок процесса
Q0 = np.diag([process_var, process_var, process_var, process_var_w])
G = np.array([[T**2/2, 0,      0     , 0],
              [T,      0,      0     , 0],
              [0,      T**2/2, 0     , 0],
              [0,      T,      0     , 0],
              [0,      0,      T**2/2, 0],
              [0,      0,      T     , 0],
              [0,      0,      0     , T]])

##print("Q0:")
##print(Q0)

Q = G@Q0@G.T

##print("Q:")
##print(Q)

# Матрица ошибок измерения
R = np.diag([meas_std*meas_std, meas_std*meas_std, meas_std*meas_std])
Rvel = np.diag([velo_std*velo_std, velo_std*velo_std, velo_std*velo_std])

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

X2g0r=make_true_data_round(initialState2gr, 500, T)
X5g0r=make_true_data_round(initialState5gr, 500, T)
X8g0r=make_true_data_round(initialState8gr, 500, T)

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
#################################################################################################################
#tests
xx = H_Jacobian(Xn2g0r)
print("H_Jacobian:")
print(xx)
zz = hx(Xn2g0r[:,0])
print("hx:")
print(zz)
#################################################################################################################
x0 = np.array([initialState2gr[0],0,initialState2gr[2],0,initialState2gr[4],0,0])
#x0 = Xn2g0r[:,0]#!!!
#print("x0["+str(x0.shape[0])+"]:")
#print(x0)
step_amount = Zn2g0r.shape[1]
#################################################################################################################
#print("=== filter(1) ===")
def step1(x0,P0,Q,R,Z,amount,T):
#    print("=== STEP-1 ===")
#    print("[0] x0["+str(x0.shape[0])+"]")
#    print(x0)
#    print("[0] P0["+str(P0.shape[0])+","+str(P0.shape[1])+"]")
#    print(P0)
#    print("[0] Q["+str(Q.shape[0])+","+str(Q.shape[1])+"]")
#    print(Q)
#    print("[0] R["+str(R.shape[0])+","+str(R.shape[1])+"]")
#    print(R)
    ekf = e.BindEKFE_xyz_ct(x0, P0, Q, R)
    xs = []
    for i in range(amount-1):
        z = Z[:, i + 1]
        xp = ekf.predict(T)
#        if i<3:
#            print("[0] it "+str(i))
#            print("[0] pred.xp["+str(xp.shape[0])+"]")
#            print(xp)
#            #print("[0] pred.P["+str(rk0.P.shape[0])+","+str(rk0.P.shape[1])+"]")
#            #print(rk0.P)
        m1 = np.array([z[0], z[1], z[2]])
        xc = ekf.correct(m1.T)
#        if i<3:
#            print("[0] corr.xc["+str(xc.shape[0])+"]")
#            print(xc)
        xs.append(np.squeeze(xc[:]))
    return asarray(xs)
#print("--------------")
xs = step1(x0,P0,Q,R,Zn2g0r,step_amount,T)
#print("xs["+str(xs.shape[0])+","+str(xs.shape[1])+"]")
#################################################################################################################
#print("=== filter(0) ===")
def step0(x0,P0,Q,R,Z,amount,T):
#    print("=== STEP-0 ===")
#    print("[0] x0["+str(x0.shape[0])+"]")
#    print(x0)
#    print("[0] P0["+str(P0.shape[0])+","+str(P0.shape[1])+"]")
#    print(P0)
#    print("[0] Q["+str(Q.shape[0])+","+str(Q.shape[1])+"]")
#    print(Q)
#    print("[0] R["+str(R.shape[0])+","+str(R.shape[1])+"]")
#    print(R)
    rk0 = ExtendedKalmanFilter(dim_x=7, dim_z=3)
    rk0.x = x0
    rk0.R = R
    rk0.Q = Q
    rk0.P = P0
    xs0, track0 = [], []
    for i in range(amount-1):
        rk0.F = F_Jacobian(rk0.x,T)
        rk0.predict()
#        if i<3:
#            print("[0] it "+str(i))
#            print("[0] pred.x["+str(rk0.x.shape[0])+"]")
#            print(rk0.x)
#            print("[0] pred.P["+str(rk0.P.shape[0])+","+str(rk0.P.shape[1])+"]")
#            print(rk0.P)
        Z1 = Z.transpose()
        z = Z1[i+1,:]
        rk0.H = H_Jacobian(rk0.x)
        rk0.update(z, H_Jacobian, hx)
#        if i<3:
#            print("[0] corr.x["+str(rk0.x.shape[0])+"]")
#            print(rk0.x)
#            print("[0] corr.P["+str(rk0.P.shape[0])+","+str(rk0.P.shape[1])+"]")
#            print(rk0.P)
        xs0.append(rk0.x)
    return asarray(xs0)

xs0 = step0(x0,P0,Q,R,Zn2g0r,step_amount,T)
#print("xs0["+str(xs0.shape[0])+","+str(xs0.shape[1])+"]")
#print("--------------")
#################################################################################################################

#################################################################################################################
#print("=== graphics ===")

def print_step(Z,X,T):
    #x0 = X[:,0]
    x0 = np.array([X[0,0],0,X[2,0],0,X[4,0],0,0])

    xs0 = step0(x0,P0,Q,R,Z,step_amount,T)
    xs = step1(x0,P0,Q,R,Z,step_amount,T)

    time0 = np.arange(0, step_amount*T,T)
    Z0 = Z.transpose()
    X0 = X.transpose()

    fig = plt.figure(figsize=(18,25))

    ax1 = fig.add_subplot(2,2,1)
    ax1.plot(X0[:,2], X0[:,0], label='.',color='black')
    ax1.plot(Z0[:,1], Z0[:,0], label='.', marker='x', color='grey')
    ax1.plot(xs0[:,2], xs0[:,0], label='.', color='green')
    ax1.plot(xs[:,2], xs[:,0], label='.', color='red')
    ax1.set_title(".")
    ax1.set_xlabel('.')
    ax1.set_ylabel('.')
    ax1.grid(True)

    ax2 = fig.add_subplot(3,2,2)
    ax2.plot(time0[1:], X0[1:,0], label='.',color='black')
    ax2.plot(time0[1:], Z0[1:,0], label='.', marker='x', color='grey')
    ax2.plot(time0[1:], xs0[:,0], label='.', color='green')
    ax2.plot(time0[1:], xs[:,0], label='.', color='red')
    ax2.set_title(".")
    ax2.set_xlabel('.')
    ax2.set_ylabel('.')
    ax2.grid(True)

    ax4 = fig.add_subplot(3,2,4)
    ax4.plot(time0[1:], X0[1:,2], label='.',color='black')
    ax4.plot(time0[1:], Z0[1:,1], label='.', marker='x', color='grey')
    ax4.plot(time0[1:], xs0[:,2], label='.', color='green')
    ax4.plot(time0[1:], xs[:,2], label='.', color='red')
    ax4.set_title(".")
    ax4.set_xlabel('.')
    ax4.set_ylabel('.')
    ax4.grid(True)

    ax6 = fig.add_subplot(3,2,6)
    ax6.plot(time0[1:], X0[1:,4], label='.',color='black')
    ax6.plot(time0[1:], Z0[1:,2], label='.', marker='x', color='grey')
    ax6.plot(time0[1:], xs0[:,4], label='.', color='green')
    ax6.plot(time0[1:], xs[:,4], label='.', color='red')
    ax6.set_title(".")
    ax6.set_xlabel('.')
    ax6.set_ylabel('.')
    ax6.grid(True)

    ax3 = fig.add_subplot(2,2,3)
    ax3.plot(time0[1:], X0[1:,6], label='.',color='black')
    ax3.plot(time0[1:], xs0[:,6], label='.', color='green')
    ax3.plot(time0[1:], xs[:,6], label='.', color='red')
    ax3.set_title(".")
    ax3.set_xlabel('.')
    ax3.set_ylabel('.')
    ax3.grid(True)

    plt.show()

print_step(Zn2g0r,Xn2g0r,T)
print_step(Zn5g0r,Xn5g0r,T)
print_step(Zn8g0r,Xn8g0r,T)


#################################################################################################################
#print("=== statistic ===")
def calc_err(X,dt,Q,R,step_function):
    Xn = add_process_noise(X,Q)
    Zn = make_meas(Xn,R)
    amount = Zn.shape[1]
    #x0=Xn[:,0]
    x0 = np.array([Xn[0,0],0,Xn[2,0],0,Xn[4,0],0,0])
    est0 = step_function(x0,P0,Q,R,Zn,amount,dt)
    est = est0.transpose()
    err = est - Xn[:, 1:]
    return err

from tqdm import tqdm

def calc_std_err(X,dt,Q,R,step_function,iterations):
    num_iterations = iterations
    var_err = np.zeros((X.shape[0], X.shape[1]-1))
    for i in tqdm(range(num_iterations)):
        err = calc_err(X,dt,Q,R,step_function)
        var_err += err ** 2

    var_err /= num_iterations
    return np.sqrt(var_err)

#std_err0 = calc_std_err(X2g0r,T,Q,R,step0)
#std_err1 = calc_std_err(X2g0r,T,Q,R,step1)
##################################################################################################################
def print_stat(X,T,Q,R):

    std_err0 = calc_std_err(X,T,Q,R,step0,2000)
    std_err1 = calc_std_err(X,T,Q,R,step1,2000)

    plt.figure(figsize=(20,45))

    plt.subplot(6, 1, 1)
    plt.plot((np.arange(len(std_err0[0, :]))+1)*T, std_err0[0, :].T, label='.',color='green')
    plt.plot((np.arange(len(std_err1[0, :]))+1)*T, std_err1[0, :].T, label='.',color='red')
    plt.grid(True)
    plt.xlabel('Time,s')
    plt.ylabel('std_x, met')

    plt.subplot(6, 1, 2)
    plt.plot((np.arange(len(std_err0[1, :]))+1)*T, std_err0[1, :].T, label='.',color='green')
    plt.plot((np.arange(len(std_err1[1, :]))+1)*T, std_err1[1, :].T, label='.',color='red')
    plt.grid(True)
    plt.xlabel('Time,s')
    plt.ylabel('std_vx, m/s')

    plt.subplot(6, 1, 3)
    plt.plot((np.arange(len(std_err0[2, :]))+1)*T, std_err0[2, :].T, label='.',color='green')
    plt.plot((np.arange(len(std_err1[2, :]))+1)*T, std_err1[2, :].T, label='.',color='red')
    plt.grid(True)
    plt.xlabel('Time,s')
    plt.ylabel('std_y, met')

    plt.subplot(6, 1, 4)
    plt.plot((np.arange(len(std_err0[4, :]))+1)*T, std_err0[4, :].T, label='.',color='green')
    plt.plot((np.arange(len(std_err1[4, :]))+1)*T, std_err1[4, :].T, label='.',color='red')
    plt.grid(True)
    plt.xlabel('Time,s')
    plt.ylabel('std_z, met')

    plt.subplot(6, 1, 5)
    plt.plot((np.arange(len(std_err0[5, :]))+1)*T, std_err0[5, :].T, label='.',color='green')
    plt.plot((np.arange(len(std_err1[5, :]))+1)*T, std_err1[5, :].T, label='.',color='red')
    plt.grid(True)
    plt.xlabel('Time,s')
    plt.ylabel('std_vz, m/s')

    plt.subplot(6, 1, 6)
    plt.plot((np.arange(len(std_err0[6, :]))+1)*T, std_err0[6, :].T, label='.',color='green')
    plt.plot((np.arange(len(std_err1[6, :]))+1)*T, std_err1[6, :].T, label='.',color='red')
    plt.grid(True)
    plt.xlabel('Time,s')
    plt.ylabel('std_w, m/s')

    plt.show()

print_stat(X2g0r,T,Q,R)
print_stat(X5g0r,T,Q,R)
print_stat(X8g0r,T,Q,R)
