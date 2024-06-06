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

def predict_xx(x,fx):
    """
    Predicts the next state of X. If you need to
    compute the next state yourself, override this function. You would
    need to do this, for example, if the usual Taylor expansion to
    generate F is not providing accurate results for you.
    """
    return fx(x)


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

def fx(x,t):
    return e.stateModel_CTx(x,t)

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

# Угловая скорость на развороте в радианах
w2g_r = 0.098
w5g_r = 0.245
w8g_r = 0.392
# Угловая скорость на развороте в градусах
w2g_d = w2g_r*180./math.pi
w5g_d = w5g_r*180./math.pi
w8g_d = w8g_r*180./math.pi
#print("w2g_d: "+str(w2g_d))
#print("w5g_d: "+str(w5g_d))
#print("w8g_d: "+str(w8g_d))

# Ошибки процесса
process_var = 0.01
process_var_w = 0.000001

# Ошибки измерений
meas_std = 300.
velo_std = 30.
w_std = w8g_r


# Матрица ошибок процесса
Q0 = np.diag([process_var, process_var, process_var, process_var_w])
G = np.array([[T**2/2, 0,      0     , 0],
              [T,      0,      0     , 0],
              [0,      T**2/2, 0     , 0],
              [0,      T,      0     , 0],
              [0,      0,      T**2/2, 0],
              [0,      0,      T     , 0],
              [0,      0,      0     , T]])
Q = G@Q0@G.T

# Матрица ошибок измерения
R = np.diag([meas_std*meas_std, meas_std*meas_std, meas_std*meas_std])
Rvel = np.diag([velo_std*velo_std, velo_std*velo_std, velo_std*velo_std])

# Векторы входных данных
initialState2gr = np.array([30000., -200., 0., 0., 0., 0., w2g_r])#radian
initialState5gr = np.array([30000., -200., 0., 0., 0., 0., w5g_r])#radian
initialState8gr = np.array([30000., -200., 0., 0., 0., 0., w8g_r])#radian

Hp = np.array([[1., 0., 0., 0., 0., 0., 0.],
               [0., 0., 1., 0., 0., 0., 0.],
               [0., 0., 0., 0., 1., 0., 0.]])
Hv = np.array([[0., 1., 0., 0., 0., 0., 0.],
               [0., 0., 0., 1., 0., 0., 0.],
               [0., 0., 0., 0., 0., 1., 0.]])

P0  = Hp.T@R@Hp + Hv.T@Rvel@Hv;
P0[6,6] = w_std*w_std
### Создание наборов данных ##################################################################################
# Функция создания набора данных: по кругу
def make_true_data_round(x0_in, am_in, dt_in):
    # Создание обнулённой матрицы нужного размера
    X = np.zeros((x0_in.shape[0], am_in))
    # Запись первого значения
    X[:, 0] = x0_in.T
    # Цикл создания участка разворота
    for i in range(am_in-1):
        xx = e.stateModel_CTx(np.copy(X[:, i]),dt_in)
        X[:, i+1] = xx.flatten()
    return X

X2g0r=make_true_data_round(initialState2gr, 200, T)
X5g0r=make_true_data_round(initialState5gr, 200, T)
X8g0r=make_true_data_round(initialState8gr, 200, T)

### Добавление к наборам данных ошибок процесса ##############################################################
def add_process_noise(X_in,Var_in):
    Xn = X_in + np.sqrt(Var_in) @ np.random.normal(loc=0, scale=1.0, size=(X_in.shape[0], X_in.shape[1]))
    return Xn

Xn2g0r = add_process_noise(X2g0r,Q)
Xn5g0r = add_process_noise(X5g0r,Q)
Xn8g0r = add_process_noise(X8g0r,Q)

### Получение из наборов данных измерений и добавление к ним шцмов ###########################################
# Функция получения измерений
def make_meas(X_in, R_in):
    # Получение обнуленного набора измерений
    Z = np.zeros((R_in.shape[0], X_in.shape[1]))
    # Цикл по заполнению набора измерений зашумлёнными значениями
    for i in range(Z.shape[1]):
        # Получение очередного значения набора данных
        zz = e.measureModel_XwXx(np.copy(X_in[:, i]))
        Z[:, i] = zz.flatten()
    # Добавления шумов к набору измерений
    Zn = Z + np.sqrt(R_in) @ np.random.normal(loc=0, scale=math.sqrt(1.0), size=(Z.shape[0], Z.shape[1]))
    return Zn

Zn2g0r = make_meas(Xn2g0r, R)
Zn5g0r = make_meas(Xn5g0r, R)
Zn8g0r = make_meas(Xn8g0r, R)

#################################################################################################################
#tests
xx = H_Jacobian(Xn2g0r)
#print("xx:")
#print(xx)
zz = hx(Xn2g0r[:,0])
#print("zz:")
#print(zz)
#################################################################################################################
def step_ekf(x0_in,P0_in,Q_in,R_in,Z_in,amount_in,T_in):
    ekf = e.BindEKFE_xyz_ct(x0_in, P0_in, Q_in, R_in)
    xs = []
    for i in range(amount_in-1):
        z = Z_in[:, i + 1]
        xp = ekf.predict(T_in)
        m1 = np.array([z[0], z[1], z[2]])
        xc = ekf.correct(m1.T)
        xs.append(np.squeeze(xc[:]))
    return asarray(xs)
#################################################################################################################
def step_filterpy_ekf(x0_in,P0_in,Q_in,R_in,Z_in,amount_in,T_in):
    rk0 = ExtendedKalmanFilter(dim_x=7, dim_z=3)
    rk0.x = x0_in
    rk0.R = R_in
    rk0.Q = Q_in
    rk0.P = P0_in
    xs0, track0 = [], []
    for i in range(amount_in-1):
        #xx = fx(rk0.x,T)# - если закомментировать в фильтре predict_x
        #rk0.x = np.array([xx[0,0],xx[1,0],xx[2,0],xx[3,0],xx[4,0],xx[5,0],xx[6,0]])#BAD

        rk0.F = F_Jacobian(rk0.x,T)
        rk0.predict()
        Z1 = Z_in.transpose()
        z = Z1[i+1,:]
        rk0.H = H_Jacobian(rk0.x)
        rk0.update(z, H_Jacobian, hx)
        xs0.append(rk0.x)
    return asarray(xs0)
#################################################################################################################
def steps(Z_in,X_in,T_in):

    step_amount = Z_in.shape[1]
    #x0 = X_in[:,0]
    x0 = np.array([X_in[0,0],0,X_in[2,0],0,X_in[4,0],0,0])

    xs0 = step_filterpy_ekf(x0,P0,Q,R,Z_in,step_amount,T_in)
    xs = step_ekf(x0,P0,Q,R,Z_in,step_amount,T_in)

    time0 = np.arange(0, step_amount*T,T)
    Z0 = Z_in.transpose()
    X0 = X_in.transpose()

    fig = plt.figure(figsize=(18,25))

    ax1 = fig.add_subplot(2,2,1)
    ax1.plot(xs0[:,0], xs0[:,2], label='.', color='green')
    ax1.plot(xs[:,0], xs[:,2], label='.', color='red')
    ax1.plot(Z0[:,0], Z0[:,1], label='.', marker='x', color='grey')
    ax1.plot(X0[:,0], X0[:,2], label='.',color='black')
    ax1.set_title("x(y) real(black) measurements(grey) filterpy-ekf(green) my-ekf(red)")
    ax1.set_xlabel('x.m')
    ax1.set_ylabel('y.m')
    ax1.grid(True)

    ax2 = fig.add_subplot(3,2,2)
    ax2.plot(X0[1:,0], label='.',color='black')
    ax2.plot(Z0[1:,0], label='.', marker='x', color='grey')
    ax2.plot(xs0[:,0], label='.', color='green')
    ax2.plot(xs[:,0], label='.', color='red')
    ax2.set_title("x")
    ax2.set_xlabel('x.m')
    ax2.set_ylabel('amount')
    ax2.grid(True)

    ax4 = fig.add_subplot(3,2,4)
    ax4.plot(X0[1:,2], label='.',color='black')
    ax4.plot(Z0[1:,1], label='.', marker='x', color='grey')
    ax4.plot(xs0[:,2], label='.', color='green')
    ax4.plot(xs[:,2], label='.', color='red')
    ax4.set_title("y")
    ax4.set_xlabel('y.m')
    ax4.set_ylabel('amount')
    ax4.grid(True)

    ax6 = fig.add_subplot(3,2,6)
    ax6.plot(X0[1:,4], label='.',color='black')
    ax6.plot(Z0[1:,2], label='.', marker='x', color='grey')
    ax6.plot(xs0[:,4], label='.', color='green')
    ax6.plot(xs[:,4], label='.', color='red')
    ax6.set_title("z")
    ax6.set_xlabel('z.m')
    ax6.set_ylabel('amount')
    ax6.grid(True)

    ax3 = fig.add_subplot(2,2,3)
    ax3.plot(X0[1:,6], label='.',color='black')
    ax3.plot(xs0[:,6], label='.', color='green')
    ax3.plot(xs[:,6], label='.', color='red')
    ax3.set_title("w")
    ax3.set_xlabel('w.rad')
    ax3.set_ylabel('amount')
    ax3.grid(True)

    #plt.show()

    return xs, xs0

steps(Zn2g0r,Xn2g0r,T)
steps(Zn5g0r,Xn5g0r,T)
steps(Zn8g0r,Xn8g0r,T)

plt.show()

#################################################################################################################
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
##################################################################################################################
def print_stat(X,T,Q,R):

    std_err0 = calc_std_err(X,T,Q,R,step_filterpy_ekf,2000)
    std_err1 = calc_std_err(X,T,Q,R,step_ekf,2000)

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

    #plt.show()

#print_stat(X2g0r,T,Q,R)
#print_stat(X5g0r,T,Q,R)
#print_stat(X8g0r,T,Q,R)

plt.show()
