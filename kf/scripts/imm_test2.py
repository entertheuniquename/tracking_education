#!/usr/bin/python3

print("imm_test2.py")

import json                                     # загрузка библиотеки считывания данных
import numpy as np                              # загрузка математической библиотеки
from scipy.stats import poisson, norm, uniform  # загрузка статистических функций
import matplotlib.pyplot as plt                 # загрузка функций рисования графиков

import os
cwd = os.getcwd()

print("cwd:"+str(cwd))

file = 'projects/kf/scripts/data/measurements_maneuvre_polar.json'
with open(file) as readfile:
    data = json.load(readfile)

n = len(data) # число тактов ТО в файле

print("n:"+str(n))

class UnscentedKalmanFilter():
    """
    Unscented Kalman filter example
    """
    def __init__(self, T, sa, sr, sp):
        # предсказанные значения вектора состояния
        self.predictedState = {'Xe':[], 'Pe':[], 'Se':[], 'K':[],'ze':[]}
        # полученная оценка вектора состояния
        self.correctedState = {'X':[],'P':[]}
        # сигма-вектора
        self.sigmaVectors = {'Xue':{},'w':{}}
        # настроечные параметры для ускорения
        self.settings = {'T':T,'sa':sa,'sr':sr,'sp':sp,'kappa':-1}

    def cv(self, X,P):
        T = self.settings['T']
        sa = self.settings['sa']
        F = np.array([[1, T, 0, 0],[0,1,0,0],[0,0,1,T],[0,0,0,1]])
        G = np.array([[T**2/2, 0],[T,0],[0,T**2/2],[0,T]])
        Q = np.array([[sa**2, 0],[0,sa**2]])

        # вычисление сигма векторов
        kappa = self.settings['kappa'] # масштабный коэффициент
        n = 4
        U = np.sqrt(n+kappa)*np.linalg.cholesky(P) # взятие матричного корня
        Xu = {**{0:X}, **{i+1:X+U[:,i] for i in range(n)}, **{i+n+1:X-U[:,i] for i in range(n)}} # см. синтаксис объединения словарей
        w = {**{0:kappa/(n+kappa)}, **{i+1:1/2/(n+kappa) for i in range(2*n)}}  # веса векторов от 0 до 2n
        self.sigmaVectors['w'] = w
        # экстраполяция сигма-векторов
        Xue = {i:F@Xu[i] for i in Xu.keys()}
        self.sigmaVectors['Xue'] = Xue
        # статистическая оценка экстраполированного вектора состояния
        Xe = np.zeros(X.shape)
        for i in Xue.keys():
            Xe = Xe + w[i]*Xue[i]
        # статистическая оценка матрицы ковариации экстраполированного вектора состояния
        Pe = np.zeros(P.shape)
        for i in Xue.keys():
            dX = Xue[i]-Xe
            Pe = Pe + w[i]*(np.reshape(dX,(4,1))@np.reshape(dX,(1,4)))
        Pe = Pe + (G@Q)@np.transpose(G)

        return (Xe,Pe)

    def hcv(self,Xe,Pe):
        sr = self.settings['sr']
        sp = self.settings['sp']
        Xue = self.sigmaVectors['Xue']
        w = self.sigmaVectors['w']
        R = np.array([[sr**2, 0],[0,sp**2]])

        # экстраполированные сигма векторая измерений по нелинейным функциям
        zue = {i:np.array([np.sqrt(Xue[i][0]**2+Xue[i][2]**2),np.arctan2(Xue[i][2],Xue[i][0])]) for i in Xue.keys()}
        # статистическая оценка экстраполированного вектора измерения
        ze = np.zeros((1,2))
        for i in zue.keys():
            ze = ze+w[i]*zue[i]
        # статистическая оценка матрицы ковариации экстраполированного вектора измерения
        Pzz = np.zeros((2,2))
        for i in zue.keys():
            v = zue[i]-ze
            Pzz = Pzz + w[i]*np.reshape(v,(2,1))@np.reshape(v,(1,2))
        Se = Pzz + R
        # статистическая оценка матрицы ковариации экстраполированного вектора измерения
        Pxz = np.zeros((4,2))
        for i in zue.keys():
            dX = Xue[i]-Xe
            v = zue[i]-ze
            Pxz = Pxz + w[i]*np.reshape(dX,(4,1))@np.reshape(v,(1,2))

        ze = ze.flatten()
        return (ze,Se,Pxz)

    def predict(self, state_model, measurement_model, X, P):
        (Xe,Pe) = state_model(X,P)
        (ze,Se,Pxz) = measurement_model(Xe,Pe)
        K = Pxz@np.linalg.inv(Se)

        self.predictedState['Xe'] = Xe
        self.predictedState['Pe'] = Pe
        self.predictedState['ze'] = ze
        self.predictedState['Se'] = Se
        self.predictedState['K'] = K

        return (Xe,Pe,Se,ze)

    def correct(self,z):
        Xe = self.predictedState['Xe']
        Pe = self.predictedState['Pe']
        ze = self.predictedState['ze']
        Se = self.predictedState['Se']
        K = self.predictedState['K']

        X = Xe + K@(z-ze)
        P = Pe - (K@Se)@np.transpose(K)

        self.correctedState['X'] = X
        self.correctedState['P'] = P
        return (X,P)

# процедура отображения текущего измерения, предсказанного измерения и строба
def plot_strobe(ze,Se, Pg = 0.9):
    plt.plot(ze[0]*np.cos(ze[1]),ze[0]*np.sin(ze[1]),'*r')   # текущее предсказанное измерение
    phi = np.linspace(0,2*np.pi,720)            # перебор возможных направлений измерений
    cosxy = np.stack((np.cos(phi),np.sin(phi))) # расчет поврота орта
    gamma = poisson.ppf(q = Pg, mu = 2)         # расчет границы по уровню вероятности Pg
    U = np.linalg.cholesky(Se)
    gz = np.tile(np.reshape(ze,(-1,1)),(1,720)) # копрование массив для корректного матричного суммирования
    g = gz + np.sqrt(gamma)*U@cosxy        # вычисляем координаты границ строба
    # добавляем пересчет в декартову СК
    x = g[0][0:]*np.cos(g[1][0:])
    y = g[0][0:]*np.sin(g[1][0:])
    plt.plot(x,y,'r')

# plot_strobe(np.array([2000,0.85]),np.array([[1500,0],[0,1e-3]]),Pg = 0.99) # проверка рисования строба

class IMMfilter():
    """
    IMM filter example
    """
    def __init__(self, f1, f2, mu, tp):
        self.f1 = f1  # первый фильтр
        self.f2 = f2  # второй фильтр
        self.mu = np.array(mu) # начальная матрица вероятностей
        self.tp = np.array(tp) # матрица переходных вероятностей

    def init(self,X,P):
        # инициализация начальных состояний фильтров одинаковыми векторами состояния
        self.f1.correctedState['X'] = X
        self.f1.correctedState['P'] = P
        self.f2.correctedState['X'] = X
        self.f2.correctedState['P'] = P

    def prob(self,z,ze,Se):
        #print("== prob ==")
        #print("z:")
        #print(z)
        #print("ze:")
        #print(ze)
        #print("Se:")
        #print(Se)
        n = 2    # размерность вектора измерений
        v = z-ze # невязка
        #print("v:")
        #print(v)
        #print("np.reshape(v,(1,2)):")
        #print(np.reshape(v,(1,2)))
        #print("np.reshape(v,(2,1)):")
        #print(np.reshape(v,(2,1)))
        power = -0.5*(np.reshape(v,(1,2))@np.linalg.inv(Se))@np.reshape(v,(2,1))
        print("power:")
        print(power)
#        print("prob:")
#        print((1/2/np.pi)**(n/2)/np.sqrt(np.linalg.det(Se))*np.exp(power))
        return (1/2/np.pi)**(n/2)/np.sqrt(np.linalg.det(Se))*np.exp(power)

    def step(self, z):
        print("== step ==")
        X1 = self.f1.correctedState['X']
        P1 = self.f1.correctedState['P']
        X2 = self.f2.correctedState['X']
        P2 = self.f2.correctedState['P']

#        print("X1:")
#        print(X1)
#        print("P1:")
#        print(P1)
#        print("X2:")
#        print(X2)
#        print("P2:")
#        print(P2)

        # 1) расчет переходных вероятностей для формирования смешанных оценок
        mu = self.mu
        tp = self.tp

#        print("mu:")
#        print(mu)
#        print("tp:")
#        print(tp)

        fi = np.zeros((2,2))#[+]
        fi[0] = (mu[0]*tp[0][0]+mu[1]*tp[1][0])
        fi[1] = (mu[0]*tp[0][1]+mu[1]*tp[1][1])

        mx = np.zeros((2,2))
        mx[0][0] = tp[0][0]*mu[0]/(mu[0]*tp[0][0]+mu[1]*tp[1][0])
        mx[1][0] = tp[1][0]*mu[1]/(mu[0]*tp[0][0]+mu[1]*tp[1][0])
        mx[0][1] = tp[0][1]*mu[0]/(mu[0]*tp[0][1]+mu[1]*tp[1][1])
        mx[1][1] = tp[1][1]*mu[1]/(mu[0]*tp[0][1]+mu[1]*tp[1][1])

#        print("mx:")
#        print(mx)

        # 2) вычисление смешанных оценок вектора состояния и смешанных ковариаций
        X01 = mx[0][0]*X1+mx[1][0]*X2
        X02 = mx[0][1]*X1+mx[1][1]*X2

#        print("X01:")
#        print(X01)
#        print("X02:")
#        print(X02)

        dX101 = X1-X01
        dX201 = X2-X01
        dX102 = X1-X02
        dX202 = X2-X02

#        print("dX101:")
#        print(dX101)
#        print("P1:")
#        print(P1)
#        print("np.reshape(dX101,(4,1)):")
#        print(np.reshape(dX101,(4,1)))
#        print("np.reshape(dX101,(1,4)):")
#        print(np.reshape(dX101,(1,4)))
#        print("np.reshape(dX101,(4,1))@np.reshape(dX101,(1,4)):")
#        print(np.reshape(dX101,(4,1))@np.reshape(dX101,(1,4)))
#        print("dX102:")
#        print(dX102)
#        print("dX202:")
#        print(dX202)
        #print("np.reshape(dX101,(4,1)):")
        #print(np.reshape(dX101,(4,1)))

        P01 = (mx[0][0]*(P1+np.reshape(dX101,(4,1))@np.reshape(dX101,(1,4)))+
               mx[1][0]*(P2+np.reshape(dX201,(4,1))@np.reshape(dX201,(1,4))))
        P02 = (mx[0][1]*(P1+np.reshape(dX102,(4,1))@np.reshape(dX102,(1,4)))+
               mx[1][1]*(P2+np.reshape(dX202,(4,1))@np.reshape(dX202,(1,4))))

#        print("P01:")
#        print(P01)
#        print("P02:")
#        print(P02)

        # 3) экстраполяция смешанных оценок
        (Xe1,Pe1,Se1,ze1) = self.f1.predict(self.f1.cv,self.f1.hcv,X01,P01)
        (Xe2,Pe2,Se2,ze2) = self.f2.predict(self.f2.cv,self.f2.hcv,X02,P02)
#        print("Se1:")
#        print(Se1)
#        print("Se2:")
#        print(Se2)
#        print("ze1:")
#        print(ze1)
#        print("ze2:")
#        print(ze2)
        # 4) получение оценок каждой модели
        (X1,P1) = self.f1.correct(z)
        (X2,P2) = self.f2.correct(z)

        # 5) вычисление обновленной вероятнотси
        print("z:")
        print(z)
        print("ze1:")
        print(ze1)
        print("Se1:")
        print(Se1)
        print("self.prob(z,ze1,Se1):")
        print(self.prob(z,ze1,Se1))
        mu[0] = self.prob(z,ze1,Se1)*(mx[0][0]*mu[0]+mx[1][0]*mu[1])
        mu[1] = self.prob(z,ze2,Se2)*(mx[0][1]*mu[0]+mx[1][1]*mu[1])
        denominator = mu[0]+mu[1]
        mu[0] = mu[0]/denominator
        mu[1] = mu[1]/denominator
        self.mu = mu

        # 6) вычисление результирующей оценки вектора состояния
        X = mu[0]*X1+mu[1]*X2
        dX1 = X1-X
        dX2 = X2-X
        P = (mu[0]*(P1+np.reshape(dX1,(4,1))@np.reshape(dX1,(1,4)))+
             mu[1]*(P2+np.reshape(dX2,(4,1))@np.reshape(dX2,(1,4))))
        # подготовка данных для стробирования
        (Xe1,Pe1,Se1,ze1) = self.f1.predict(self.f1.cv,self.f1.hcv,X1,P1)
        (Xe2,Pe2,Se2,ze2) = self.f2.predict(self.f2.cv,self.f2.hcv,X2,P2)
        ze = mu[0]*ze1+mu[1]*ze2
        dz1 = ze1-ze
        dz2 = ze2-ze
        Se = (mu[0]*(Se1+np.reshape(dz1,(2,1))@np.reshape(dz1,(1,2)))+
              mu[1]*(Se2+np.reshape(dz2,(2,1))@np.reshape(dz2,(1,2))))

        return (X,P,Se,ze,mu)

vmax = 50      # априорные сведения о максимальной скорости движения цели
kappa = 3      # коэффициент расширения
sRng = 15        # ско измерений дальности
sPhi = 1/100      # ско измерения углов

pltx = []  # массив отфильтрованных координат x
plty = []  # массив отфильтрованных координат y
pltpx = [] # массив ско по координате x

ukf1 = UnscentedKalmanFilter(T=2,sa=0.1,sr=sRng, sp=sPhi)
ukf2 = UnscentedKalmanFilter(T=2,sa=0.5,sr=sRng, sp=sPhi)
imm = IMMfilter(ukf1,ukf2,mu=[0.99,0.01],tp=[[0.95,0.05],
                                           [0.05,0.95]])

print("filters - DONE!")

# цикл по тактам траекторной обработки содержащим массивы измерений
for i in range(n):
    z = np.array([data[i]['rho'],data[i]['phi']]) # считанное измерение за такт ТО
    if i == 0:
        # процедура инициализации вектора состояния
        X = np.array([z[0]*np.cos(z[1]),0,z[0]*np.sin(z[1]),0])            # (x,vx,y,vy)^T
        P = np.array([[sRng**2,0,0,0],[0,(vmax/kappa)**2,0,0],[0,0,sRng**2,0],[0,0,0,(vmax/kappa)**2]])
        imm.init(X,P)

    # один шаг фильтрации за такт ТО в ПСК
    (X,P,Se,ze,mu) = imm.step(z)

    # обработка логики стробирования и отождествления
    plot_strobe(ze,Se,Pg=0.997)

    # рисуем оценку и сохраняем данные для отображения

    plt.plot(z[0]*np.cos(z[1]),z[0]*np.sin(z[1]),'+b') # текущее измерение в ПСК
    plt.plot(X[0],X[2],'sg')                           # полученная оценка в ДСК

    pltx.append(mu[0])
    plty.append(mu[1])
    pltpx.append(P[0][0])

plt.grid(True)

plt.show()

print("~imm_test2.py")
