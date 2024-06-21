#!/usr/bin/python3

import math
import numpy as np
import filterpy
from numpy.random import randn
from numpy import eye, array, asarray
import matplotlib.pyplot as plt
import estimator as e

### Создание наборов данных ##################################################################################
# Функция создания набора данных: по кругу
def make_true_data_round(x0, modelF, amount, period):
    X = np.zeros((x0.shape[0], amount))
    X[:, 0] = x0.T
    for i in range(amount-1):
        xx = modelF(np.copy(X[:, i]),period)
        X[:, i+1] = xx.flatten()
    return X

### Добавление к наборам данных ошибок процесса ##############################################################
def add_process_noise(X,Var):
    Xn = X + np.sqrt(Var) @ np.random.normal(loc=0, scale=1.0, size=(X.shape[0], X.shape[1]))
    return Xn

### Получение из наборов данных измерений и добавление к ним шцмов ###########################################
def make_meas(X, R, modelH):
    Z = np.zeros((R.shape[0], X.shape[1]))
    for i in range(Z.shape[1]):
        zz = modelH(np.copy(X[:, i]))
        Z[:, i] = zz.flatten()
    Zn = Z + np.sqrt(R) @ np.random.normal(loc=0, scale=math.sqrt(1.0), size=(Z.shape[0], Z.shape[1]))
    return Zn

def generate(state_list, amount_list, Q, R, period, modelF=e.stateModel_CTx, modelH=e.measureModel_XwXx):
    X=make_true_data_round(state_list[0], modelF, amount_list[0], period)
    for i in range(len(state_list)-1):
        xx = X[:,-1]
        for j in range(state_list[i+1].shape[0]):
            if state_list[i+1][j] != None:
                xx[j] = state_list[i+1][j]
        XX=make_true_data_round(xx, modelF, amount_list[i+1], period)
        X = np.hstack((X,XX))

    Xn = add_process_noise(X,Q)
    Zn = make_meas(Xn, R, modelH)
    return X, Xn, Zn
