#!/usr/bin/python3

import math
import sympy
import numpy

x, vx, ax, y, vy, ay, z, vz, az, w, t = sympy.symbols('x vx ax y vy ay z vz az w t')

FCT7 = sympy.matrices.Matrix([[1., sympy.sin(w*t)/w    , 0., (sympy.cos(w*t)-1)/w, 0., 0., 0.],
                              [0., sympy.cos(w*t)      , 0., -sympy.sin(w*t)     , 0., 0., 0.],
                              [0., (1-sympy.cos(w*t))/w, 1., sympy.sin(w*t)/w    , 0., 0., 0.],
                              [0., sympy.sin(w*t)      , 0., sympy.cos(w*t)      , 0., 0., 0.],
                              [0., 0.                  , 0., 0.                  , 1., t , 0.],
                              [0., 0.                  , 0., 0.                  , 0., 1., 0.],
                              [0., 0.                  , 0., 0.                  , 0., 0., 1.]])

FCT10 = sympy.matrices.Matrix([[1., sympy.sin(w*t)/w    , 0., 0., (sympy.cos(w*t)-1)/w, 0., 0., 0., 0., 0.],
                               [0., sympy.cos(w*t)      , 0., 0., -sympy.sin(w*t)     , 0., 0., 0., 0., 0.],
                               [0., 0.                  , 1., 0., 0.                  , 0., 0., 0., 0., 0.],
                               [0., (1-sympy.cos(w*t))/w, 0., 1., sympy.sin(w*t)/w    , 0., 0., 0., 0., 0.],
                               [0., sympy.sin(w*t)      , 0., 0., sympy.cos(w*t)      , 0., 0., 0., 0., 0.],
                               [0., 0.                  , 0., 0., 0.                  , 1., 0., 0., 0., 0.],
                               [0., 0.                  , 0., 0., 0.                  , 0., 1., t , 0., 0.],
                               [0., 0.                  , 0., 0., 0.                  , 0., 0., 1., 0., 0.],
                               [0., 0.                  , 0., 0., 0.                  , 0., 0., 0., 1., 0.],
                               [0., 0.                  , 0., 0., 0.                  , 0., 0., 0., 0., 1.]])

FCA10 = sympy.matrices.Matrix([[1., t , (t*t)/2., 0., 0.,       0., 0., 0.,       0., 0.],
                               [0., 1.,       t , 0., 0.,       0., 0., 0.,       0., 0.],
                               [0., 0.,       1., 0., 0.,       0., 0., 0.,       0., 0.],
                               [0., 0.,       0., 1., t , (t*t)/2., 0., 0.,       0., 0.],
                               [0., 0.,       0., 0., 1.,       t , 0., 0.,       0., 0.],
                               [0., 0.,       0., 0., 0.,       1., 0., 0.,       0., 0.],
                               [0., 0.,       0., 0., 0.,       0., 1., t , (t*t)/2., 0.],
                               [0., 0.,       0., 0., 0.,       0., 0., 1.,       t , 0.],
                               [0., 0.,       0., 0., 0.,       0., 0., 0.,       1., 0.],
                               [0., 0.,       0., 0., 0.,       0., 0., 0.,       0., 1.]])

H7 = sympy.matrices.Matrix([[1., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 1., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 1., 0., 0.]])

State7 = sympy.matrices.Matrix([[x],
                                [vx],
                                [y],
                                [vy],
                                [z],
                                [vz],
                                [w]])

State10 = sympy.matrices.Matrix([[x],
                                 [vx],
                                 [ax],
                                 [y],
                                 [vy],
                                 [ay],
                                 [z],
                                 [vz],
                                 [az],
                                 [w]])

print("FCT7:")
sympy.pprint(FCT7)
print("FCA10:")
sympy.pprint(FCA10)
print("FCT10:")
sympy.pprint(FCT10)
print("H7:")
sympy.pprint(H7)
print("State7("+str(State7.shape[0])+","+str(State7.shape[1])+"):")
sympy.pprint(State7)
print("State10("+str(State10.shape[0])+","+str(State10.shape[1])+"):")
sympy.pprint(State10)

ModelFCT7 = FCT7*State7
ModelFCA10 = FCA10*State10
ModelFCT10 = FCT10*State10
ModelH7 = H7*State7

print("ModelFCT7:")
sympy.pprint(ModelFCT7)
print("ModelFCA10:")
sympy.pprint(ModelFCA10)
print("ModelFCT10:")
sympy.pprint(ModelFCT10)
print("ModelH7:")
sympy.pprint(ModelH7)

JFCT7 = sympy.matrices.zeros(ModelFCT7.shape[0], State7.shape[0])
JFCA10 = sympy.matrices.zeros(ModelFCA10.shape[0], State10.shape[0])
JFCT10 = sympy.matrices.zeros(ModelFCT10.shape[0], State10.shape[0])
JH7 = sympy.matrices.zeros(ModelH7.shape[0], State7.shape[0])

for i in range(ModelFCT7.shape[0]):
    for j in range(State7.shape[0]):
        #print("sympy.diff("+str(Model[i,0])+","+str(State[j,0])+")")
        JFCT7[i,j] = sympy.diff(ModelFCT7[i,0],State7[j,0])

for i in range(ModelFCA10.shape[0]):
    for j in range(State10.shape[0]):
        #print("sympy.diff("+str(Model[i,0])+","+str(State[j,0])+")")
        JFCA10[i,j] = sympy.diff(ModelFCA10[i,0],State10[j,0])

for i in range(ModelFCT10.shape[0]):
    for j in range(State10.shape[0]):
        #print("sympy.diff("+str(Model[i,0])+","+str(State[j,0])+")")
        JFCT10[i,j] = sympy.diff(ModelFCT10[i,0],State10[j,0])

for i in range(ModelH7.shape[0]):
    for j in range(State7.shape[0]):
        #print("sympy.diff("+str(Model[i,0])+","+str(State[j,0])+")")
        JH7[i,j] = sympy.diff(ModelH7[i,0],State7[j,0])

print("JACOBIAN_F_CT_7:")
sympy.pprint(JFCT7)

print("JACOBIAN_F_CA_10:")
sympy.pprint(JFCA10)

print("JACOBIAN_F_CT_10:")
sympy.pprint(JFCT10)

print("JACOBIAN_H_7:")
sympy.pprint(JH7)


