#!/usr/bin/python3

import math
import sympy
import numpy

x, vx, y, vy, z, vz, w, t = sympy.symbols('x vx y vy z vz w t')

F = sympy.matrices.Matrix([[1., sympy.sin(w*t)/w    , 0., (sympy.cos(w*t)-1)/w, 0., 0., 0.],
                           [0., sympy.cos(w*t)      , 0., -sympy.sin(w*t)     , 0., 0., 0.],
                           [0., (1-sympy.cos(w*t))/w, 1., sympy.sin(w*t)/w    , 0., 0., 0.],
                           [0., sympy.sin(w*t)      , 0., sympy.cos(w*t)      , 0., 0., 0.],
                           [0., 0.                  , 0., 0.                  , 1., t , 0.],
                           [0., 0.                  , 0., 0.                  , 0., 1., 0.],
                           [0., 0.                  , 0., 0.                  , 0., 0., 1.]])

H = sympy.matrices.Matrix([[1., 0., 0., 0., 0., 0., 0.],
                           [0., 0., 1., 0., 0., 0., 0.],
                           [0., 0., 0., 0., 1., 0., 0.]])

State = sympy.matrices.Matrix([[x],
                               [vx],
                               [y],
                               [vy],
                               [z],
                               [vz],
                               [w]])

print("F:")
sympy.pprint(F)
print("H:")
sympy.pprint(H)
print("State("+str(State.shape[0])+","+str(State.shape[1])+"):")
sympy.pprint(State)

ModelF = F*State
ModelH = H*State

print("ModelF:")
sympy.pprint(ModelF)
print("ModelH:")
sympy.pprint(ModelH)

JF = sympy.matrices.zeros(ModelF.shape[0], State.shape[0])
JH = sympy.matrices.zeros(ModelH.shape[0], State.shape[0])

for i in range(ModelF.shape[0]):
    for j in range(State.shape[0]):
        #print("sympy.diff("+str(Model[i,0])+","+str(State[j,0])+")")
        JF[i,j] = sympy.diff(ModelF[i,0],State[j,0])

for i in range(ModelH.shape[0]):
    for j in range(State.shape[0]):
        #print("sympy.diff("+str(Model[i,0])+","+str(State[j,0])+")")
        JH[i,j] = sympy.diff(ModelH[i,0],State[j,0])

print("JACOBIAN_F:")
sympy.pprint(JF)

print("JACOBIAN_H:")
sympy.pprint(JH)


