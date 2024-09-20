#!/usr/bin/python3

import math
import sympy
import numpy

x, vx, ax, y, vy, ay, z, vz, az, w, t = sympy.symbols('x vx ax y vy ay z vz az w t')

#########################################################################################

print("# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # ")

FCT5 = sympy.matrices.Matrix([[1., sympy.sin(w*t)/w    , 0., (sympy.cos(w*t)-1)/w, 0.],
                              [0., sympy.cos(w*t)      , 0., -sympy.sin(w*t)     , 0.],
                              [0., (1-sympy.cos(w*t))/w, 1., sympy.sin(w*t)/w    , 0.],
                              [0., sympy.sin(w*t)      , 0., sympy.cos(w*t)      , 0.],
                              [0., 0.                  , 0., 0.                  , 1.]])

State5 = sympy.matrices.Matrix([[x],
                                [vx],
                                [y],
                                [vy],
                                [w]])

print("FCT5:")
sympy.pprint(FCT5)
print("State5("+str(State5.shape[0])+","+str(State5.shape[1])+"):")
sympy.pprint(State5)

ModelFCT5 = FCT5*State5

print("ModelFCT5:")
sympy.pprint(ModelFCT5)

JFCT5 = sympy.matrices.zeros(ModelFCT5.shape[0], State5.shape[0])

for i in range(ModelFCT5.shape[0]):
    for j in range(State5.shape[0]):
        #print("sympy.diff("+str(Model[i,0])+","+str(State[j,0])+")")
        JFCT5[i,j] = sympy.diff(ModelFCT5[i,0],State5[j,0])

print("JACOBIAN_F_CT_5:")
sympy.pprint(JFCT5)

#########################################################################################

print("# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # ")

StateP4 = sympy.matrices.Matrix([[x],
                                [vx],
                                [y],
                                [vy]])

print("StateP4("+str(StateP4.shape[0])+","+str(StateP4.shape[1])+"):")
sympy.pprint(StateP4)

ModelP4 = sympy.matrices.Matrix([[sympy.sqrt(x*x+y*y)],
                                 [sympy.atan(y/x)]   ])

print("ModelP4:")
sympy.pprint(ModelP4)

JP4 = sympy.matrices.zeros(ModelP4.shape[0], StateP4.shape[0])

for i in range(ModelP4.shape[0]):
    for j in range(StateP4.shape[0]):
        #print("sympy.diff("+str(ModelP4[i,0])+","+str(StateP4[j,0])+")")
        JP4[i,j] = sympy.diff(ModelP4[i,0],StateP4[j,0])

print("JP4:")
sympy.pprint(JP4)

#########################################################################################

print("# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # ")

StateP10 = sympy.matrices.Matrix([[x],
                                  [vx],
                                  [ax],
                                  [y],
                                  [vy],
                                  [ay],
                                  [z],
                                  [vz],
                                  [az],
                                  [w]])

print("StateP10("+str(StateP10.shape[0])+","+str(StateP10.shape[1])+"):")
sympy.pprint(StateP10)

ModelP10 = sympy.matrices.Matrix([[sympy.sqrt(x*x+y*y+z*z)],
                                  [sympy.atan(y/x)],
                                  [sympy.atan(z/sympy.sqrt(y*y+x*x))]])

print("ModelP10:")
sympy.pprint(ModelP10)

JP10 = sympy.matrices.zeros(ModelP10.shape[0], StateP10.shape[0])

for i in range(ModelP10.shape[0]):
    for j in range(StateP10.shape[0]):
        #print("sympy.diff("+str(ModelP10[i,0])+","+str(StateP10[j,0])+")")
        JP10[i,j] = sympy.diff(ModelP10[i,0],StateP10[j,0])

print("JP10:")
sympy.pprint(JP10)

