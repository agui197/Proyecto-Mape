# -*- coding: utf-8 -*-
"""
Created on Wed May  6 12:13:54 2020

@author: RIEMANNRUIZ
"""

#Importar librerias
import numpy as np
import matplotlib.pyplot as plt

#%% Generacion de los datos
n = 20
x1 = np.linspace(-10,10,n)
x2 = np.linspace(-10,10,n)
X1,X2 = np.meshgrid(x1,x2)
Y = 2*((np.matrix(X1)**3-3*X1)>X2)-1

x1m = np.ravel(X1.T)
x2m = np.ravel(X2.T)
Xm = np.c_[x1m,x2m]
y = np.ravel(Y.T)
Ym = np.matrix(y).T*np.matrix(y)

#%% Visualizar los datos
plt.scatter(x1m,x2m,c=y)
plt.show()

#%% Kernel matrix
from sklearn.metrics.pairwise import linear_kernel
K = linear_kernel(Xm,Xm)

#%% Optimization Variables
H = np.multiply(K,Ym)
onev = -np.ones((y.shape[0],1))
Aeq = np.float64(np.reshape(y,(1,y.shape[0])))
beq = 0.0

#%%
from cvxopt import matrix, solvers
P = matrix(H)
q = matrix(onev)
A = matrix(Aeq)
b = matrix(beq)
sol=solvers.qp(P=P, q=q, A=A, b=b)

#%%
#from cvxopt import matrix, solvers
#Q = 2*matrix([ [2, .5], [.5, 1] ])
#p = matrix([1.0, 1.0])
#G = matrix([[-1.0,0.0],[0.0,-1.0]])
#h = matrix([0.0,0.0])
#A = matrix([1.0, 1.0], (1,2))
#b = matrix(1.0)
#sol=solvers.qp(P=Q, q=p, A=A, b=b)