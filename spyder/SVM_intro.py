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
c=0.1
G = np.float64(np.concatenate((np.identity(y.shape[0]),-np.identity(y.shape[0]))))
#G = np.float64(np.identity(y.shape[0]))
h=np.float64(np.concatenate((c*np.ones((y.shape[0],1)),np.zeros((y.shape[0],1)))))

#%% Solucion al problema de optimizacion cuadratica
from cvxopt import matrix, solvers
P = matrix(H)
q = matrix(onev)
A = matrix(Aeq)
b = matrix(beq)
G = matrix(G)
h = matrix(h)
sol=solvers.qp(P=P, q=q,G=G,h=h, A=A, b=b)

alphas = np.array(sol['x'])

#%% Support vector values
indx = alphas > 1e-10
alpha_sv = alphas[indx]
x_sv = Xm[indx[:,0],:]
y_sv = y[indx[:,0]]

#%% Calculo de las w
w = np.sum(np.c_[alpha_sv*y_sv,alpha_sv*y_sv]*x_sv,axis=0)
b = np.mean(1/y_sv-np.dot(x_sv,w))

#%% Visualizar los datos
plt.scatter(x1m,x2m,c=y)
plt.plot(x1,(-w[0]*x1-b)/w[1])
plt.xlim(-10,10)
plt.ylim(-10,10)
plt.show()

