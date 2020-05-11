# -*- coding: utf-8 -*-
"""
Created on Wed May  6 12:13:54 2020

@author: RIEMANNRUIZ
"""

#Importar librerias
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics.pairwise import linear_kernel
from cvxopt import matrix, solvers
import cvxpy as cp #https://www.cvxpy.org/

#%% Generacion de los datos
n = 20
x1 = np.linspace(-10,10,n)
x2 = np.linspace(-10,10,n)
X1,X2 = np.meshgrid(x1,x2)
Y = 2*X1+3*X2+40+(5*np.random.rand(X1.shape[0],X1.shape[0])-2.5)

x1m = np.ravel(X1.T)
x2m = np.ravel(X2.T)
Xm = np.c_[x1m,x2m]
y = np.ravel(Y.T)

#%% Visualizar los datos
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x1m, x2m, y, c=y)
plt.show()




#%% Kernel matrix
K = linear_kernel(Xm,Xm)

#%% Optimization Variables usando cvxpy
epsilon = 1 # margin max
c = 10 # alphas constraint
onev = np.ones((y.shape[0],1))
Ev = onev*epsilon
error = 1E-5 # vector support 

alpha1 = cp.Variable((y.shape[0],1))
alpha2 = cp.Variable((y.shape[0],1))

objective = cp.Minimize((1/2)*cp.quad_form(alpha1-alpha2, K) + Ev.T @ (alpha1+alpha2) - y.T @ (alpha1 - alpha2))
constraints = [onev.T @ (alpha1-alpha2) == 0, alpha1 >= 0, alpha1 <= c,alpha2 >= 0, alpha2 <= c]
prob = cp.Problem(objective,constraints)

# The optimal objective value is returned by `prob.solve()`.
result = prob.solve()


alpha1 = np.array(alpha1.value)
alpha2 = np.array(alpha2.value)
alphas = alpha1-alpha2
indx = abs(alphas) > error
alpha_sv = alphas[indx]
x_sv = Xm[indx[:,0],:]
y_sv = y[indx[:,0]]


w = np.sum(np.c_[alpha_sv,alpha_sv]*x_sv,axis=0)
b = np.mean(y_sv-np.dot(x_sv,w))

print('w=[%0.3f,%0.3f]'%(w[0],w[1]))
print('b=%0.3f'%b)

#%% Visualizar los datos
ym = w[0]*x1m+w[1]*x2m+b

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x1m, x2m, y, c=y,s=5)
ax.scatter(x1m, x2m, ym, c='r',s=10)
ax.view_init(30, 0)
plt.show()

