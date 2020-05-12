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
from sklearn.metrics import mean_squared_error

#%% Funcion MAPE
def mean_absolute_percentage_error(y_true,y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

#%% Generacion de los datos
np.random.seed(1)
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

#%% Optimization RMSE usando cvxpy
epsilon = 0.1 # margin max
c = 10 # alphas constraint
onev = np.ones((y.shape[0],1))
error = 1E-5 # vector support 

alpha1 = cp.Variable((y.shape[0],1))
alpha2 = cp.Variable((y.shape[0],1))

#% RMSE minimization
Ev = onev*epsilon
objective = cp.Minimize((1/2)*cp.quad_form(alpha1-alpha2, K) + Ev.T @ (alpha1+alpha2) - y.T @ (alpha1 - alpha2))
h=np.float64(c*np.ones((y.shape[0],1)))
constraints = [onev.T @ (alpha1-alpha2) == 0, alpha1 >= 0, alpha1 <= h,alpha2 >= 0, alpha2 <= h]
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


w_rmse = np.sum(np.c_[alpha_sv,alpha_sv]*x_sv,axis=0)
b_rmse = np.mean(y_sv-np.dot(x_sv,w_rmse))

print('w_rmse=[%0.3f,%0.3f]'%(w_rmse[0],w_rmse[1]))
print('b_rmse=%0.3f'%b_rmse)

#% Visualizar los resultados
y_rmse = w_rmse[0]*x1m+w_rmse[1]*x2m+b_rmse

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x1m, x2m, y, c=y,s=5)
ax.scatter(x1m, x2m, y_rmse, c='r',s=10)
ax.view_init(30, 0)
plt.show()






######################################
#%% Optimization MAPE usando cvxpy
epsilon = 0.1 # margin max
c = 20 # alphas constraint
onev = np.ones((y.shape[0],1))

error = 1E-5 # vector support 

alpha1 = cp.Variable((y.shape[0],1))
alpha2 = cp.Variable((y.shape[0],1))
#%
Ev = np.reshape(y,(y.shape[0],1))*epsilon
objective = cp.Minimize((1/2)*cp.quad_form(alpha1-alpha2, K) + Ev.T @ (alpha1+alpha2) - y.T @ (alpha1 - alpha2))

h=np.float64(abs(c/np.reshape(y,(y.shape[0],1))))

constraints = [onev.T @ (alpha1-alpha2) == 0, alpha1 >= 0, alpha1 <= h,alpha2 >= 0, alpha2 <= h]
prob = cp.Problem(objective,constraints)
#%
# The optimal objective value is returned by `prob.solve()`.
result = prob.solve()


alpha1 = np.array(alpha1.value)
alpha2 = np.array(alpha2.value)
alphas = alpha1-alpha2
indx = abs(alphas) > error
alpha_sv = alphas[indx]
x_sv = Xm[indx[:,0],:]
y_sv = y[indx[:,0]]


w_mape = np.sum(np.c_[alpha_sv,alpha_sv]*x_sv,axis=0)
b_mape = np.mean(y_sv-np.dot(x_sv,w_mape))

print('w_mape=[%0.3f,%0.3f]'%(w_mape[0],w_mape[1]))
print('b_mape=%0.3f'%b_mape)

#% Visualizar los resultados
y_mape = w_mape[0]*x1m+w_mape[1]*x2m+b_mape

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x1m, x2m, y, c=y,s=5)
ax.scatter(x1m, x2m, y_mape, c='r',s=10)
ax.view_init(30, 0)
plt.show()

#%% Evaluacion de ambas implementaciones
rmse1,mape1 = mean_squared_error(y,y_rmse),mean_absolute_percentage_error(y,y_rmse)
rmse2,mape2 = mean_squared_error(y,y_mape),mean_absolute_percentage_error(y,y_mape)
print('\n\n\t Obj RMSE\t Obj MAPE\n RMSE\t %0.4f\t\t %0.4f\nMAPE\t %0.4f\t\t %0.4f'%(rmse1,rmse2,mape1,mape2))