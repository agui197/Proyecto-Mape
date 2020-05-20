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
import cvxpy as cp #https://www.cvxpy.org/
from sklearn.metrics import mean_squared_error

#%% Funcion MAPE
def mean_absolute_percentage_error(y_true,y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

#%% Generacion de los datos
np.random.seed(1)
n = 20
x1 = np.linspace(1,20,n)
x2 = np.linspace(1,20,n)
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

#%% Funcion SVR_E
def SVR_E(X,y,epsilon=0.01,c=10):
    # epsilon = 0.01 # margin max
    # c = 10 # alphas constraint
    umbral = 1E-5 # umbral to define a vector support
    
    nsamples,nfeatures = np.shape(X)
    onev = np.ones((nsamples,1))
    
    # Kernel matrix
    K = linear_kernel(X,X)

    # Optimization E-regression usando cvxpy
    alpha1 = cp.Variable((nsamples,1))
    alpha2 = cp.Variable((nsamples,1))
    
    #% Forma Original
    Ev = onev*epsilon
    objective = cp.Minimize((1/2)*cp.quad_form(alpha1-alpha2, K) + Ev.T @ (alpha1+alpha2) - y.T @ (alpha1 - alpha2))
    
    # Restricciones forma matricial
    G = np.float64(np.concatenate((np.identity(nsamples),-np.identity(nsamples))))
    h = np.float64(np.concatenate((c*np.ones((nsamples,1)),np.zeros((nsamples,1)))))
    
    constraints = [onev.T @ (alpha1-alpha2) == 0, G @ alpha1 <= h, G @ alpha2 <= h]
    
    # The optimal objective value is returned by `prob.solve()`.
    prob = cp.Problem(objective,constraints)
    result = prob.solve()
    
    alpha1 = np.array(alpha1.value)
    alpha2 = np.array(alpha2.value)
    alphas = alpha1-alpha2
    indx = abs(alphas) > umbral
    alpha_sv = alphas[indx]
    x_sv = X[indx[:,0],:]
    y_sv = y[indx[:,0]]
    
    
    w = np.sum(np.transpose(np.tile(alpha_sv,(nfeatures,1)))*x_sv,axis=0)
    b = np.mean(y_sv-np.dot(x_sv,w))
    
    print('w=')
    print(w)
    print('b=')
    print(b)
    
    return w,b
#%% Aplicar la regresion epsilon
w_Ereg,b_Ereg = SVR_E(Xm,y,epsilon=0.01,c=10)

#% Visualizar los resultados
y_Ereg = w_Ereg[0]*x1m+w_Ereg[1]*x2m+b_Ereg

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x1m, x2m, y, c=y,s=5)
ax.scatter(x1m, x2m, y_Ereg, c='r',s=10)
ax.view_init(30, 0)
plt.show()






######################################
#%% Optimization E-regression MAPE usando cvxpy
def SVR_E_MAPE(X,y,epsilon=0.01,c=10):
    # epsilon = 0.01 # margin max
    # c = 10 # alphas constraint
    umbral = 1E-5 # umbral to define a vector support
    
    nsamples,nfeatures = np.shape(X)
    onev = np.ones((nsamples,1))
    
    # Kernel matrix
    K = linear_kernel(X,X)
    
    alpha1 = cp.Variable((nsamples,1))
    alpha2 = cp.Variable((nsamples,1))
    
    #% Forma MAPE
    Ev = np.reshape(y,(nsamples,1))*epsilon
    objective = cp.Minimize((1/2)*cp.quad_form(alpha1-alpha2, K) + Ev.T @ (alpha1+alpha2) - y.T @ (alpha1 - alpha2))
    
    # Restricciones forma matricial
    G = np.float64(np.concatenate((np.identity(nsamples),-np.identity(nsamples))))
    h=np.float64(np.concatenate((c/np.reshape(y,(nsamples,1)),np.zeros((nsamples,1)))))
    constraints = [onev.T @ (alpha1-alpha2) == 0, G @ alpha1 <= h, G @ alpha2 <= h]
    
    # The optimal objective value is returned by `prob.solve()`.
    prob = cp.Problem(objective,constraints)
    result = prob.solve()
    
    
    alpha1 = np.array(alpha1.value)
    alpha2 = np.array(alpha2.value)
    alphas = alpha1-alpha2
    indx = abs(alphas) > umbral
    alpha_sv = alphas[indx]
    x_sv = X[indx[:,0],:]
    y_sv = y[indx[:,0]]
    
    
    w = np.sum(np.transpose(np.tile(alpha_sv,(nfeatures,1)))*x_sv,axis=0)
    b = np.mean(y_sv-np.dot(x_sv,w))
    
    print('w=')
    print(w)
    print('b=')
    print(b)
    
    return w,b

#%% Aplicar la regresion epsilon MAPE
w_mape,b_mape = SVR_E_MAPE(Xm,y,epsilon=0.01,c=10)
#% Visualizar los resultados
y_mape = w_mape[0]*x1m+w_mape[1]*x2m+b_mape

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x1m, x2m, y, c=y,s=5)
ax.scatter(x1m, x2m, y_mape, c='r',s=10)
ax.view_init(30, 0)
plt.show()

#%% Optimization classic v formulation E-regression usando cvxpy
def SVR_vE(X,y,epsilon=0.01,c=10,v=1):
#    epsilon = 0.01 # margin max
#    v = 1 # New term
#    c = 10 # alphas constraint
    
    umbral = 1E-5 # vector support 
    nsamples,nfeatures = np.shape(X)
    onev = np.ones((nsamples,1))
    
    # Kernel matrix
    K = linear_kernel(X,X)
    
    alpha1 = cp.Variable((nsamples,1))
    alpha2 = cp.Variable((nsamples,1))
    
    #% Forma Original
    objective = cp.Minimize((1/2)*cp.quad_form(alpha1-alpha2, K) - y.T @ (alpha1 - alpha2))
    
    # Restricciones forma matricial
    G = np.float64(np.concatenate((np.identity(nsamples),-np.identity(nsamples))))
    h = np.float64(np.concatenate((c*np.ones((nsamples,1)),np.zeros((nsamples,1)))))
    
    constraints = [onev.T @ (alpha1-alpha2) == 0,
                   onev.T @ (alpha1+alpha2) == c*v,
                   G @ alpha1 <= h,
                   G @ alpha2 <= h]
    
    # The optimal objective value is returned by `prob.solve()`.
    prob = cp.Problem(objective,constraints)
    result = prob.solve()
    
    alpha1 = np.array(alpha1.value)
    alpha2 = np.array(alpha2.value)
    alphas = alpha1-alpha2
    indx = abs(alphas) > umbral
    alpha_sv = alphas[indx]
    x_sv = X[indx[:,0],:]
    y_sv = y[indx[:,0]]
    
    
    w = np.sum(np.transpose(np.tile(alpha_sv,(nfeatures,1)))*x_sv,axis=0)
    b = np.mean(y_sv-np.dot(x_sv,w))
    
    print('w=')
    print(w)
    print('b=')
    print(b)
    
    return w,b

#%% Aplicar la regresion epsilon con formulacion v
w_vE,b_vE = SVR_vE(Xm,y,epsilon=0.01,c=10,v=1)
#% Visualizar los resultados
y_vE = w_vE[0]*x1m+w_vE[1]*x2m+b_vE

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x1m, x2m, y, c=y,s=5)
ax.scatter(x1m, x2m, y_vE, c='r',s=10)
ax.view_init(30, 0)
plt.show()

#%% Optimization v formulation MAPE-regression usando cvxpy
def SVR_vMAPE(X,y,epsilon=0.01,c=10,v=1):
#    epsilon = 0.01 # margin max
#    v = 1 # New term
#    c = 10 # alphas constraint
    
    umbral = 1E-5 # vector support 
    nsamples,nfeatures = np.shape(X)
    onev = np.ones((nsamples,1))
    
    # Kernel matrix
    K = linear_kernel(X,X)
    
    alpha1 = cp.Variable((nsamples,1))
    alpha2 = cp.Variable((nsamples,1))
    
    #% Forma MAPE
    objective = cp.Minimize((1/2)*cp.quad_form(alpha1-alpha2, K) - y.T @ (alpha1 - alpha2))
    
    # Restricciones forma matricial
    G = np.float64(np.concatenate((np.identity(nsamples),-np.identity(nsamples))))
    h=np.float64(np.concatenate((c/np.reshape(y,(nsamples,1)),np.zeros((nsamples,1)))))
    constraints = [onev.T @ (alpha1-alpha2) == 0,
                   y.T @ (alpha1+alpha2) == c*v,
                   G @ alpha1 <= h,
                   G @ alpha2 <= h]
    
    # The optimal objective value is returned by `prob.solve()`.
    prob = cp.Problem(objective,constraints)
    result = prob.solve()
    
    alpha1 = np.array(alpha1.value)
    alpha2 = np.array(alpha2.value)
    alphas = alpha1-alpha2
    indx = abs(alphas) > umbral
    alpha_sv = alphas[indx]
    x_sv = X[indx[:,0],:]
    y_sv = y[indx[:,0]]
    
    
    w = np.sum(np.transpose(np.tile(alpha_sv,(nfeatures,1)))*x_sv,axis=0)
    b = np.mean(y_sv-np.dot(x_sv,w))
    
    print('w=')
    print(w)
    print('b=')
    print(b)
    
    return w,b

#%% Aplicar la regresion epsilon con formulacion v
w_vmape,b_vmape = SVR_vMAPE(Xm,y,epsilon=0.01,c=10,v=1)
#% Visualizar los resultados
y_vmape = w_vmape[0]*x1m+w_vmape[1]*x2m+b_vmape

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x1m, x2m, y, c=y,s=5)
ax.scatter(x1m, x2m, y_vmape, c='r',s=10)
ax.view_init(30, 0)
plt.show()

#%% Evaluacion de ambas implementaciones
rmse1,mape1 = mean_squared_error(y,y_Ereg),mean_absolute_percentage_error(y,y_Ereg)
rmse2,mape2 = mean_squared_error(y,y_mape),mean_absolute_percentage_error(y,y_mape)
print('\n\n\t Obj E_reg\t Obj MAPE\n RMSE\t %0.4f\t\t %0.4f\n MAPE\t %0.4f\t\t %0.4f'%(rmse1,rmse2,mape1,mape2))