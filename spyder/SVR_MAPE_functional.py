# -*- coding: utf-8 -*-
"""
Created on Wed May  6 12:13:54 2020

@author: RIEMANNRUIZ
"""

#Importar librerias
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics.pairwise import (linear_kernel,rbf_kernel)
import cvxpy as cp #https://www.cvxpy.org/
from sklearn.metrics import mean_squared_error

#%% Funcion MAPE
def mean_absolute_percentage_error(y_true,y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

#%% Kernel linear-rbf linear combination
def linrbf_kernel(X1,X2,gamma=None,lck=1):
    return lck*linear_kernel(X1,X2)+(1-lck)*rbf_kernel(X1,X2,gamma=gamma)

#%% Generacion de los datos
np.random.seed(1)
n = 29
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
def SVR_E(X,y,epsilon=0.01,c=10,kernel='linear',gamma=None,lck=1):
    # epsilon = 0.01 # margin max
    # c = 10 # alphas constraint
    # kernel = 'linear' # kernel type, options: ('linear','rbf','linrbf')
    # gamma = None # gamma parameter for rbf-kernel and linrbf-kernel
    # lck = 1 # constant for kernel linear combination, 0<=lck<=1
    
    umbral = 1E-5 # umbral to define a vector support
    
    nsamples,nfeatures = np.shape(X)
    onev = np.ones((nsamples,1))
    
    # Kernel matrix
    if kernel == 'linear':
        K = linear_kernel(X,X)
    elif kernel == 'rbf':
        K = rbf_kernel(X,X,gamma=gamma)
    elif kernel == 'linrbf':
        K = linrbf_kernel(X,X,gamma=gamma,lck=lck)

    # Optimization E-regression
    alpha1 = cp.Variable((nsamples,1))
    alpha2 = cp.Variable((nsamples,1))
    
    Ev = onev*epsilon
    objective = cp.Minimize((1/2)*cp.quad_form(alpha1-alpha2, K) + Ev.T @ (alpha1+alpha2) - y.T @ (alpha1 - alpha2))
    
    # Constrains in matrix form
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
# Parameters values
#   epsilon default: 0.1, epsilon>0
#   c default: 10, c>0
#   kernel default: 'linear', other values ('rbf','linrbf')
#   gamma default: None, other values gamma>0
#   lck default: 1 (linear kernel), other values 0<=lck<=1

# kernel = 'linear' # kernel type selection
# lck = 1 # constant to kernel linear combination
# gamma = None # parameter

w_Ereg,b_Ereg = SVR_E(Xm,y,epsilon=0.01,c=10,kernel='linear',gamma=None,lck=1)

#% Simular el modelo
y_Ereg = np.dot(Xm,w_Ereg)+b_Ereg


#% Visualizar los resultados
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x1m, x2m, y, c=y,s=5)
ax.scatter(x1m, x2m, y_Ereg, c='r',s=10)
ax.view_init(30, 0)
plt.show()






######################################
#%% Optimization E-regression MAPE usando cvxpy
def SVR_E_MAPE(X,y,epsilon=0.01,c=10,kernel='linear',gamma=None,lck=1):
    # epsilon = 0.01 # margin max
    # c = 10 # alphas constraint
    # kernel = 'linear' # kernel type, options: ('linear','rbf','linrbf')
    # gamma = None # gamma parameter for rbf-kernel and linrbf-kernel
    # lck = 1 # constant for kernel linear combination, 0<=lck<=1
    umbral = 1E-5 # umbral to define a vector support
    
    nsamples,nfeatures = np.shape(X)
    onev = np.ones((nsamples,1))
    
    # Kernel matrix
    if kernel == 'linear':
        K = linear_kernel(X,X)
    elif kernel == 'rbf':
        K = rbf_kernel(X,X,gamma=gamma)
    elif kernel == 'linrbf':
        K = linrbf_kernel(X,X,gamma=gamma,lck=lck)
    
    # Optimization E-regression with MAPE
    alpha1 = cp.Variable((nsamples,1))
    alpha2 = cp.Variable((nsamples,1))
    
    Ev = np.reshape(y,(nsamples,1))*epsilon
    objective = cp.Minimize((1/2)*cp.quad_form(alpha1-alpha2, K) + Ev.T @ (alpha1+alpha2) - y.T @ (alpha1 - alpha2))
    
    # Constrains in matrix form
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
w_mape,b_mape = SVR_E_MAPE(Xm,y,epsilon=0.01,c=10,kernel='linear',gamma=None,lck=1)

#% Simular el modelo
y_mape = np.dot(Xm,w_mape)+b_mape

#% Visualizar los resultados
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x1m, x2m, y, c=y,s=5)
ax.scatter(x1m, x2m, y_mape, c='r',s=10)
ax.view_init(30, 0)
plt.show()

#%% Optimization classic v formulation E-regression usando cvxpy
def SVR_vE(X,y,epsilon=0.01,c=10,v=1,kernel='linear',gamma=None,lck=1):
    #    epsilon = 0.01 # margin max
    #    v = 1 # New term
    #    c = 10 # alphas constraint
    # kernel = 'linear' # kernel type, options: ('linear','rbf','linrbf')
    # gamma = None # gamma parameter for rbf-kernel and linrbf-kernel
    # lck = 1 # constant for kernel linear combination, 0<=lck<=1
    
    umbral = 1E-5 # vector support 
    nsamples,nfeatures = np.shape(X)
    onev = np.ones((nsamples,1))
    
    # Kernel matrix
    if kernel == 'linear':
        K = linear_kernel(X,X)
    elif kernel == 'rbf':
        K = rbf_kernel(X,X,gamma=gamma)
    elif kernel == 'linrbf':
        K = linrbf_kernel(X,X,gamma=gamma,lck=lck)
    
    # Optimization formulation vE-regression 
    alpha1 = cp.Variable((nsamples,1))
    alpha2 = cp.Variable((nsamples,1))
    
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
w_vE,b_vE = SVR_vE(Xm,y,epsilon=0.01,c=10,v=1,kernel='linear',gamma=None,lck=1)

#% Simular el modelo
y_vE = np.dot(Xm,w_vE)+b_vE

#% Visualizar los resultados
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x1m, x2m, y, c=y,s=5)
ax.scatter(x1m, x2m, y_vE, c='r',s=10)
ax.view_init(30, 0)
plt.show()

#%% Optimization v formulation MAPE-regression usando cvxpy
def SVR_vMAPE(X,y,epsilon=0.01,c=10,v=1,kernel='linear',gamma=None,lck=1):
    #    epsilon = 0.01 # margin max
    #    v = 1 # New term
    #    c = 10 # alphas constraint
    # kernel = 'linear' # kernel type, options: ('linear','rbf','linrbf')
    # gamma = None # gamma parameter for rbf-kernel and linrbf-kernel
    # lck = 1 # constant for kernel linear combination, 0<=lck<=1
    
    umbral = 1E-5 # vector support 
    nsamples,nfeatures = np.shape(X)
    onev = np.ones((nsamples,1))
    
    # Kernel matrix
    if kernel == 'linear':
        K = linear_kernel(X,X)
    elif kernel == 'rbf':
        K = rbf_kernel(X,X,gamma=gamma)
    elif kernel == 'linrbf':
        K = linrbf_kernel(X,X,gamma=gamma,lck=lck)
    
    # Optimization formulation vE-regression with MAPE
    alpha1 = cp.Variable((nsamples,1))
    alpha2 = cp.Variable((nsamples,1))
    
    objective = cp.Minimize((1/2)*cp.quad_form(alpha1-alpha2, K) - y.T @ (alpha1 - alpha2))
    
    # Restricciones forma matricial
    G = np.float64(np.concatenate((np.identity(nsamples),-np.identity(nsamples))))
    h=np.float64(np.concatenate((100*c/np.reshape(y,(nsamples,1)),np.zeros((nsamples,1)))))
    constraints = [onev.T @ (alpha1-alpha2) == 0,
                   (y/100).T @ (alpha1+alpha2) == c*v,
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
w_vmape,b_vmape = SVR_vMAPE(Xm,y,epsilon=0.01,c=10,v=1,kernel='linear',gamma=None,lck=1)

#% Simular el modelo
y_vmape = np.dot(Xm,w_vmape)+b_vmape

#% Visualizar los resultados
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x1m, x2m, y, c=y,s=5)
ax.scatter(x1m, x2m, y_vmape, c='r',s=10)
ax.view_init(30, 0)
plt.show()

#%% Evaluacion de ambas implementaciones
rmse_ereg,mape_ereg = mean_squared_error(y,y_Ereg),mean_absolute_percentage_error(y,y_Ereg)
rmse_mape,mape_mape = mean_squared_error(y,y_mape),mean_absolute_percentage_error(y,y_mape)
rmse_vE,mape_vE = mean_squared_error(y,y_vE),mean_absolute_percentage_error(y,y_vE)
rmse_vmape,mape_vmape = mean_squared_error(y,y_vmape),mean_absolute_percentage_error(y,y_vmape)

#%%
print('\n\n\t\t\t RMSE\t\t MAPE\n Formulation Ereg\t %0.4f\t\t %0.4f\n Formulation Emape\t %0.4f\t\t %0.4f\n Formulation vE\t\t %0.4f\t\t %0.4f\n Formulation vmape\t %0.4f\t\t %0.4f'%(rmse_ereg,mape_ereg,rmse_mape,mape_mape,rmse_vE,mape_vE,rmse_vmape,mape_vmape))