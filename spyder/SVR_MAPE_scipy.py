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

#%% Generate X data points
def genX(lmin=0,lmax=10,npoints=29):
    x1 = np.linspace(lmin,lmax,npoints)
    x2 = np.linspace(lmin,lmax,npoints)
    X1,X2 = np.meshgrid(x1,x2)
    x1m = np.ravel(X1.T)
    x2m = np.ravel(X2.T)
    Xm = np.c_[x1m,x2m]
    return Xm


#%% Test function 1 (Hyperplane)
def testfunction1(X,noise=False):
    # Modelo: Y = 2*X1+3*X2+40
    X1 = X[:,0]
    X2 = X[:,1]
    if noise:
        Y = 2*X1+3*X2+40+(5*np.random.rand(X.shape[0])-2.5)
    else:
        Y = 2*X1+3*X2+40
    
    y = np.ravel(Y.T)
    return y

#%% Test function 2
def testfunction2(X,noise=False):
    # Modelo: Y = (x1^2-x2^2)*sin(0.5*x1)+10
    X1 = X[:,0]
    X2 = X[:,1]
    if noise:
        Y = 100+(X1**2-X2**2)*np.sin(0.5*X1)+(5*np.random.rand(X.shape[0])-2.5)
    else:
        Y = 100+(X1**2-X2**2)*np.sin(0.5*X1)
    
    y = np.ravel(Y.T)
    return y

#%% Test function 3
def testfunction3(X,noise=False):
    # Modelo: Y = sin(sqrt(x1^2+x2^2))/sqrt(x1^2+x2^2)+10
    X1 = X[:,0]
    X2 = X[:,1]
    if noise:
        Y = 10+np.sin(np.sqrt(X1**2+X2**2))/np.sqrt(X1**2+X2**2)+(0.5*np.random.rand(X.shape[0])-0.25)
    else:
        Y = 10+np.sin(np.sqrt(X1**2+X2**2))/np.sqrt(X1**2+X2**2)
    
    y = np.ravel(Y.T)
    return y

#%% Test function 4
def testfunction4(X,noise=False):
    # Modelo: Y = x1^2+x2^2-np.cos(2*x1)-np.cos(2*x2)+10
    X1 = X[:,0]
    X2 = X[:,1]
    if noise:
        Y = 10+X1**2+X2**2-10*np.cos(2*X1)-10*np.cos(2*X2)+(5*np.random.rand(X.shape[0])-2.5)
    else:
        Y = 10+X1**2+X2**2-10*np.cos(2*X1)-10*np.cos(2*X2)
    
    y = np.ravel(Y.T)
    return y

#%% Generacion de un hyperplano
#np.random.seed(1)
lmin = 1
lmax = 10
n = 29
Xm = genX(lmin=lmin,lmax=lmax,npoints=n)
#y = testfunction1(X=Xm,noise=True)
#y = testfunction2(X=Xm,noise=True)
y = testfunction3(X=Xm,noise=False)
#y = testfunction4(X=Xm,noise=True)

nsamples=Xm.shape[0]

#%% Visualizar los datos
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(Xm[:,0],Xm[:,1], y, c=y)
plt.show()




#%% Kernel matrix
kernel = 'rbf' # seleccion del kernel que se quiere aplicar
lck = 0.5 # constant to kernel linear combination
gamma = None # parameter

def custom_kernel(X1,X2,kernel='linear',gamma=None,lck=1):
    # Kernel matrix
    if kernel == 'linear':
        K = linear_kernel(X1,X2)
    elif kernel == 'rbf':
        K = rbf_kernel(X1,X2,gamma=gamma)
    elif kernel == 'linrbf':
        K = lck*linear_kernel(X1,X2)+(1-lck)*rbf_kernel(X1,X2,gamma=gamma)
    return K

K = custom_kernel(Xm,Xm,kernel=kernel,gamma=gamma,lck=lck)

#%% Optimization E-regression usando scipy
epsilon = 0.01 # margin max
c = 10 # alphas constraint
onev = np.ones((nsamples,1))
error = 1E-5 # vector support 

#%%
from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.optimize import NonlinearConstraint
alphas = np.random.rand(nsamples*2)
n = np.shape(alphas)[0]
def objective_fun(x):
    
    n = np.shape(x)[0]
    alpha1 = np.reshape(x[0:int(n/2)],(int(n/2),1))
    alpha2 = np.reshape(x[int(n/2):n],(int(n/2),1))
    onev = np.ones((int(n/2),1))
    Ev = onev*epsilon
    objective = (1/2)*np.dot((alpha1-alpha2).T,np.dot(K,(alpha1-alpha2)))+np.dot(Ev.T,(alpha1+alpha2))-np.dot(y.T,(alpha1-alpha2))
    return objective[0][0]
objective_fun(alphas)

# Restricciones forma matricial
bounds = Bounds(np.zeros(n),c*np.ones(n))

def const_f(x):
    n = np.shape(alphas)[0]
    alpha1 = np.reshape(alphas[0:int(n/2)],(int(n/2),1))
    alpha2 = np.reshape(alphas[int(n/2):n],(int(n/2),1))
    onev = np.ones((int(n/2),1))
    return np.dot(onev.T,alpha1-alpha2)[0][0]

nonlinear_constraint = NonlinearConstraint(const_f, 0, 0)
#%%
res = minimize(objective_fun, alphas, method='trust-constr',
               constraints=nonlinear_constraint,
               options={'verbose': 1}, bounds=bounds)


#%%
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
indx = abs(alphas) > error
alpha_sv = alphas[indx]
x_sv = Xm[indx[:,0],:]
y_sv = y[indx[:,0]]

# Evaluacion del modelo
b_Ereg = np.mean(y_sv-np.dot(alpha_sv,custom_kernel(x_sv,x_sv,kernel=kernel,gamma=gamma,lck=lck)))
K_sv = custom_kernel(x_sv,Xm,kernel=kernel,gamma=gamma,lck=lck)
y_Ereg = np.dot(alpha_sv,K_sv)+b_Ereg


#w_Ereg = np.sum(np.c_[alpha_sv,alpha_sv]*x_sv,axis=0)
#b_Ereg = np.mean(y_sv-np.dot(x_sv,w_Ereg))
#
#print('w_Ereg=[%0.3f,%0.3f]'%(w_Ereg[0],w_Ereg[1]))
#print('b_Ereg=%0.3f'%b_Ereg)

##% Visualizar los resultados
#y_Ereg = w_Ereg[0]*Xm[:,0]+w_Ereg[1]*Xm[:,1]+b_Ereg

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(Xm[:,0],Xm[:,1], y, c=y,s=5)
ax.scatter(Xm[:,0],Xm[:,1], y_Ereg, c='r',s=10)
#ax.view_init(30, 0)
plt.show()






######################################
#%% Optimization E-MAPE usando cvxpy
epsilon = 0.01 # margin max
c = 10 # alphas constraint
onev = np.ones((nsamples,1))

error = 1E-5 # vector support 

alpha1 = cp.Variable((nsamples,1))
alpha2 = cp.Variable((nsamples,1))

#% Forma MAPE
Ev = np.reshape(y,(nsamples,1))*epsilon
objective = cp.Minimize((1/2)*cp.quad_form(alpha1-alpha2, K) + Ev.T @ (alpha1+alpha2) - y.T @ (alpha1 - alpha2))

# Restricciones forma matricial
G = np.float64(np.concatenate((np.identity(nsamples),-np.identity(nsamples))))
h=np.float64(np.concatenate((c/np.reshape(y,(nsamples,1)),np.zeros((nsamples,1)))))
constraints = [onev.T @ (alpha1-alpha2) == 0,
               G @ alpha1 <= h,
               G @ alpha2 <= h]

# The optimal objective value is returned by `prob.solve()`.
prob = cp.Problem(objective,constraints)
result = prob.solve()


alpha1 = np.array(alpha1.value)
alpha2 = np.array(alpha2.value)
alphas = alpha1-alpha2
indx = abs(alphas) > error
alpha_sv = alphas[indx]
x_sv = Xm[indx[:,0],:]
y_sv = y[indx[:,0]]


# Evaluacion del modelo
b_Emape = np.mean(y_sv-np.dot(alpha_sv,custom_kernel(x_sv,x_sv,kernel=kernel,gamma=gamma,lck=lck)))
K_sv = custom_kernel(x_sv,Xm,kernel=kernel,gamma=gamma,lck=lck)
y_mape = np.dot(alpha_sv,K_sv)+b_Emape

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(Xm[:,0],Xm[:,1], y, c=y,s=5)
ax.scatter(Xm[:,0],Xm[:,1], y_mape, c='r',s=10)
#ax.view_init(30, 0)
plt.show()

#%% Optimization classic v-formulation. E-regression usando cvxpy
epsilon = 0.01 # margin max
v = 1 # New term
c = 10 # alphas constraint
onev = np.ones((nsamples,1))
error = 1E-5 # vector support 

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
indx = abs(alphas) > error
alpha_sv = alphas[indx]
x_sv = Xm[indx[:,0],:]
y_sv = y[indx[:,0]]


# Evaluacion del modelo
b_vE = np.mean(y_sv-np.dot(alpha_sv,custom_kernel(x_sv,x_sv,kernel=kernel,gamma=gamma,lck=lck)))
K_sv = custom_kernel(x_sv,Xm,kernel=kernel,gamma=gamma,lck=lck)
y_vE = np.dot(alpha_sv,K_sv)+b_vE

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(Xm[:,0],Xm[:,1], y, c=y,s=5)
ax.scatter(Xm[:,0],Xm[:,1], y_vE, c='r',s=10)
#ax.view_init(30, 0)
plt.show()

#%% Optimization v-formulation E-MAPE usando cvxpy
epsilon = 0.01 # margin max
c = 10 # alphas constraint
v = 1 # New term
onev = np.ones((nsamples,1))

error = 1E-5 # vector support 

alpha1 = cp.Variable((nsamples,1))
alpha2 = cp.Variable((nsamples,1))

#% Forma MAPE
objective = cp.Minimize((1/2)*cp.quad_form(alpha1-alpha2, K) - y.T @ (alpha1 - alpha2))

# Restricciones forma matricial
G = np.float64(np.concatenate((np.identity(nsamples),-np.identity(nsamples))))
#h=np.float64(np.concatenate((c/np.reshape(y,(nsamples,1)),np.zeros((nsamples,1)))))
h=np.float64(np.concatenate(((100*c)/np.reshape(y,(nsamples,1)),np.zeros((nsamples,1)))))
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
indx = abs(alphas) > error
alpha_sv = alphas[indx]
x_sv = Xm[indx[:,0],:]
y_sv = y[indx[:,0]]


# Evaluacion del modelo
b_vmape = np.mean(y_sv-np.dot(alpha_sv,custom_kernel(x_sv,x_sv,kernel=kernel,gamma=gamma,lck=lck)))
K_sv = custom_kernel(x_sv,Xm,kernel=kernel,gamma=gamma,lck=lck)
y_vmape = np.dot(alpha_sv,K_sv)+b_vmape

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(Xm[:,0],Xm[:,1], y, c=y,s=5)
ax.scatter(Xm[:,0],Xm[:,1], y_vmape, c='r',s=10)
#ax.view_init(30, 0)
plt.show()

#%% Evaluacion de ambas implementaciones
rmse1,mape1 = mean_squared_error(y,y_Ereg),mean_absolute_percentage_error(y,y_Ereg)
rmse2,mape2 = mean_squared_error(y,y_mape),mean_absolute_percentage_error(y,y_mape)
print('\n\n\t Obj E_reg\t Obj MAPE\n RMSE\t %0.4f\t\t %0.4f\n MAPE\t %0.4f\t\t %0.4f'%(rmse1,rmse2,mape1,mape2))

#%%
import numpy as np
from scipy.optimize import minimize

def rosen(x):
    """The Rosenbrock function"""
    return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)

x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
res = minimize(rosen, x0, method='nelder-mead',
               options={'xatol': 1e-8, 'disp': True})

print(res.x)