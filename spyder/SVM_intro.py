# -*- coding: utf-8 -*-
"""
Created on Wed May  6 12:13:54 2020

@author: RIEMANNRUIZ
"""

#Importar librerias
from numpy import linspace, meshgrid, matrix, ravel, c_, multiply, \
    ones, float64, reshape, concatenate, identity, zeros, array, sum, mean, dot
from matplotlib.pyplot import scatter, show, xlim, ylim, plot
from sklearn.metrics.pairwise import linear_kernel
from cvxopt import matrix as mat
from cvxopt import solvers

#%% Generacion de los datos
n = 20
x1 = linspace(-10, 10, n)
x2 = linspace(-10, 10, n)
X1, X2 = meshgrid(x1, x2)
Y = 2 * ((matrix(X1)**3 - 3 * X1) > X2) - 1

x1m = ravel(X1.T)
x2m = ravel(X2.T)
Xm = c_[x1m, x2m]
y = ravel(Y.T)
Ym = matrix(y).T * matrix(y)

#%% Visualizar los datos
scatter(x1m, x2m, c=y)
show()

#%% Kernel matrix

K = linear_kernel(Xm, Xm)

#%% Optimization Variables
H = multiply(K, Ym)
onev = -ones((y.shape[0], 1))
Aeq = float64(reshape(y, (1, y.shape[0])))
beq = 0.0
c = 0.1
G = float64(concatenate((identity(y.shape[0]), -identity(y.shape[0]))))
# G = np.float64(np.identity(y.shape[0]))
h = float64(concatenate((c * ones((y.shape[0], 1)), zeros((y.shape[0], 1)))))

#%% Solucion al problema de optimizacion cuadratica

P = mat(H)
q = mat(onev)
A = mat(Aeq)
b = mat(beq)
G = mat(G)
h = mat(h)
sol=solvers.qp(P=P, q=q, G=G, h=h, A=A, b=b)

alphas = array(sol['x'])

#%% Support vector values
indx = alphas > 1e-10
alpha_sv = alphas[indx]
x_sv = Xm[indx[:, 0], :]
y_sv = y[indx[:, 0]]

#%% Calculo de las w
w = sum(c_[alpha_sv * y_sv, alpha_sv * y_sv] * x_sv, axis=0)
b = mean(1 / y_sv - dot(x_sv, w))

#%% Visualizar los datos
scatter(x1m, x2m, c=y)
plot(x1,(-w[0] * x1 - b) / w[1])
xlim(-10, 10)
ylim(-10, 10)
show()

