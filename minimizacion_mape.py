from numpy import mean, abs, shape, ones, float64, concatenate, identity, zeros, array, \
    sum, transpose, mean, tile, dot, reshape, random, linspace, meshgrid, ravel, c_
from sklearn.metrics.pairwise import linear_kernel
from cvxpy import Variable, Minimize, quad_form, Problem
from sklearn.metrics import mean_squared_error


def mean_absolute_percentage_error(y_true, y_pred):
    return mean(abs((y_true - y_pred) / y_true)) * 100


def SVR_E(X, y, epsilon=0.01, c=10):
    umbral = 1E-5 
    
    nsamples, nfeatures = shape(X)
    onev = ones((nsamples, 1))
    
    K = linear_kernel(X, X)

    alpha1 = Variable((nsamples, 1))
    alpha2 = Variable((nsamples, 1))
    
    Ev = onev * epsilon
    objective = Minimize(
        (1/2) * quad_form(alpha1 - alpha2, K) + Ev.T @ (alpha1 + alpha2) - y.T @ (alpha1 - alpha2))
    
    G = float64(concatenate((identity(nsamples), -identity(nsamples))))
    h = float64(concatenate((c * ones((nsamples,1)), zeros((nsamples, 1)))))
    
    constraints = [onev.T @ (alpha1 - alpha2) == 0, G @ alpha1 <= h, G @ alpha2 <= h]
    
    prob = Problem(objective, constraints)
    result = prob.solve()
    
    alpha1 = array(alpha1.value)
    alpha2 = array(alpha2.value)
    alphas = alpha1 - alpha2
    indx = abs(alphas) > umbral
    alpha_sv = alphas[indx]
    x_sv = X[indx[:, 0], :]
    y_sv = y[indx[:, 0]]
    
    
    w = sum(transpose(tile(alpha_sv, (nfeatures, 1))) * x_sv, axis=0)
    b = mean(y_sv - dot(x_sv, w))
    
    return w, b

def SVR_E_MAPE(X,y,epsilon=0.01,c=10):
    umbral = 1E-5 
    
    nsamples, nfeatures = shape(X)
    onev = ones((nsamples, 1))
    
    K = linear_kernel(X, X)
    
    alpha1 = Variable((nsamples, 1))
    alpha2 = Variable((nsamples, 1))
    
    Ev = reshape(y, (nsamples, 1)) * epsilon
    objective = Minimize(
        (1/2) * quad_form(alpha1 - alpha2, K) + Ev.T @ (alpha1 + alpha2) - y.T @ (alpha1 - alpha2))
    
    G = float64(concatenate((identity(nsamples), -identity(nsamples))))
    h = float64(concatenate((c / reshape(y, (nsamples, 1)), zeros((nsamples, 1)))))
    constraints = [onev.T @ (alpha1 - alpha2) == 0, G @ alpha1 <= h, G @ alpha2 <= h]
    
    prob = Problem(objective, constraints)
    result = prob.solve()
    
    
    alpha1 = array(alpha1.value)
    alpha2 = array(alpha2.value)
    alphas = alpha1 - alpha2
    indx = abs(alphas) > umbral
    alpha_sv = alphas[indx]
    x_sv = X[indx[:, 0], :]
    y_sv = y[indx[:, 0]]
    
    
    w = sum(transpose(tile(alpha_sv, (2, 1))) * x_sv, axis=0)
    b = mean(y_sv - dot(x_sv, w))
    
    return w, b


random.seed(1)
n = 20
x1 = linspace(1, 20, n)
x2 = linspace(1, 20, n)
X1, X2 = meshgrid(x1, x2)
Y = 2 * X1 + 3 * X2 + 40 + (5 * random.rand(X1.shape[0], X1.shape[0]) - 2.5)

x1m = ravel(X1.T)
x2m = ravel(X2.T)
Xm = c_[x1m, x2m]
y = ravel(Y.T)

w_Ereg, b_Ereg = SVR_E(Xm, y, epsilon=0.01, c=10)

y_Ereg = w_Ereg[0] * x1m + w_Ereg[1] * x2m + b_Ereg

w_mape, b_mape = SVR_E_MAPE(Xm, y, epsilon=0.01, c=10)

y_mape = w_mape[0] * x1m + w_mape[1] * x2m + b_mape

rmse1, mape1 = mean_squared_error(y, y_Ereg), mean_absolute_percentage_error(y, y_Ereg)
rmse2, mape2 = mean_squared_error(y, y_mape), mean_absolute_percentage_error(y, y_mape)

print('\n\n\t Obj E_reg\t Obj MAPE\n RMSE\t %0.4f\t\t %0.4f\n MAPE\t %0.4f\t\t %0.4f'%(
        rmse1,
        rmse2,
        mape1,
        mape2))