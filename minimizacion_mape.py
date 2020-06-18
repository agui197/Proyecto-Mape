from numpy import mean, abs, shape, ones, float64, concatenate, identity, zeros, array, \
    sum, transpose, mean, tile, dot, reshape, random, linspace, meshgrid, ravel, c_, \
        arange, sin, cos, pi, argmin, sqrt
from pandas import DataFrame, get_dummies, read_excel, concat
from scipy.ndimage.interpolation import shift
from scipy import signal
from sklearn.metrics.pairwise import linear_kernel
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from cvxpy import Variable, Minimize, quad_form, Problem



def mean_absolute_percentage_error(y_true, y_pred):
    return mean(abs((y_true - y_pred) / y_true)) * 100


def SVR_E(X, y, epsilon=0.01, c=10, kernel='linear', gamma=None, lck=1):
    umbral = 1E-5 
    
    nsamples, nfeatures = shape(X)
    onev = ones((nsamples, 1))
    
    if kernel == 'linear':
        K = linear_kernel(X, X)
    elif kernel == 'rbf':
        K = rbf_kernel(X, X, gamma=gamma)
    elif kernel == 'linrbf':
        K = linrbf_kernel(X, X, gamma=gamma, lck=lck)

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
    x_sv = X[indx[:, 0]]
    y_sv = y[indx[:, 0]]
    
    
    w = sum(transpose(tile(alpha_sv, (nfeatures, 1))) * x_sv, axis=0)
    b = mean(y_sv - dot(x_sv, w))
    
    return w, b

def SVR_E_MAPE(X, y, epsilon=0.01, c=10, kernel='linear', gamma=None, lck=1):
    umbral = 1E-5 
    
    nsamples, nfeatures = shape(X)
    onev = ones((nsamples, 1))
    
    if kernel == 'linear':
        K = linear_kernel(X, X)
    elif kernel == 'rbf':
        K = rbf_kernel(X, X, gamma=gamma)
    elif kernel == 'linrbf':
        K = linrbf_kernel(X, X, gamma=gamma, lck=lck)
    
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
    x_sv = X[indx[:, 0]]
    y_sv = y[indx[:, 0]]
    
    
    w = sum(transpose(tile(alpha_sv, (nfeatures, 1))) * x_sv, axis=0)
    b = mean(y_sv - dot(x_sv, w))
    
    return w, b

def SVR_vE(X, y, epsilon=0.01, c=10, v=1, kernel='linear', gamma=None, lck=1):
    umbral = 1E-5 
    nsamples, nfeatures = shape(X)
    onev = ones((nsamples, 1))
    
    if kernel == 'linear':
        K = linear_kernel(X, X)
    elif kernel == 'rbf':
        K = rbf_kernel(X, X, gamma=gamma)
    elif kernel == 'linrbf':
        K = linrbf_kernel(X, X, gamma=gamma, lck=lck)
    
    alpha1 = Variable((nsamples, 1))
    alpha2 = Variable((nsamples, 1))
    
    objective = Minimize((1/2) * quad_form(alpha1 - alpha2, K) - y.T @ (alpha1 - alpha2))
    
    G = float64(concatenate((identity(nsamples), -identity(nsamples))))
    h = float64(concatenate((c * ones((nsamples, 1)), zeros((nsamples, 1)))))
    
    constraints = [onev.T @ (alpha1-alpha2) == 0,
                   onev.T @ (alpha1+alpha2) == c * v,
                   G @ alpha1 <= h,
                   G @ alpha2 <= h]
    
    prob = Problem(objective, constraints)
    result = prob.solve()
    
    alpha1 = array(alpha1.value)
    alpha2 = array(alpha2.value)
    alphas = alpha1 - alpha2
    indx = abs(alphas) > umbral
    alpha_sv = alphas[indx]
    x_sv = X[indx[:, 0]]
    y_sv = y[indx[:, 0]]
    
    
    w = sum(transpose(tile(alpha_sv, (nfeatures, 1))) * x_sv, axis=0)
    b = mean(y_sv - dot(x_sv, w))
    
    return w, b

def SVR_vMAPE(X, y, epsilon=0.01, c=10, v=1, kernel='linear', gamma=None, lck=1):
    umbral = 1E-5
    nsamples, nfeatures = shape(X)
    onev = ones((nsamples, 1))
    
    if kernel == 'linear':
        K = linear_kernel(X, X)
    elif kernel == 'rbf':
        K = rbf_kernel(X, X, gamma=gamma)
    elif kernel == 'linrbf':
        K = linrbf_kernel(X, X, gamma=gamma, lck=lck)
    
    alpha1 = Variable((nsamples, 1))
    alpha2 = Variable((nsamples, 1))
    
    objective = Minimize((1/2) * quad_form(alpha1 - alpha2, K) - y.T @ (alpha1 - alpha2))
    
    G = float64(concatenate((identity(nsamples), -identity(nsamples))))
    h = float64(concatenate((100 * c / reshape(y, (nsamples, 1)), zeros((nsamples, 1)))))
    constraints = [onev.T @ (alpha1-alpha2) == 0,
                   (y / 100).T @ (alpha1+alpha2) == c * v,
                   G @ alpha1 <= h,
                   G @ alpha2 <= h]
    
    prob = Problem(objective, constraints)
    result = prob.solve()
    
    alpha1 = array(alpha1.value)
    alpha2 = array(alpha2.value)
    alphas = alpha1 - alpha2
    indx = abs(alphas) > umbral
    alpha_sv = alphas[indx]
    x_sv = X[indx[:, 0]]
    y_sv = y[indx[:, 0]]
    
    w = sum(transpose(tile(alpha_sv, (nfeatures, 1))) * x_sv, axis=0)
    b = mean(y_sv - dot(x_sv, w))
    
    return w,b

def cargar():
    consumofeb = read_excel('C:/Users/tripl/Downloads/EntregaFinal/Consumo_feb.xlsx').set_index("fecha").loc["2007-01-01":"2020-03-30"]
    consumofeb = consumofeb.drop(["prom", "Lluvia", "Velocidad_viento"], 1)
    
    consumo = read_excel('C:/Users/tripl/Downloads/EntregaFinal/Data1.xlsx').set_index("fecha").loc["2007-01-01":"2020-03-30"]
    while True:
        if len(consumofeb.index) != len(consumo.index): 
            raise RuntimeError("Climatic variable and consumption variable length does not match")
            break
        else: return consumo, consumofeb

def kronecker(data1:'Dataframe 1', data2:'Dataframe 2'):
    Combinacion = DataFrame()
    d1 = DataFrame()

    for i in data2.columns:
        d1 = data1.multiply(data2[f"{i}"] , axis="index")
        d1.columns = [f'{i}_{j}' for j in data1.columns]
        Combinacion = concat([Combinacion, d1], axis=1)
    return Combinacion

def difs():
    inv1 = [310.5 - 365.25, 8.6529, 138.941176470588]
    pr = [8.6529, 138.941176470588, 221.9375]
    vr = [138.941176470588, 221.9375, 310.5]
    ot = [221.9375, 310.5, 365.25 + 8.6529]
    ot2 = [221.9375 - 365.25, 310.5 - 365.25, 8.6529]
    inv2 = [310.5, 365.25 + 8.6529, 365.25 + 138.941176470588]

    height = {'Invierno1': inv1,
              'Primavera': pr,
              'verano': vr,
              'Otoño': ot,
              'Otoño2': ot2,
              'Invierno2': inv2,} 
    return height 
    
def triangular():
    ma = DataFrame()
    x = consumofeb.index.dayofyear.tolist()

    height = difs()
    for hei in height:
        ba = DataFrame(x, columns=["x"])
        ba["a"] = height[hei][0]
        ba["b"] = height[hei][1]
        ba["c"] = height[hei][2]
        
        ca = DataFrame()
        ca["bo"] = (ba.x - ba.a) / (ba.b - ba.a)
        ca["ba"] = (ba.c - ba.x) / (ba.c - ba.b)
        ca = ca.min(axis=1)
        ca[ca < 0] = 0
        ma = concat([ma, ca], axis=1) 
        
    ma.index = consumofeb.index
    ma.columns = height.keys()
    return ma

def pba(j, n):
    f, asd = signal.periodogram(consumofeb[j], 1)

    picos = DataFrame(asd, 1 / (f), columns=["potencia"])
    picos = picos.sort_values(by="potencia", ascending=False).reset_index().head(12)
    picos.columns = ["periodo", "potencia"]
    
    sencos = DataFrame(index=consumofeb.index)
    t = arange(1, len(consumofeb) + 1)
    sencos["t"] = t
    for i  in  picos.periodo:
        sencos[f"{i:.2f}_sen"] = abs(sin(((2 * pi) / (i)) * t))
        sencos[f"{i:.2f}_cos"] = abs(cos(((2 * pi) / (i)) * t))
    sencos['ones'] = 1
    sencos['sen1'] = abs(sin(((2 * pi) / (365.25)) * t))
    sencos['cos1'] = abs(cos(((2 * pi) / (365.25)) * t))
    
    sencos = kronecker(sencos, triangular())

    X_train, X_test, y_train, y_test = train_test_split(
                                        sencos, consumofeb[j], test_size=n, shuffle=False)
    
    reg = LinearRegression(n_jobs=-1).fit(X_train, y_train)
    y_predict = reg.predict(X_test)
        
    nov = DataFrame(
        concatenate([reg.predict(X_train), y_predict]), index=consumofeb.index, columns=[f"{j}"])

    return nov

def modelosclima(n):
    ca = DataFrame(index=consumo.index)
    for i in ["mín", "max", "Nubosidad"]:
        ca = concat([ca, pba(i, n)], axis=1)
    return ca

def festivos():
    festivos = read_excel('C:/Users/tripl/Downloads/EntregaFinal/Festivos.xlsx')
    festivos2 = read_excel('C:/Users/tripl/Downloads/EntregaFinal/Festivos2.xlsx')

    Dum = DataFrame(index = consumo.index)

    for col in festivos.columns:
        Dum[f"{col}"] = consumo.index.isin(festivos[f"{col}"])
        Dum[col] = Dum[col].replace([False, True], [0, 1])
        clean = Dum[col].values
        Dum[col] = Dum[col] + shift(clean, 1, cval=0) * .4
        Dum[col] = Dum[col] + shift(clean, -1, cval=0) * .4
        Dum[col] = Dum[col] + shift(clean, 2, cval=0) * .1
        Dum[col] = Dum[col] + shift(clean, -2, cval=0) * .1
        Dum[col] = Dum[col] / 2

    Dum["ones"] = 1
    Dum["t"] = arange(1, len(consumo.index) + 1)    
    
    for col in festivos2.columns:
        Dum[col] = consumo.index.isin(festivos2[col])
    Dum = Dum.replace([False, True], [0, 1])
    return Dum

def dummies(n):   
    consumo2 = DataFrame(index=consumo.index)
    consumo2["num"] = arange(1,len(consumo.index) + 1)
    consumo2["day"] = consumo.index.weekday
    consumo2["eureka1"] = consumo2.day**3
    consumo2["eureka2"] = modclima["max"] * consumo2.num
    consumo2["month"] = consumo.index.month
    X = consumo2.join(modclima)
    
    X_norm = X / X.max()
    
    X_norm["gplearn1"] = cos(X_norm.day) * X_norm.num
    X_norm["gplearn2"] = X_norm.num * X_norm["mín"]
    X_norm["gplearn3"] = 3.319**X_norm["max"] * cos(X_norm.day)
    X_norm["gplearn8"] = 3.319**X_norm["max"] * X_norm.num 
    X_norm["gplearn4"] = cos(X_norm.day) * sin(X_norm.day)
    X_norm["gplearn6"] = X_norm["max"]**2 * X_norm["mín"]**2
    X_norm["gplearn7"] = 1 / cos(X_norm.day)
    X_norm["gplearn10"] = X_norm["max"] * X_norm["mín"]**3 * cos(X_norm.month)**2
    
    X_norm = get_dummies(X_norm, columns=["month", "day"], prefix=["month", "day"], drop_first=True)
    
    X_norm = X_norm.drop(["mín", "max", "Nubosidad"], axis=1)

    X_norm = X_norm.join(festivos())
    X_norm = X_norm.join(triangular())
    
    X_norm["fin"] = consumo.index.weekday
    X_norm["entre"] = consumo.index.weekday
    X_norm["fin"] = X_norm["fin"].replace([0, 1, 2, 3, 4, 5, 6], [.5, 0, 0, 0, .5, 1, 1])
    X_norm["entre"] = X_norm["entre"].replace([0, 1, 2, 3, 4, 5, 6], [.5, 1, 1, 1, .5, 0, 0])
    return X_norm

def separar(n, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = n, shuffle = False)
    return (X_train, X_test), (y_train, y_test)

def reg(n:'n = 7', pre_forc:'general o forecasting'):
    y = consumo
    X = dummies(n)
    
    Xo, yo = separar(n, X, y)  
    
    while True:
        if pre_forc == "general": 
            pre_forc = 0
            break
        elif pre_forc == "forecasting": 
            pre_forc = 1
            break
        else: 
            raise NameError(f"expected 'general' or 'forecasting', got '{pre_forc}' instead")
            break
            
    reg = LinearRegression(n_jobs =-1).fit(Xo[0], yo[0])
    y_predict = reg.predict(Xo[pre_forc])
    
    comp = DataFrame(y_predict, columns=["predict"], index=yo[pre_forc].index)
    comp["real"] = yo[pre_forc]
    comp["resta"] = comp.real - comp.predict
    return comp
    

def perio(n, fg):
    reg1 = reg(n, fg)
    
    f, asd = signal.periodogram(reg1.resta, 1)
    
    picos = DataFrame(asd, columns=["potencia"])
    picos["periodo"] = 1 / f
    picos = picos.sort_values(by="potencia", ascending=False).reset_index(drop=True).head(12)
    return picos[2:12]

def kron(n):
    X = dummies(n) 
    
    sencos = DataFrame()
    t = arange(1,len(consumo.MWh)+1)
    sencos["t"] = t
    
    p = perio(n, "general")
    
    for i  in  p.periodo:
        sencos["{}_sen".format(i)] = abs(sin(((2 * pi) / (i)) * t))
        sencos["{}_cos".format(i)] = abs(cos(((2 * pi) / (i)) * t))

    sencos['ones'] = 1
    sencos['sen'] = abs(sin(((2 * pi) / 14) * t))
    sencos['cos'] = abs(cos(((2 * pi) / 14) * t))
    sencos['sen1'] = abs(sin(((2 * pi) / (365.25)) * t))
    sencos['cos1'] = abs(cos(((2 * pi) / (365.25)) * t))
    sencos['sen2'] = abs(sin(((2 * pi) / (28)) * t))
    sencos['cos2'] = abs(cos(((2 * pi) / (28)) * t))
    
    sencos.index = consumo.index
    
    combination = kronecker(X, sencos)
    return combination.join(modclima)

def f(x, i):
    epsilon, c, v = x[0], x[1], x[2]

    if i == 0:
        try:
            w_mape, b_mape = SVR_E_MAPE(Xm, y, epsilon=epsilon, c=c)
            y_mape = dot(Xm, w_mape) + b_mape

            rmse_mape, mape_mape = mean_squared_error(y, y_mape), mean_absolute_percentage_error(y, y_mape)
            
        except: 
            mape_mape = 100000
        print(mape_mape)
        print(x)
        return mape_mape
    elif i == 1:
        try:
            w_vmape, b_vmape = SVR_vMAPE(Xm, y, epsilon=epsilon, c=c, v=v)
            y_vmape = dot(Xm, w_vmape) + b_vmape

            rmse_vmape, mape_vmape = mean_squared_error(y, y_vmape), mean_absolute_percentage_error(y, y_vmape)
            
        except: 
            mape_vmape = 100000
        print(mape_vmape)
        print(x)
        return mape_vmape
    elif i == 2:
        try:
            w_Ereg, b_Ereg = SVR_E(Xm, y, epsilon=epsilon, c=c)
            y_Ereg = dot(Xm, w_Ereg) + b_Ereg

            rmse_ereg, mape_ereg = mean_squared_error(y, y_Ereg), mean_absolute_percentage_error(y, y_Ereg)
        except:
            mape_ereg = 100000

        print(mape_ereg)
        print(x)
        return mape_ereg
    else:
        try:
            w_vE, b_vE = SVR_vE(Xm, y, epsilon=epsilon, c=c, v=v)
            y_vE = dot(Xm, w_vE) + b_vE

            rmse_vE, mape_vE = mean_squared_error(y, y_vE), mean_absolute_percentage_error(y, y_vE)
            
        except:
            mape_vE = 100000
        print(mape_vE)
        print(x)
        return mape_vE


def pso(np, iterations, formulation):
    x1p = reshape([random.uniform(0, .01, np), random.randint(1, 100, np), random.randint(1, 10, np)], (3,np)).T
    
    if formulation == 0:
        x1p[0] = [0.005371852, 29836008.08, 2.55952]
        x1p[1] = [0.007994567, 107409535.5, 2.903568]
    elif formulation == 1:
        x1p[0] = [0.005530169, 359906.908, 7.558]
        x1p[1] = [0.002981206, 215980.1448, 17.0066]
    elif formulation == 2:
        x1p[0] = [0.009445345, 701, 4.6]
    else:
        x1p[0] = [0.002321687, 2904.54, 7.2]
        x1p[1] = [0.006036079, 364.8, 11.2]

    x1pg = [.01, 10, 1]

    vx1 = x1p
    x1pL = x1p

    fxpg = 1000000
    fxpL = ones([np, 1]) * fxpg

    c1 = .3
    c2 = .3

    for i in range(iterations):
        fx = ones([np, 1])
        a = 1000
        
        for variables, j in zip(x1p, range(np)):
            print(i, j)
            fx[j] = f(variables, formulation)
            
        ind = argmin(fx)
        val = fx[ind]
        print('previous global ', fxpg, 'val ',val)
        if val < fxpg:
            x1pg = x1p[ind]
            fxpg = val
        
        for p in range(np):
            if fx[p] < fxpL[p]:
                x1pL[p] = x1p[p]

        for p in range(np):
            vx1[p] = vx1[p] \
                + c1 * array([random.uniform(0,.01), random.randint(1,100), random.randint(1,10)]) * (x1pg - x1p[p]) \
                + c2 * array([random.uniform(0,.01), random.randint(1,100), random.randint(1,10)]) * (x1pL[p] - x1p[p])
    
    return x1pg[0], x1pg[1], x1pg[2]

n = 7
c = 7
consumo, consumofeb = cargar()
modclima = modelosclima(c)
X = kron(n)
y = ravel(consumo)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=n, shuffle=False)

for i in range(7):
    Xm = X_train[:200 + i]
    y = y_train[:200 + i]
    np = 25
    iterations = 7

    X_test = X_train[200 + i:207 + i]
    y_test = y_train[200 + i:207 + i]

    kernel = 'linear'
    gamma = None
    lck = 1

    epsilon, c, v = pso(np, iterations, 2)
    w_Ereg, b_Ereg = SVR_E(Xm, y, epsilon=epsilon, c=c, kernel=kernel, gamma=gamma, lck=lck)
    y_Ereg = dot(Xm, w_Ereg) + b_Ereg

    epsilon1, c1, v1 = pso(np, iterations, 0)
    w_mape, b_mape = SVR_E_MAPE(Xm, y, epsilon=epsilon1, c=c1, kernel=kernel, gamma=gamma, lck=lck)
    y_mape = dot(Xm, w_mape) + b_mape

    epsilon2, c2, v2 = pso(np, iterations, 3)
    w_vE, b_vE = SVR_vE(Xm, y, epsilon=epsilon2, c=c2, v=v2, kernel=kernel, gamma=gamma, lck=lck)
    y_vE = dot(Xm, w_vE) + b_vE

    epsilon3, c3, v3 = pso(np, iterations, 1)
    w_vmape, b_vmape = SVR_vMAPE(Xm, y, epsilon=epsilon3, c=c3, v=v3, kernel=kernel, gamma=gamma, lck=lck)
    y_vmape = dot(Xm, w_vmape) + b_vmape

    rmse_ereg, mape_ereg = mean_squared_error(y, y_Ereg), mean_absolute_percentage_error(y, y_Ereg)
    rmse_mape, mape_mape = mean_squared_error(y, y_mape), mean_absolute_percentage_error(y, y_mape)

    rmse_vE, mape_vE = mean_squared_error(y, y_vE), mean_absolute_percentage_error(y, y_vE)
    rmse_vmape, mape_vmape = mean_squared_error(y, y_vmape), mean_absolute_percentage_error(y, y_vmape)

    results = DataFrame([[sqrt(rmse_ereg), mape_ereg], 
                        [sqrt(rmse_mape), mape_mape], 
                        [sqrt(rmse_vE), mape_vE], 
                        [sqrt(rmse_vmape), mape_vmape]])

    hiperparams = DataFrame([[epsilon, c, v],
                            [epsilon1, c1, v1],
                            [epsilon2, c2, v2],
                            [epsilon3, c3, v3]])

    results.to_csv('results_train_'+kernel+'_D'+str(i)+'.csv')
    hiperparams.to_csv('hiperparams_'+kernel+'_D'+str(i)+'.csv')

    print('\n\n\t\t\t\t\t RMSE\t\t MAPE\n \
            Formulation Ereg\t %0.4f\t\t %0.4f\n \
            Formulation Emape\t %0.4f\t\t %0.4f\n \
            Formulation vE\t\t %0.4f\t\t %0.4f\n \
            Formulation vmape\t %0.4f\t\t %0.4f'%(
                sqrt(rmse_ereg), mape_ereg,
                sqrt(rmse_mape), mape_mape,
                sqrt(rmse_vE), mape_vE,
                sqrt(rmse_vmape), mape_vmape))

    y_EregR = dot(X_test, w_Ereg) + b_Ereg
    y_mapeR = dot(X_test, w_mape) + b_mape
    y_vER = dot(X_test, w_vE) + b_vE
    y_vmapeR = dot(X_test, w_vmape) + b_vmape

    rmse_ereg, mape_ereg = mean_squared_error(y_test, y_EregR), mean_absolute_percentage_error(y_test, y_EregR)
    rmse_mape, mape_mape = mean_squared_error(y_test, y_mapeR), mean_absolute_percentage_error(y_test, y_mapeR)

    rmse_vE, mape_vE = mean_squared_error(y_test, y_vER), mean_absolute_percentage_error(y_test, y_vER)
    rmse_vmape, mape_vmape = mean_squared_error(y_test, y_vmapeR), mean_absolute_percentage_error(y_test, y_vmapeR)

    print('\n\n\t\t\t\t\t RMSE\t\t MAPE\n \
            Formulation Ereg\t %0.4f\t\t %0.4f\n \
            Formulation Emape\t %0.4f\t\t %0.4f\n \
            Formulation vE\t\t %0.4f\t\t %0.4f\n \
            Formulation vmape\t %0.4f\t\t %0.4f'%(
                sqrt(rmse_ereg), mape_ereg,
                sqrt(rmse_mape), mape_mape,
                sqrt(rmse_vE), mape_vE,
                sqrt(rmse_vmape), mape_vmape))

    Xm.to_csv('X_train_'+kernel+'_D'+str(i)+'.csv')
    X_test.to_csv('X_test_'+kernel+'_D'+str(i)+'.csv')
    ydf = DataFrame(array([y, y_Ereg, y_mape, y_vE, y_vmape]).T)
    yRdf = DataFrame(array([y_test, y_EregR, y_mapeR, y_vER, y_vmapeR]).T)

    ydf.to_csv('y_train_'+kernel+'_D'+str(i)+'.csv')
    yRdf.to_csv('y_test_'+kernel+'_D'+str(i)+'.csv')

    results = DataFrame([[sqrt(rmse_ereg), mape_ereg], 
                        [sqrt(rmse_mape), mape_mape], 
                        [sqrt(rmse_vE), mape_vE], 
                        [sqrt(rmse_vmape), mape_vmape]])

    results.to_csv('results_test_'+kernel+'_D'+str(i)+'.csv')
