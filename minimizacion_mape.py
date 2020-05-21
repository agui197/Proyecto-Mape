from numpy import mean, abs, shape, ones, float64, concatenate, identity, zeros, array, \
    sum, transpose, mean, tile, dot, reshape, random, linspace, meshgrid, ravel, c_, \
        arange, sin, cos, pi
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


def SVR_E(X, y, epsilon=0.01, c=10):
    print(shape(X),shape(y))
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
    x_sv = X[indx[:, 0]]
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

def cargar():
    consumofeb = read_excel('C:/Users/tripl/Downloads/EntregaFinal/Consumo_feb.xlsx').set_index("fecha").loc["2007-01-01":"2020-03-30"]
    consumofeb = consumofeb.drop(["prom", "Lluvia", "Velocidad_viento"], 1)
    
    consumo = read_excel('C:/Users/tripl/Downloads/EntregaFinal/Data1.xlsx').set_index("fecha").loc["2007-01-01":"2020-03-30"]
    while True:
        if len(consumofeb.index) != len(consumo.index): 
            raise RuntimeError("Climatic variable and consumption variable length does not match")
            break
        else: return consumo, consumofeb

def kronecker(data1:'Dataframe 1',data2:'Dataframe 2'):
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

n = 7
c = 7
consumo, consumofeb = cargar()
modclima = modelosclima(c)
X = kron(n)
y = ravel(consumo)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=n, shuffle=False)
w_Ereg, b_Ereg = SVR_E(X_train[:200], y_train[:200], epsilon=0.01, c=10)
# w_mape, b_mape = SVR_E_MAPE(X_train, y_train, epsilon=0.01, c=10)


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