"""
Practica 3: algoritmos de regresión no lineal 
Departamento de Ingeniería Eléctrica y Electronica, Ingeniería biomédica
Tecnológico Nacional de México [TecNM - Tijuana]
Blvd. Alberto Limón padilla s/n, C.P. 22454, Tijuana, B.C., México


Nombre del alumno: Frida Daniela Mendivil Ramirez 
Numero de Control: 20211983
Correo Institucional: l20211983@tectijuana.edu.mx


Asignatura: Gemelos Dígitales
Docente: Paul Antonio Valle Trujillo; paul.valle@tectijuana.edu.mx
"""
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.stats
import statistics as st
import warnings
warnings.filterwarnings("ignore")


data = pd.read_csv('data.csv')

T = data.iloc[:,0]; to=T.to_numpy()
X1 = data.iloc[:,1]; x1o=X1.to_numpy()
X2 = data.iloc[:,2]; x2o=X2.to_numpy()
xo = np.mean([x1o,x2o], axis=0)

print(data)
#%%%
def plotdata(t,x1,x2,xo):
    fig = plt.figure(); 
    fig = plt.figure(figsize = (8,4))
    plt.plot(t,x1,linestyle = 'none', marker = 'x',
             color = [0.11,0.30,0.42],label = '$x_1(t)$')
    plt.plot(t,x2,linestyle = 'none', marker = 'x',
             color = [0.65,0.27,0.34],label = '$x_2(t)$')
    plt.plot(t,xo, '-o', linewidth = 0.5, markersize = 3,
             color = [0.25,0.25,0.25],label = '$x_o(t)$')
    plt.xlabel('$t$ $[months]$')
    plt.ylabel('$x_i(t)$ $[cells]$')
    plt.xlim(0,72)
    xticks = np.arange(0,73,8)
    plt.xticks(xticks)
    plt.ylim(0,11E8)
    yticks = np.arange(0,12E8,1E8)
    plt.yticks(yticks)
    plt.legend(bbox_to_anchor=(1.01, 1.03), fontsize = 10,
               title= "$Experimental$ $data$", frameon = True)
    plt.show()
    fig.tight_layout()
    fig.savefig('python_data.pdf')
    
plotdata(to,x1o,x2o,xo)

#%%
def mdl(to,xo,k0,b,S):

    def sigmoidal(t,k):
        dt = 1E-1
        n = round(max(t)/dt)
        time = np.linspace(0,max(t),n+1)
        x= np.zeros(n+1); x[0] = xo[0]
        
        def f(x):
            if S == 1:
                dx = k*x*(1-b*x) 
            elif S == 2:
                  dx = k*x**(2/3)*(1 - b*x**(1/3))
            elif S == 3: 
                  dx = k*x**(3/4)*(1 - b*x**(1/4))
            elif S == 4:
                  dx = k*x*(1-b*np.log(x))
            elif S == 5:
                  dx = k*x*np.log(b/x)
            return dx
        for i in range(0,n):
            fx = f(x[i])
            xn = x[i] + fx*dt
            fxn = f(xn)
            x[i+1] = x[i] + (fx + fxn)*dt/2
      
        xi = np.zeros(len(t))

        for i in range(0,len(t)):
            k = abs(time-t[i]) < 1E-9
            xi[i] = x[k]
        return xi
 
    npar = len(k0)
    low = np.ones(npar)*math.inf*(-1)
    sup = np.ones(npar)*math.inf

    Estimate, cov = curve_fit(sigmoidal, to, xo, k0, bounds = (low,sup))
    k = Estimate[0]
    xi = sigmoidal(to,k)
    
    return xi,Estimate,cov

k0= [0.001]
xmax = np.max(xo)
b = [1/xmax, 
     xmax**(-1/3), 
     xmax**(-1/4), 
     1/np.log(xmax), 
     xmax]
#%%
def biostatistcs(Estimate,cov,xo,xa):
    alpha = 0.05
    dof = len(xo)-len(Estimate)
    tval = scipy.stats.t.ppf(q = 1-alpha/2, df = dof)
    SE = np.diag(cov)**(0.5)
    pvalue = 2*scipy.stats.t.sf(np.abs(Estimate/SE), dof)
    MoE = SE*tval
    CI95 = np.zeros([len(Estimate),2])
    for i in range(0,len(Estimate)):
        CI95[i,0] = Estimate[i]-MoE[i]
        CI95[i,1] = Estimate[i]+MoE[i]
    print('\nStatistic results:\n')
    Parameter = ['k']
    df = pd.DataFrame(list(zip(Parameter,Estimate,SE,MoE,CI95,pvalue)),
                      columns = ['Parameters', 'Estimate','SE','MoE','CI95','pValue'])
    print(df.to_string(index = False))
    
def plotResults(to,xo,xa,S):
    fig = plt.figure();
    fig = plt.figure(figsize = (8,4))
    plt.plot(to,xo,linestyle = 'none', marker = 'x',
             color = [0.11,0.30,0.42],label = '$x_o(t)$')
    plt.plot(to,xa,'-', marker = '.',
             color = [0.65,0.27,0.34],label = '$x_a(t)$')
    plt.xlabel('$t$ $[months]$')
    plt.ylabel('$x_i(t)$ $[cells]$')
    plt.xlim(0,72)
    xticks = np.arange(0,73,8)
    plt.xticks(xticks)
    plt.ylim(0,11E8)
    yticks = np.arange(0,12E8,1E8)
    plt.yticks(yticks)
    plt.legend(bbox_to_anchor=(1.01, 1.03), fontsize = 10,
               title= "$Results$", frameon = True)
    plt.show()
    fig.tight_layout()
    if S == 1:
        sigmoidal = 'logistic'
    elif S == 2:
        sigmoidal = 'AllometricSphere'
    elif S == 3: 
        sigmoidal = 'AllometricFractal'
    elif S == 4:
        sigmoidal = 'Gompertz'
    elif S == 5:
        sigmoidal = 'GompertzSimplified'
    namepdf = 'python_Results_' + sigmoidal + '.pdf'
    fig.savefig(namepdf)        

for i in range(0,len(b)):
        S= i+1
        xa,Estimate,cov = mdl(to,xo,k0,b[i],S)
        plotResults(to,xo,xa,S)

