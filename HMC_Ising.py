import sys
import os
import io
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import special

# V-0.2

def artH(p,phi,settings):
    N, J, h, beta = settings
    S = np.square(phi)/(2*beta*J) - N*np.log(2*np.cosh(beta*h+phi))
    return np.square(p)/2 + S

def pdot(phi,settings):
    N, J, h, beta = settings
    return N*np.tanh(beta*h+phi) - phi/(beta*J)

def leapfrog(p,phi,N_md,settings,length=1):
    epsilon = length/N_md
    
    # first step
    phi += (epsilon/2) * p

    # more steps
    for i in range(N_md-1):
        p += epsilon * pdot(phi,settings)
        phi += epsilon * p

    # last step
    p += epsilon * pdot(phi,settings)
    phi += (epsilon/2) * p

    return p,phi

def step(phi0,N_md,settings):
    p0 = np.random.uniform(0,1)
    p,phi = leapfrog(p0,phi0,N_md,settings)
    if np.random.uniform(0,1) <= np.exp(artH(p0,phi0,settings)-artH(p,phi,settings)): return phi, True
    else: return phi0, False

def HMC(phi,N_md,N_therm,N_cfg,settings=(10,0.25,0.5,1),nbs=100):
    N, J, h, beta = settings
    
    # Thermalization
    for i in range(N_therm): phi = step(phi,N_md,settings)[0]

    # Sampling of Probability Distribution
    phi = [(phi,)]
    time1 = time.time()
    Text = "J: {:.2f}, N_md: {:.0f}".format(J, N_md)
    for i in range(N_cfg):
        FLUSH((i+1)/N_cfg,time.time()-time1,Text)
        phi.append(step(phi[-1][0],N_md,settings))
    phi = np.array(phi[1:])

    # Acceptance
    acceptance = phi[:,1].mean()
    phi = phi[:,0]

    # Obervables
    m = np.tanh(beta*h+phi)
    e = - (np.square(phi/beta)/(2*N*J) + h*m)

    return m.mean(), e.mean(), acceptance, bootstrap(m,nbs), bootstrap(e,nbs)

def bootstrap(s,nbs):
    return s[np.random.randint(s.size,size=(nbs,s.size))].mean(axis=1).std()

# ====================================================================================================================================================================

# EXACT SOLUTION

def f(x,settings):
    N,J,h,beta = settings
    return np.exp(beta/2 * J * np.square(x) + beta * h *x)

def exact(N,J,h,beta):
    settings = (N,J,h,beta)
    Z = np.sum([special.comb(N,n,exact=True) * f(N - 2*n, settings) for n in range(N+1)])
    m = 1/(N*Z) * np.sum([special.comb(N,n,exact=True) * (N - 2*n) * f(N - 2*n, settings) for n in range(N+1)])
    e = -1/(N*Z) * np.sum([special.comb(N,n,exact=True) * (beta/2 * J *np.square(N - 2*n) + beta*h*(N - 2*n)) * f(N - 2*n, settings) for n in range(N+1)])
    return m, e

# ====================================================================================================================================================================

def TimeFormat(CalcTime):
    X = str("{:.1f}".format(CalcTime%60))+"s"
    if int(CalcTime/(60)%60) > 0: X = str(int(CalcTime/(60)%60)) + "m " + X
    if int(CalcTime/(60*60)%24) > 0: X = str(int(CalcTime/(60*60)%24)) + "h " + X
    if int(CalcTime/(60*60*24)) > 0: X = str(int(CalcTime/(60*60*24))) + "d " + X
    return X

def FLUSH(F,T,Text,Finish=False):
    ''' Terminal output flush method;
        F is the completion percentage;
        T is the expired time;
        Text is an additional output text'''
    Y = "{:.2%}%".format(F)
    OUTX = "\r"+Text+" ["+"="*round(F*40)+"-"*round((1-F)*40)+"] "+(8-len(Y))*" "+Y+" done after "+TimeFormat(T)+" , remaining time "+TimeFormat((1-F)*(T)/(F))
    if Finish: OUTX = "\r"+Text+" done after "+TimeFormat(T) # cleanup
    sys.stdout.write(OUTX + " "*(120-len(OUTX)))
    if Finish: sys.stdout.write("\n") # cleanup
    sys.stdout.flush()

# ====================================================================================================================================================================

phi = 10
N_md = 1
N_therm = 2000
N_cfg = 5000

N = 5
J_vals = np.linspace(0.2,2,91) #91
h = 0.5
beta = 1

#y = [HMC(phi,N_md,N_therm,N_cfg,settings=(N,J,h,beta)) for J in J_vals]
#y = np.array(y)
#m, e, a = y[:,0], y[:,1], y[:,2]

y = []
for J in J_vals:
    N_md = 1
    y.append(HMC(phi,N_md,N_therm,N_cfg,settings=(N,J,h,beta)))
    time0 = time.time()
    while y[-1][2]<0.5:
        N_md += 1
        y[-1] = HMC(phi,N_md,N_therm,N_cfg,settings=(N,J,h,beta))
    FLUSH(1,time.time()-time0,"J: {:.2f}, N_md: {:.0f}, Acceptance: {:.2f},".format(J,N_md,y[-1][2]),Finish=True)
y = np.array(y)
m, e, a, m_err, e_err = y[:,0], y[:,1], y[:,2], y[:,3], y[:,4]

y_ex = np.array([exact(N,J,h,beta) for J in J_vals])
m_ex, e_ex = y_ex[:,0], y_ex[:,1]

plt.figure(figsize=(9,6))
plt.errorbar(J_vals,m,yerr=m_err,fmt='x',capsize=3)
plt.plot(J_vals,m_ex,'.')
plt.xlabel('J')
plt.ylabel(r'$\langle m\rangle$')
plt.grid()
plt.savefig(f'varJ_m_{N}.pdf')
plt.close()

plt.figure(figsize=(9,6))
plt.errorbar(J_vals,e,yerr=e_err,fmt='x',capsize=3)
plt.plot(J_vals,e_ex,'.')
plt.xlabel('J')
plt.ylabel(r'$\langle e\rangle$')
plt.grid()
plt.savefig(f'varJ_e_{N}.pdf')
plt.close()

plt.figure(figsize=(9,6))
plt.plot(J_vals,a,'x')
plt.xlabel('J')
plt.ylabel('Acceptance')
plt.grid()
plt.savefig(f'varJ_a_{N}.pdf')
plt.close()