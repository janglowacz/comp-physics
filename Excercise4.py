import sys
import os
import io
import numpy as np
import matplotlib.pyplot as plt
from scipy import special
from numba import jit

fileformat = '.pdf' # figures are saved in this format

def artH(p,phi,settings):
    '''Artificial Hamiltonian for long range Ising model
        
        Input:  conjugated momentum p
                configuration phi
                settings = N, J, h, beta (of Ising model)
                
        Return: Hamiltonian at this state'''

    N, J, h, beta = settings # note: J is, what is denoted as J^{hat} on the exercise sheets
    S = np.square(phi)/(2*beta*J) - N*np.log(2*np.cosh(beta*h+phi))
    return np.square(p)/2 + S


def pdot(phi,settings):
    '''Time derivate of conjugated momentum p
    
        Input:  configuration phi
                settings = N, J, h, beta (of Ising model)
                
        Return: Time derivate of conjugated momentum p at position phi'''

    N, J, h, beta = settings
    return N*np.tanh(beta*h+phi) - phi/(beta*J)


def leapfrog(p,phi,N_md,settings,length=1):
    '''Leapfrog integration algorithm
    
        Input:  initial conjugated momentum p
                initial configuration phi
                number of integration steps N_md
                settings = N, J, h, beta (of Ising model)
                integration length
                
        Return: final conjugated momentum
                final configuration'''

    epsilon = length/N_md # length of each integration step
    
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
    '''1 step of HMC algorithm
    
        Input:  initial configuration phi0
                number of integration steps N_md
                settings = N, J, h, beta (of Ising model)

        Return: new coordinate phi
                boolean: true if accepted, false if rejected'''

    p0 = np.random.normal(0,1) # sample initial conjugate momentum from normal distribution
    p,phi = leapfrog(p0,phi0,N_md,settings) # perform leapfrog integration to optain new coordinates
    if np.random.uniform(0,1) <= np.exp(artH(p0,phi0,settings)-artH(p,phi,settings)): return phi, True # accept/reject step with artificial hamiltonian
    else: return phi0, False


def HMC(phi,N_md,N_therm,N_cfg,settings=(10,0.25,0.5,1)):
    '''HMC algorithm
    
        Input:  initial coordinate phi
                number of integration steps N_md
                number of thermalization steps N_therm
                number of measurements steps N_cfg
                settings = N, J, h, beta (of Ising model) --> default is (10,0.25,0.5,1)
                
        Return: trajectory of configuration phi'''

    N, J, h, beta = settings
    
    # Thermalization
    for i in range(N_therm): phi = step(phi,N_md,settings)[0]

    # Sampling of Probability Distribution
    phi = [(phi,)]
    for i in range(N_cfg): phi.append(step(phi[-1][0],N_md,settings))
    phi = np.array(phi[1:])

    # Optional code to retrieve acceptance:
    # acceptance = phi[:,1].mean()

    phi = phi[:,0]

    return phi


def m(phi,settings):
    '''Average magnetization
    
        Input:  coordinate phi
                settings = N, J, h, beta (of Ising model)
                
        Return: average magnetization at configuration phi'''

    N, J, h, beta = settings
    return np.tanh(beta*h+phi)

# ====================================================================================================

# Settings
N = 5
j = 0.1
h = 0.5
beta = 1

J = j/N
settings = (N,J,h,beta)

N_therm = 2000
N_cfg = 12800
phi0 = 0

# ====================================================================================================

# Create trajectories for N_md=100 and N_md=4
phi_a = HMC(phi0,100,N_therm,N_cfg,settings=settings)
phi_b = HMC(phi0,4,N_therm,N_cfg,settings=settings)

# Calculate average magnetization at each configuration
m_a = m(phi_a,settings)
m_b = m(phi_b,settings)

# plot
x = np.arange(0,250,1)
y_a, y_b = m_a[x], m_b[x]

plt.plot(x,y_a,label=r'HMC$_a$')
plt.plot(x,y_b,label=r'HMC$_b$')
plt.legend(loc='best')
plt.xlabel('Trajectory Index')
plt.ylabel(r'm')
plt.grid()
plt.savefig('tsk_1'+fileformat)
plt.close()

# calculate mean magnetization over whole trajectory
mm_a, mm_b = m_a.mean(), m_b.mean()
print(f"mean m(HMC_a) = {mm_a}")
print(f"mean m(HMC_b) = {mm_b}")

# ====================================================================================================

def autocorr(m,t):
    '''autocorrelation function
    
        Input:  trajectory of observable m
                time t
                
        Return: autocorrelation function of m at time t'''

    t = int(t) # make sure, t is an integer
    mean = m.mean() # calculate mean of observable
    Gamma = 0
    k = 0
    while((k+t)<m.size): # loop over all pairs of points in trajectury, which are t apart from each other
        Gamma += (m[k]-mean)*(m[k+t]-mean)
        k += 1
    #if k == 0: return 0
    return Gamma/k

# ====================================================================================================

# plot normalized autocorrelation function C

t = np.arange(0,50,1)

ac0_a = autocorr(m_a,0)
ac0_b = autocorr(m_b,0)

C_a = np.array([autocorr(m_a,T)/ac0_a for T in t])
C_b = np.array([autocorr(m_b,T)/ac0_b for T in t])

plt.plot(t,C_a,label=r'HMC$_a$')
plt.plot(t,C_b,label=r'HMC$_b$')
plt.legend(loc='best')
plt.grid()
plt.xlabel(r'$\tau$')
plt.ylabel(r'$C(\tau)$')
plt.savefig('tsk_2'+fileformat)
plt.close()

# ====================================================================================================

def blocking(m,b):
    '''blocking algorithm
        (reduces autocorrelation)
        
        Input:  trajectory of observable m
                block size b
                
        Return: new blocked trajectory'''

    m = m[:len(m)-np.remainder(len(m),b)] # throw away remaining data that doesn't fit into blocks
    m = m.reshape(len(m)//b,b) # block
    m = m.mean(axis=1) # thake mean of each block
    return m

    # alternative implementation in 1 line:
    # return m[:len(m)-np.remainder(len(m),b)].reshape(len(m)//b,b).mean(axis=1)

# ====================================================================================================

# create blocked trajectories from HMC_a for several block sizes
B = np.power(2,np.arange(1,7,1)) # block sizes
m_blocked = [blocking(m_a,b) for b in B]

# plot (normalized) autocorrelation function C
t = np.arange(0,25,1)
ac0 = [autocorr(m,0) for m in m_blocked]
C  = [np.array([autocorr(m_blocked[i],T)/ac0[i] for T in t]) for i in range(len(B))]

plt.plot(t,C_a[:len(t)],label=r'HMC$_a$')
for i in range(len(B)):
    plt.plot(t,C[i],label=f'blocksize: {B[i]}')
plt.legend(loc='best')
plt.grid()
plt.xlabel(r'$\tau$')
plt.ylabel(r'$C(\tau)$')
plt.savefig('tsk_3_AC'+fileformat)
plt.close()

# ====================================================================================================

# calculation of the naive standard error
naive = [m_blocked[i].std()/np.sqrt(len(m_blocked[i])) for i in range(len(B))]

# plot naive standard error
plt.plot(B,naive,'o')
plt.grid()
plt.xlabel(r'$b$')
plt.ylabel(r'$\sigma / \sqrt{N / b}$')
plt.savefig('tsk_3_sigma'+fileformat)
plt.close()

# ====================================================================================================

def bootstrap(s,nbs):
    '''bootstrap algorithm for error estimation
    
        Input:  trajectory s
                number of bootstrap elements nbs
                
        Return: boostrap estimate for the error of mean(s)'''
    return s[np.random.randint(s.size,size=(nbs,s.size))].mean(axis=1).std()  

# ====================================================================================================

bootstraped_1 = [bootstrap(m_a,i+1) for i in range(50)] # calculate bootstrap errors for HMC_a (with different numbers of bootstrap elements nbs)
BS = [[bootstrap(m_blocked[j],i+1) for i in range(50)] for j in range(len(B))] # calculate bootstrap errors for blocked trajectories (with different numbers of bootstrap elements nbs)

# plot everything

plt.plot([i+1 for i in range(len(bootstraped_1))],bootstraped_1,label=r'HMC$_a$')
for i in range(len(B)):
    plt.plot([j+1 for j in range(len(BS[i]))],BS[i],label=f'blocksize: {B[i]}')
plt.legend(loc='best')
plt.grid()
plt.xlabel(r'$N_{bs}$')
plt.ylabel(r'$\delta m$')
plt.savefig('tsk_4_bootstrap'+fileformat)
plt.close()

plt.plot(B,naive,'o',label='naive standard error')
plt.plot(B,[bs[20] for bs in BS],'o',label='bootstrap error')
plt.legend(loc='best')
plt.grid()
plt.xlabel(r'$b$')
plt.ylabel('error estimate')
plt.savefig('tsk_4_comparison'+fileformat)
plt.close()