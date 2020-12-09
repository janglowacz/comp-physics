import sys
import os
import io
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#import numba
#@numba.jit(nopython=True)

saveformat = '.pdf'

verbose = True

def FLUSH(i,I):
    '''Display percentage of completion'''
    sys.stdout.write("\r"+"{:.2f}% done ({}/{})".format(i/I*100,i,I))
    sys.stdout.flush()


def DeltaS(u,x,r,delta,a,beta):
    nux = u[x]+r*delta
    return (2*beta/a) * (np.square(nux)-np.square(u[x]) + (u[x-1]+u[x+1])*(u[x]-nux))


def sweep(u,delta,a,beta):
    N = u.size-1
    for i in range(N-1):
        x = np.random.randint(1,N) # choose random site to change
        r = np.random.uniform(-1,1)
        if np.random.uniform(0,1) <= np.exp(-DeltaS(u,x,r,delta,a,beta)) : u[x]+=r*delta # accept / reject
    return u


def metropolis_hastings(N,delta,a,beta):
    
    N = int(N) # make sure we have an integer number of lattice sites

    nTherm = 500 # number of thermalization steps
    nMeas = 5000 # number of mesurements

    u = np.random.uniform(-1,1,N+1) # initial configuration (BETTER STARTING POINT?)

    # Dirichlet boundary conditions
    u[0] = 0
    u[N] = 0

    # thermalization
    for i in range(nTherm):
        u = sweep(u,delta,a,beta)

    traj = [u] # trajectory

    # measurements
    for j in range(nMeas):
        traj.append(sweep(traj[-1].copy(),delta,a,beta)) # take measurement
    
    return traj


def calc_m(u):
    return u[1:-1].sum() / (u.size-1)


def calc_H(u,a):
    return np.square(u[1:]-u[:-1]).sum() / a


def calc_m_squared(u):
    return np.square(calc_m(u))


print('Metropolis Hastings')

delta = 2
N = 64
beta = 1
a = 1

traj = metropolis_hastings(N,delta,a,beta)
m = np.array([calc_m(u) for u in traj])


x = np.arange(0,len(m),1)
plt.plot(x,m,',',label='measured values')
plt.hlines(m.mean(),0,len(m),colors='C1',label='mean')
plt.grid()
plt.xlabel('Measurement step')
plt.ylabel(r'$\langle m \rangle$')
plt.legend(loc='best')
plt.savefig('MH-m'+saveformat)
plt.close()


H = np.array([calc_H(u,a) for u in traj])


plt.plot(np.arange(0,len(H),1),H,',',label='measured values')
plt.hlines(H.mean(),0,len(H),colors='C1',label='mean')
plt.grid()
plt.xlabel('Measurement step')
plt.ylabel(r'$\langle E \rangle$')
plt.legend(loc='best')
plt.savefig('MH-E'+saveformat)
plt.close()


print(f'mean m = {m.mean()}')
print(f'mean E = {H.mean()}\n')


m_squared_MH = np.array([calc_m_squared(np.array(u)) for u in traj])


def fine_to_coarse(u):
    return u[::2]


def coarse_to_fine(u):
    nu = np.ones(2*u.size-1)
    for i in range(nu.size):
        if i%2==1:  nu[i] = (u[(i-1)//2] + u[(i+1)//2]) / 2
        else: nu[i] = u[i//2]
    return nu


def coarse_phi(phi):
    nphi = np.zeros(phi.size//2 + 1)
    for i in np.arange(1,nphi.size-1,1):
        nphi[i] = 1 / 2 * ( phi[2*i] + 1 / 2 * (phi[2*i+1] + phi[2*i-1]))
    return nphi


def DeltaS_with_phi(u,x,r,delta,a,beta,phi):
    nux = u[x]+r*delta
    return (2*beta/a) * (np.square(nux)-np.square(u[x]) + (u[x-1]+u[x+1])*(u[x]-nux)) + beta*phi[x]*(nux-u[x])


def sweep_with_phi(u,delta,a,beta,phi):
    N = u.size-1
    for i in range(N-1):
        x = np.random.randint(1,N) # choose random site to change
        r = np.random.uniform(-1,1)
        if np.random.uniform(0,1) <= np.exp(-DeltaS_with_phi(u,x,r,delta,a,beta,phi)) : u[x]+=r*delta # accept / reject
    return u


def multigrid_recursion(u,delta,a,beta,gamma,n_level,level,nu_pre,nu_post,phi):
    for i in range(nu_pre[level]): u = sweep_with_phi(u,delta,a,beta,phi) # do nu_pre pre-coarsening sweeps at the current level
    if level < n_level - 1:
        nphi = coarse_phi(phi) # generate next coarser level for grid_spacing
        nu = np.zeros(u.size//2+1) # generate the starting coarser nu
        for y in range(gamma[level]):
            nu = multigrid_recursion(nu,delta,a/2,beta,gamma,n_level,level+1,nu_pre,nu_post,nphi) # do gamma multigrid cycles for the coarser level
        u += coarse_to_fine(nu) # update the current u
    for i in range(nu_post[level]): u = sweep_with_phi(u,delta,a,beta,phi) # do nu_post post-prolongation sweeps at the current level
    return u


def multigrid(N,delta,a,beta,gamma,n_level,nu_pre,nu_post):
    
    N = int(N) # make sure we have an integer number of lattice sites

    nTherm = 500 # number of thermalization steps
    nMeas = 5000 # number of mesurements

    u = np.random.uniform(-1,1,N+1) # initial configuration (BETTER STARTING POINT?)

    # Dirichlet boundary conditions
    u[0] = 0
    u[N] = 0

    phi = np.zeros(N+1)

    # thermalization
    for i in range(nTherm):
        u = multigrid_recursion(u,delta,a,beta,gamma,n_level,0,nu_pre,nu_post,phi)

    traj = [u] # trajectory

    # measurements
    for j in range(nMeas):
        traj.append(multigrid_recursion(traj[-1].copy(),delta,a,beta,gamma,n_level,0,nu_pre,nu_post,phi)) # take measurement
    
    return traj


print('Multigrid')
    
delta = 2
N = 64
gamma = [1,2]
n_level = 3
nu_pre, nu_post = [4,2,1] , [4,2,1]


traj = multigrid(N,delta,1,1,gamma,n_level,nu_pre,nu_post)
m = np.array([calc_m(u) for u in traj])


x = np.arange(0,len(m),1)
plt.plot(x,m,',',label='measured values')
plt.hlines(m.mean(),0,len(m),colors='C1',label='mean')
plt.grid()
plt.xlabel('Measurement step')
plt.ylabel(r'$\langle m \rangle$')
plt.legend(loc='best')
plt.savefig('MG-m'+saveformat)
plt.close()


H = np.array([calc_H(u,a) for u in traj])


plt.plot(np.arange(0,len(H),1),H,',',label='measured values')
plt.hlines(H.mean(),0,len(H),colors='C1',label='mean')
plt.grid()
plt.xlabel('Measurement step')
plt.ylabel(r'$\langle E \rangle$')
plt.legend(loc='best')
plt.savefig('MG-E'+saveformat)
plt.close()


print(f'mean m = {m.mean()}')
print(f'mean E = {H.mean()}')


m_squared_MG = np.array([calc_m_squared(np.array(u)) for u in traj])


def autocorr(X,T):
    T = int(T)
    mean = X.mean()
    Gamma = 0
    k = 0
    while((k+T)<X.size):
        Gamma += (X[k]-mean)*(X[k+T]-mean)
        k += 1
    #if k == 0: return 0
    return Gamma/k


tau_m = np.min([len(m_squared_MH),len(m_squared_MG)])

ref = autocorr(m_squared_MH,0)
a_MH = np.array([autocorr(m_squared_MH,i)/ref for i in range(tau_m)])

ref = autocorr(m_squared_MG,0)
a_MG = np.array([autocorr(m_squared_MG,i)/ref for i in range(tau_m)])


plt.plot(np.arange(0,tau_m,1),a_MH,label='Metropolis-Hastings')
plt.plot(np.arange(0,tau_m,1),a_MG,label='Multigrid')
plt.grid()
plt.xlabel(r'$\tau$')
plt.ylabel(r'$C(\tau)(m^2)$')
plt.legend(loc='best')
plt.savefig('Autocorr'+saveformat)
plt.close()