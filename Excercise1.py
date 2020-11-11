import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys


verbose = True

def FLUSH(i,I):
    '''Display percentage of completion'''
    sys.stdout.write("\r"+"{:.2f}% done ({}/{})".format(i/I*100,i,I))
    sys.stdout.flush()


# Ising Hamiltonian
def Hamiltonian(s,J,h):
    '''1-dim Ising Hamiltonian'''
    H_ext = -h * np.sum(s) # part of Hamiltonian coupling to external field
    H_int = -J * np.sum(s*np.roll(s,1)) # part of Hamiltonian that couples spins to one another; you may add factor 2 to include double counting
    return H_int + H_ext


# Boltzmann factor
def bf(s, T, J, h):
    '''The Boltzmann factor'''
    return np.exp(-Hamiltonian(s,J,h)/T)


# Random Sping Configuration Generator
def spin_config(N):
    '''Generate random configuration of N spins'''
    return np.random.choice((-1,+1),int(N))


# Monte Carlo sampler
def MC_s(sample_size, N, T, J, h):
    '''Monte Carlo sampler for the 1-dim Ising model;
       Takes sample size, number of spins N, temperature T and couplings J and h;
       Returns average magnetization per spin;'''
    num, den = 0, 0
    for i in range(int(sample_size)): # loop over samples
        s = spin_config(N) # generate random spin configuration
        BF = bf(s,T,J,h) # calculate Boltzmann weight
        num += s.sum() * BF # magnetization with Boltzmann weight ( = the contribution to the sum in numerator of equation 3)
        den += BF # contribution to partition function Z
    return (num/den)/N # return average magnetization per spin
vMC_s = np.vectorize(MC_s) # vectorize function, so we can use it with numpy arrays


def Exact(N, T, J, h):
    '''Numeric calculator for the exact solution of the 1-dim Ising model;
       Takes number of spins N, temperature T and couplings J and h;
       Returns average magnetization per spin;'''
    num, den = 0, 0
    size = np.power(2,N) # number of possible spin configs
    i = 0
    while(i<size): # loop over all possible spin configs
        s = np.array(list(np.binary_repr(i, width=int(N))), dtype=int)*2-1 # create spin config
        EZ = bf(s,T,J,h) # Boltzmann factor
        num += s.sum() * EZ 
        den += EZ # partition function
        i+=1
    return (num/den)/N # expected magnetization per spin
vExact = np.vectorize(Exact) # vectorize function


def mExactN(N,J,h):
    '''Analytic solution to 1-dim Ising model'''
    SH = np.sinh(h)
    CH = np.cosh(h)
    E4 = np.exp(-4*J)
    X = (CH-np.sqrt(np.square(SH)+E4))/(CH+np.sqrt(np.square(SH)+E4))
    return (1-np.power(X,N))/(1+np.power(X,N)) * SH/np.sqrt(np.square(SH)+E4)


def mExact(J,h):
    '''Analytic solution to 1-dim Ising model in thermodynamic limit (N --> infty)'''
    SH = np.sinh(h)
    CH = np.cosh(h)
    E4 = np.exp(-4*J)
    return SH/np.sqrt(np.square(SH)+E4)


# Settings
J = 1     # coupling to neighbouring spins
H = 0.25  # coupling to exterinal magnetic field
T = 1     # temperature
N = 10    # number of spins (lattice sites)

error_mc = 10 # number of Monte-Carlo simulations used to estimate the error


# print settings
print("="*100)
print("Settings")
print("N:",N,"(varying from",1,"to",N,")")
print("T:",T)
print("J:",J)
print("h:",H,"(varying from",-1,"to",1,")")

# ==============================================================================================================================

# calculation of Monte-Carlo simulations for varying number of spins (n)
print("\n"+"="*100+"\nvarying N")

n = np.linspace(1,N,N) # create array of different n
m = mExactN(n,J,H) # calculate exact solution analitically
d = np.ones(len(n))*mExact(J,H) # calculate thermodynamic limit analytically

XE = []
for i in range(error_mc):
    FLUSH(i,error_mc)
    XE.append(vMC_s(np.power(2,n+1),n,T,J,H)) # for chain of n spins, calculate MC with 2^(n+1) random spin configurations
FLUSH(error_mc,error_mc)
XE = np.array(XE)
X = XE[0,:] # single out one MC for the plot

Xavg = np.average(XE,axis=0) # calculate average result of MC simulations...
Xerr = np.sqrt(np.average(np.square(XE-Xavg[np.newaxis,:]),axis=0)) # ...and its standard deviation  # ERROR FIX WITH NEWAXIS

# plot everything
plt.plot(n,m,label="exact",color="C0")
plt.plot(n,X,label="MC_single",color="C1")
plt.plot(n,Xavg,label="MC_average",color="C2")
plt.fill_between(n,Xavg+Xerr,Xavg-Xerr,alpha=0.2,color="C2")
plt.plot(n,d,label="therm limit",color="C4")

plt.legend(loc="best")
plt.grid()
plt.xlabel("N")
plt.ylabel(r"$\langle m\rangle$")
plt.savefig("varN.pdf")
plt.close()

# ==============================================================================================================================

# calculation of Monte-Carlo simulations for varying (coupling to) external field (h)
print("\n\n"+"="*100+"\nvarying h")

h = np.linspace(-1,1,100) # create array of different h
m = mExactN(N,J,h) # calculate exact solution analitically
d = mExact(J,h) # calculate thermodynamic limit analytically

XE = []
for i in range(error_mc):
    FLUSH(i,error_mc)
    XE.append(vMC_s(np.power(2,N+1),N,T,J,h)) # for chain of n spins, calculate MC with 2^(n+1) random spin configurations
FLUSH(error_mc,error_mc)
XE = np.array(XE)
X = XE[0,:] # single out one MC for the plot

Xavg = np.average(XE,axis=0) # calculate average result of MC simulations...
Xerr = np.sqrt(np.average(np.square(XE-Xavg[np.newaxis,:]),axis=0)) # ...and its standard deviation  # ERROR FIX WITH NEWAXIS

# plot everything
plt.plot(h,m,label="exact",color="C0")
plt.plot(h,X,label="MC_single",color="C1")
plt.plot(h,Xavg,label="MC_average",color="C2")
plt.fill_between(h,Xavg+Xerr,Xavg-Xerr,alpha=0.2,color="C2")
plt.plot(h,d,label="therm limit",color="C4")

plt.legend(loc="best")
plt.grid()
plt.xlabel("h")
plt.ylabel(r"$\langle m\rangle$")
plt.savefig("varh.pdf")
plt.close()

# ==============================================================================================================================

# calculation of Monte-Carlo simulations for varying number of spins (n) AND varying (coupling to) external field (h)
print("\n\n"+"="*100+"\nvarying both")
n = np.linspace(1,N,N) # create array of different n
h = np.linspace(-1,1,100) # create array of different h
a,b = np.meshgrid(n, h)
m = mExactM(a,J,b) # calculate exact solution analitically

# plot exact solution
plt.pcolormesh(n,h,m,cmap="viridis",rasterized=True)

cbar = plt.colorbar()
cbar.set_label(r"$\langle m\rangle$")
plt.xlabel("N")
plt.ylabel("h")
plt.tight_layout()
plt.savefig("2dplot.pdf")
plt.close()


# calculate the Monte-Carlo simulation
n = np.linspace(1,N,N)
h = np.linspace(-1,1,100)
a,b = np.meshgrid(n, h)
m2 = vMC_s(np.power(2,a+1),a,T,J,b)

# plot MC results
plt.pcolormesh(n,h,m2,cmap="viridis",rasterized=True)

cbar = plt.colorbar()
cbar.set_label(r"$\langle m\rangle$")
plt.xlabel("N")
plt.ylabel("h")
plt.tight_layout()
plt.savefig("2dplot_2.pdf")
plt.close()


m3 = np.abs((m-m2)) # calculate difference between exact solution and MC results...

# ...and plot it
plt.pcolormesh(n,h,m3,cmap="viridis",rasterized=True)

cbar = plt.colorbar()
cbar.set_label("<m>")
plt.xlabel("N")
plt.ylabel("h")
plt.tight_layout()
plt.savefig("2dplot_rel.pdf")
plt.close()

# ==============================================================================================================================

# test of the 3 exact solutions
print("\n"+"="*100+"\ntest of the exact solutions")

n = np.linspace(1,N,N) # create array of different n
m = mExactN(n,J,H) # calculate exact solution analytically
c = vExact(n,T,J,H) # calculate exact solution numerically
d = np.ones(len(n))*mExact(J,H) # calculate thermodynamic limit analytically

# plot everything
plt.plot(n,m,label="exact ana",lw=4)
plt.plot(n,c,label="exact num")
plt.plot(n,d,label="therm limit")

plt.legend(loc="best")
plt.grid()
plt.xlabel("N")
plt.ylabel(r"$\langle m\rangle$")
plt.savefig("exTest.pdf")
plt.close()
