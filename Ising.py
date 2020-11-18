import sys
import os
import io
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#import numba
#@numba.jit(nopython=True)


verbose = True

def Exact_1d(N,J,h): # for T=1
    '''Analytic solution to 1-dim Ising model'''
    SH = np.sinh(h)
    CH = np.cosh(h)
    E4 = np.exp(-4*J)
    X = (CH-np.sqrt(np.square(SH)+E4))/(CH+np.sqrt(np.square(SH)+E4))
    return (1-np.power(X,N))/(1+np.power(X,N)) * SH/np.sqrt(np.square(SH)+E4)


def Hamiltonian(s,J,H):
    ''' implementation of the 2d ising Hamiltonian '''
    nx, ny = s.shape
    Energy = 0
    for x in range(nx):
        for y in range(ny):
            t = x, y
            Energy += -1 * s[t]* ( J / 2 * ( s[(x+1)%nx,y] + s[(x-1),y] + s[x,(y+1)%ny] + s[x,(y-1)] ) + H )
    return Energy

def Exact_TL_1d(J,h): #  for T=1
    '''Analytic solution to 1-dim Ising model in thermodynamic limit (N --> infty)'''
    SH = np.sinh(h)
    E4 = np.exp(-4*J)
    return SH/np.sqrt(np.square(SH)+E4)


def spin_config_1d(N):
    return np.random.choice((-1,+1),int(N))


def DeltaS_1d(s,x,J,H): # T = 1
    nx = s.size
    return 2*s[x]*(J*(s[(x+1)%nx]+s[(x-1)])+H) # change in energy from spinflip


def metropolis_hastings_1d(N,J,H):

    N = int(N) # make sure we have an integer number of lattice sites

    nTherm = 500 # number of thermalization steps
    nMeas = 1000 # number of measurements

    m = np.array([]) # holds magnetization measurements
    p = 0 # number of accepts

    s = np.array([(-1)**x for x in range(N)]) # initial spin configuration

    # thermalization
    for i in range(nTherm):
        x = np.random.randint(N) # choose random spin to flip
        if np.random.uniform(0,1) <= np.exp(-DeltaS_1d(s,x,J,H)) : s[x]*=-1 # accept / reject
    
    # measurements
    for j in range(nMeas):
        for x in range(N): # sweep through the lattice
            if np.random.uniform(0,1) <= np.exp(-DeltaS_1d(s,x,J,H)) : # accept / reject
                s[x]*=-1
                p += 1
        m=np.append(m,s.mean()) # take measurement
    
    return m.mean(), p/(nMeas*N) # mean magnetization (per spin) & probability of acceptance

v_metropolis_hastings_1d = np.vectorize(metropolis_hastings_1d)


def DeltaS_2d(s,t,J,H): # T = 1
    '''Change of action after spin flip at site t;
       s is a 2-dim array of spins;
       t is a tuple (or tuple-like structure) of coordinates'''
    x, y = t
    nx, ny = s.shape
    return 2*s[t]* ( J * ( s[(x+1)%nx,y] + s[(x-1),y] + s[x,(y+1)%ny] + s[x,(y-1)] ) + H )


def metropolis_hastings_2d(nx,ny,J,H):
    time0 = time.time() # Timestamp for runtime measurements

    ID_Text = "Nx={:.0f} Ny={:.0f} J={:.2f} H={:.2f} ".format(nx,ny,J,H) # Text formatting for Terminal Output
    ID_Text += " "*(36-len(ID_Text))
    
    nx, ny = int(nx), int(ny) # make sure we have an integer number of lattice sites

    nTherm = 1000 # number of thermalization steps
    nMeas = 2000 # number of measurements

    m = np.array([]) # holds magnetization measurements
    p = 0 # number of accepts

    # initial spin configuration
    #x_ini = np.array([(-1)**x for x in range(nx)])
    #y_ini = np.array([(-1)**y for y in range(ny)])
    #s = np.column_stack((x_ini,y_ini))

    # inital spin configuration (alternative)
    s = np.random.choice((-1,1), size=(nx,ny))

    time1 = time.time() # Timestamp for runtime measurements
    # thermalization
    for i in range(nTherm):
        FLUSH((i+1)/nTherm,time.time()-time1,ID_Text+"Thermalization") # Terminal output flush
        t = np.random.randint(nx), np.random.randint(ny) # choose random spin to flip
        if np.random.uniform(0,1) <= np.exp(-DeltaS_2d(s,t,J,H)) : s[t]*=-1 # accept / reject

    Energies = np.array([Hamiltonian(s,J,H)]) # start of the energy calculation

    time1 = time.time() # Timestamp for runtime measurements
    # measurements
    for j in range(nMeas):
        New_Energy = Energies[-1] 
        FLUSH((j+1)/nMeas,time.time()-time1,ID_Text+"Measurements  ") # Terminal output flush
        for x in range(nx): # sweep through the lattice
            for y in range(ny):
                Delta = DeltaS_2d(s,(x,y),J,H)
                New_Energy += Delta
                if np.random.uniform(0,1) <= np.exp(-Delta) : # accept / reject
                    s[x,y]*=-1
                    p += 1
        m = np.append(m,s.mean()) # take <m> measurement
        Energies = np.append(Energies,New_Energy) # take <e> measurement
    
    FLUSH(1,time.time()-time0,ID_Text, Finish=True) # Terminal output flush

    return m.mean(), p/(nMeas*nx*ny), Energies.mean() # mean magnetization (per spin) & probability of acceptance

v_metropolis_hastings_2d = np.vectorize(metropolis_hastings_2d)


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
    OUTX = "\r"+Text+" ["+"="*int(F*50)+"-"*int((1-F)*50)+"] "+"{:.2f}%".format(F*100)+" done after "+TimeFormat(T)+" , remaining time "+TimeFormat((1-F)*(T)/(F))
    if Finish: OUTX = "\r"+Text+" done after "+TimeFormat(T) # cleanup
    sys.stdout.write(OUTX + " "*(160-len(OUTX)))
    if Finish: sys.stdout.write("\n") # cleanup
    sys.stdout.flush()

def Exactthing(J):
    J_C = 0.440686793509772 # ..
    if J > J_C: return np.power(1-1/np.power(np.sinh(2*J),4),1/8)
    else: return 0

v_Exactthing = np.vectorize(Exactthing)

# calculation of 2d Ising Monte-Carlo simulations for varying number of spins (n)
J = 0.25
H = 0.25
N = 12

j = np.linspace(0.25,2,8)
h = np.linspace(-1,1,9)
n = np.linspace(4,12,9)

def var_H():
    print("\n\n"+"="*160+"\n<m> under variation of h @ N=12 & J=0.25")

    h = np.linspace(-1,1,100)  # Set the amount of H points here
    
    m = v_metropolis_hastings_2d(N,N,J,h)

    plt.plot(h,m[0],".",label="MC")

    plt.legend(loc="best")
    plt.grid()
    plt.xlabel("h")
    plt.ylabel(r"$\langle m\rangle$")
    plt.tight_layout()
    plt.savefig("m_var_H.pdf")
    plt.show()
    plt.close()   

def var_J():
    print("\n\n"+"="*160+"\n<m> under variation of J @ N=12 & h=0")

    j = np.linspace(0.1,1,100) # Set the amount of J points here 
    
    m = v_metropolis_hastings_2d(12,12,j,0)

    plt.plot(j,m[0],".",label="MC")

    plt.legend(loc="best")
    plt.grid()
    plt.xlabel("J")
    plt.ylabel(r"$\langle m\rangle$")
    plt.tight_layout()
    plt.savefig("m_var_J.pdf")
    plt.show()
    plt.close() 

def var_J_e():
    print("\n\n"+"="*160+"\n<e> under variation of J @ N=12 & h=0")

    j = np.linspace(0.1,2,100) # Set the amount of J points here
    
    e = v_metropolis_hastings_2d(12,12,j,0)

    plt.plot(j,e[2],".",label="MC")

    plt.legend(loc="best")
    plt.grid()
    plt.xlabel("J")
    plt.ylabel(r"$\langle \epsilon \rangle$")
    plt.tight_layout()
    plt.savefig("e_var_J.pdf")
    plt.show()
    plt.close() 

def var_J_m2():
    print("\n\n"+"="*160+"\n|<m>| under variation of J @ N=12 & h=0")

    j = np.linspace(0.1,1,20) # Set the amount of J points here 
    
    m = v_metropolis_hastings_2d(12,12,j,0)

    plt.plot(np.divide(1,j),np.abs(m[0]),".",label="MC")
    plt.plot(np.divide(1,j),v_Exactthing(np.divide(1,j)),"",label="exact")

    plt.legend(loc="best")
    plt.grid()
    plt.xlabel(r"$J^{-1}$")
    plt.ylabel(r"$|\langle m\rangle|$")
    plt.tight_layout()
    plt.savefig("m_var_J-1.pdf")
    plt.show()
    plt.close() 

var_H()
var_J()
var_J_e()
var_J_m2()