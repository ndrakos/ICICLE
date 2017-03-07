import numpy as np
from numpy import sqrt, pi, log, inf
from scipy import integrate, optimize
from collections import Counter

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''
				NFW FUNCTIONS
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''

def NFW_Initialize(params):
#########################################################
##Initialize
#########################################################
    return

def NFW_RootGuess(params):
#########################################################
##Returns initial guess for root finder method
#########################################################
    r_s = params['r_s']
    r_cut = params['r_cut']
    c = r_cut/r_s
    
    return c/10.0

def NFW_MD(R,x,params):
#########################################################
##Cummulative mass distribution for NFW profile
##Given random number x, root gives radius R
#########################################################
    r_s = params['r_s']
    r_cut = params['r_cut']
    c = r_cut/r_s

    MD = log(R+1)-R/(R+1)-x*(log(c+1)-c/(c+1))

    return MD

def NFW_P(R,params):
#########################################################
##Potential in an NFW profile 
#########################################################
    P = log(1+R)/R
    return P

def NFW_DF(E,params):
#########################################################
##Distribution function at relative energy E
##For NFW profile
#########################################################

    E = max(E,1e-6)

    #Find rmin
    def myfunc(R,E):
        return E-log(1+R)/R
    Rmin=optimize.newton(myfunc,1/E,args=(E,))
    Rmin=Rmin*(1+1e-4)
    
    #Integrate
    def integrand(R,E):
        L = log(1+R)-R/(1+R)
        P = log(R+1)/R
        p = 1.0/(R*(1+R)**2)
        f =  (p*R/(L*(1+R)))**2 * (6*L/p + R*(1+R)*(3*R+1))
        Y = f / sqrt(E-P)
        return Y

    F=integrate.quad(integrand,Rmin,inf,args=(E))[0]

    return F


def NFW_dimens(R,V,n,m,G,params):
#########################################################
##Turns radius and velocity into dimension-full quantities
#########################################################

    #Remove Unbound Particles
    trun = params['truncate']
    if trun:
        R,V = NFW_truncate(R,V,n,params)
    
    #Redim
    r_s = params['r_s']
    r_cut = params['r_cut']
    c = r_cut/r_s
    p0 = n*m/(4.0*pi*r_s**3*(log(1.0 + c)-c/(1.0 +c)))
    r = R*r_s
    v = V*sqrt(4*pi*G*p0*r_s**2)

    return r,v

def NFW_truncate(R,V,n,params):
#########################################################
##Removes unbound particles
#########################################################
    r_s = params['r_s']
    r_cut = params['r_cut']
    c = r_cut/r_s
    
    n_new = 1.0*n
    n_old = 2.0*n_new
    A = (log(1+c)-(c/(1+c)))/(1.0*n) #dimensionless mass per particle
    
    while (n_old>n_new):
        #Inside Potential
        R_sort, R_ind = np.unique(R, return_inverse=True) #sorted list without repeats, index
        par_int = (np.cumsum(np.concatenate(([0], np.bincount(R_ind)))))[R_ind] #number of interior particles
        P = par_int/R
        #Outside Potential
        count = Counter(R) # find repeated values
        vals = np.array(count.values())
        keys = np.array(count.keys())
        inds = keys.argsort()
        reps = vals[inds]
        R_out = reps[::-1] * 1/R_sort[::-1]
        R_out = np.cumsum(R_out) - R_out #sum 1/R for all exterior particles
        R_out = R_out[::-1] #flip direction
        P+= R_out[R_ind]
        #Calculate E
        E = A*P - V*V/2.0
        #Remove if unbound
        E_min = A*n_new/c #potential at R=c
        R = R[E>E_min]
        V = V[E>E_min]
        #Store newn
        n_old = 1.0*n_new
        n_new = R.shape[0]

    return R, V


