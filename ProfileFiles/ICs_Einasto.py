import numpy as np
from numpy import sqrt, pi, log, inf, exp
from scipy import integrate, optimize, special

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''
				Einasto FUNCTIONS
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''
def Einasto_Initialize(params):
#########################################################
##Initialize
#########################################################
    return


def Einasto_RootGuess(params):
#########################################################
##Returns initial guess for root finder method
#########################################################

    return 1.0

def Einasto_MD(R,x,params):
#########################################################
##Cummulative mass distribution for Einasto profile
##Given random number x, root gives radius R
#########################################################
    alpha = params['alpha']

    MF = special.gammainc(3.0/alpha,2.0/alpha * R) #M/Mtot

    MD = log(MF) - log(x)

    return MD

def Einasto_P(R,params):
#########################################################
##Potential in an Einasto profile
#########################################################
    alpha = params['alpha']
    gamma1 = special.gamma(3.0/alpha)*special.gammainc(3.0/alpha,2.0/alpha * R) / special.gamma(2.0/alpha)
    gamma2 = special.gammaincc(2.0/alpha,2.0/alpha * R)
    P = (2.0*R/alpha)**(-1.0/alpha) * gamma1  + gamma2
    return P

def Einasto_DF(E,params):
#########################################################
##Distribution function at relative energy E
##For Einasto profile
#########################################################

    E = max(E,1e-4)

    alpha = params['alpha']

    #Find rmin
    def myfunc(R,E,params):
        return log(E)-log(Einasto_P(R,params))
    Rmin=optimize.newton(myfunc,1.0,args=(E,params))    

    #Integrate
    def integrand(R,E,alpha,params):
        L = special.gammainc(3/alpha,2.0/alpha * R)
        P = Einasto_P(R,params)
        p = exp(-2.0/alpha * R)
        f =  2*p*R**(1/alpha)/L *(2*R + (2.0*R/alpha)**(3.0/alpha) * alpha*p /L/special.gamma(3.0/alpha)- alpha - 1)
        Y = f / sqrt(E-P)
        return Y
        
    F=integrate.quad(integrand,Rmin,inf,args=(E,alpha,params))[0]
    
    return F

    
def Einasto_dimens(R,V,n,m,G,params):
#########################################################
##Turns radius and velocity into dimension-full quantities
#########################################################
    r2 = params['r2']
    alpha = params['alpha']

    p0 = m*n/(4.0*pi*r2*r2*r2/alpha *(alpha/2.0)**(3/alpha)* special.gamma(3/alpha))
    r = R**(1/alpha)*r2
    P0 = 4*pi*G*p0*r2*r2/alpha * (alpha/2.0)**(2/alpha)* special.gamma(2/alpha)
    v = V*sqrt(P0)

    return r,v
