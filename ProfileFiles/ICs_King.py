import numpy as np
from numpy import sqrt, pi, log, inf, exp
from scipy.special import erf
from scipy import integrate, optimize

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''
				KING FUNCTIONS
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''

def King_Initialize(params):
#########################################################
##Initialize
#########################################################
    #Calculate dimensionless tidal radius
    Rt = King_Rt(params)
    params['Rt'] = Rt

    return

def King_RootGuess(params):
#########################################################
##Returns initial guess for root finder method
#########################################################
    Rt = params['Rt']
    
    return Rt/10.0

def King_Mass(R,params):
#########################################################
##Returns mass within radius R
#########################################################
    #Make sure R array
    if not isinstance(R, np.ndarray):
        R = np.array([R])
    Rt = params['Rt']
    xmax = max(R) # only have to integrate this far
    xmax = min(xmax,Rt) #make sure inside tidal radius
    x = np.linspace(0.0,xmax,1e4)
    y = King_density(x,params)*x*x #integrand
    Ls = integrate.cumtrapz(y,x,initial = 0) #Mass at x
    L = np.interp(R, x, Ls) #interpolate for mass at R

    #Fix Bounds
    L[R<0.0] = L[-1]*2.0*R[R<0.0]
    L[R>Rt] = L[-1]*(1.0 + R[R>Rt]-Rt)
    return L

def King_MD(R,x,params):
#########################################################
##Cummulative mass distribution for King profile
##Given random number x, root gives radius R
#########################################################
    Rt = params['Rt']
    MF = King_Mass(R,params)/ King_Mass(Rt,params)
    MD = MF - x
    
    return MD

def King_density(R,params):
#########################################################
##Density profile of King model
#########################################################
    P = King_P(R,params)
    p = exp(P)*erf(sqrt(P)) - 2.0*sqrt(P/pi)*(1.0 +2.0*P/3.0)

    p[P<0.0] = 0.0

    return p


def King_Rt(params):
#########################################################
##Potential in an King profile
#########################################################
    P0 = params['P0']
    Rts = np.linspace(0.0,P0*10.0,1e4)
    PRt = King_P(Rts,params)
    while PRt[-1] > 0.0:
        Rts += Rts[-1]/2.0
        PRt = King_P(Rts,params)
    Rt = np.interp(1e-4, PRt[::-1], Rts[::-1])

    #Double check P(Rt)>0.0
    while King_P(Rt,params)<0.0:
        Rt = 0.99*Rt

    return Rt

    
def King_P(R,params):
#########################################################
##Potential in an King profile
#########################################################
    #Make sure R array
    if not isinstance(R, np.ndarray):
        R = np.array([R])
    P0 = params['P0']
    a = 1.0/( exp(P0)*erf(sqrt(P0)) - 2.0*sqrt(P0/pi)*(1.0 +2.0*P0/3.0))
    def f(y,R,a):
        P,Q = y[0], y[1]
        dy1 = Q
        if R==0 or P<1e-4:
            dy2 = 0.0
        else:
            dy2 = -9*a*(exp(P)*erf(sqrt(P)) - 2.0*sqrt(P/pi)*(1.0 +2.0*P/3.0)) - 2*Q/R
        dy = np.array([dy1, dy2])
        return dy
    y0 = [P0,0.0]
    x = np.linspace(0,max(R),1e6)
    sol = integrate.odeint(f, y0, x, args=(a,))
    P = sol[:,0]
    P = np.interp(R, x, P) #interpolate for potential at R
    return P

def King_DF(E,params):
#########################################################
##Distribution function at relative energy E
##For King profile
#########################################################
    F = exp(E) - 1.0

    return F
    
def King_dimens(R,V,n,m,G,params):
#########################################################
##Turns radius and velocity into dimension-full quantities
#########################################################
    r_t = params['r_t']
    Mtot = n*m
    Rt = params['Rt']
    r0 = r_t/Rt
    Ltot = King_Mass(Rt,params)
    rho1 = Mtot/(4.0*pi*r0**3.0 * Ltot)
    rho0 = rho1 * King_density(1e-6,params) #rho1/a
    sigma = sqrt(4.0*pi*G*r0*r0*rho0/9.0)
    
    r = r0*R
    v = sigma*V
    
    return r,v
