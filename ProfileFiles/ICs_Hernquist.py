from numpy import pi,sqrt
from math import asin

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''
				HERNQUIST FUNCTIONS
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''

def Hernquist_Initialize(params):
#########################################################
##Initialize
#########################################################
    return

def Hernquist_RootGuess(params):
#########################################################
##Returns initial guess for root finder method
#########################################################
    return 1.0

def Hernquist_MD(R,x,params):
#########################################################
##Cummulative mass distribution for Hernquist profile
##Given random number x, root gives radius R
#########################################################
    MD = R - sqrt(x)*(R+1)

    return MD

def Hernquist_P(R,params):
#########################################################
##Potential in an Hernquist profile 
#########################################################
	P = 1/(R+1)
	return P

def Hernquist_DF(E,params):
#########################################################
##Distribution function at relative energy E
##For Hernquist profile
#########################################################

    F = 3*asin(sqrt(E)) + sqrt(E*(1-E))*(1-2*E)*(8*E*E-8*E-3)
    F = F/((1-E)**(2.5))

    return F
	
def Hernquist_dimens(R,V,n,m,G,params):
#########################################################
##Turns radius and velocity into dimension-full quantities
#########################################################
    a = params['a'] #scaleradius
    M = m*n #total mass
    r = R*a
    v = V*sqrt(G*M/a)

    return r,v
