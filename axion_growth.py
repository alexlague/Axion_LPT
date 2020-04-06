import numpy as np
#import astropy.units as u
from scipy.special import jv
from scipy.optimize import curve_fit, fsolve
from params import H0, Omega_M0, Omega_L, kmin, kmax

# Constants

hbar          = 6.582119513599134e-16 # eV s
hbar_over_H0  = 0.00027387891 * (70/H0) # hbar / H0 / (1e-22 eV/c^2) in Mpc^2

# Redshift to scale factor

a_of_z = lambda z: 1. / (1+z)
H_of_a = lambda a: H0*np.sqrt(Omega_M0*a**(-3) + Omega_L) # expansion rate

# Jeans scale

kJeans = lambda a, mass, frac: 66.5 * a**(1./4) * mass**(1./2) * frac**(-1./4)

# Axion growth factor

def D_axion(redshift, k, a_initial, mass, frac):
    '''
    Analytic solution for axion growth factor in matter dominated era
    '''
    
    a_today       = a_of_z(redshift)
    bk            = 2 * hbar_over_H0 * k**2 / mass # dimensionless
    prefac        = (a_initial/a_today)**(1./4)
    order         = 1. / 2 * np.sqrt((frac+24)/frac)
    growth        = prefac * jv(-order, bk/np.sqrt(a_today)) / jv(-order, bk/np.sqrt(a_initial))
    
    return growth

# Smoothed Heaviside model

smoothed_axion_growth = lambda x, x0, alpha: 1 - 1./(1 + np.exp(-2*alpha*(x-x0)))**8

# Initial condition choice

def find_a_initial(redshift, mass, frac):
    '''
    Find a_initial for the Jeans scale to correspond to the first oscillation
    '''
    
    a_today    = a_of_z(redshift)
    #first_zero = 3.95952791650110
    order      = 1. / 2 * np.sqrt((frac+24)/frac)
    f_to_solve = lambda x: jv(-order,x)
    first_zero = fsolve(f_to_solve,3.95)[0]
    a_initial  = (2 * hbar_over_H0 * kJeans(a_today,mass,frac)**2 / mass / first_zero)**2
    
    return a_initial

# Optimal k0, alpha

def find_optimal_parameters(redshift, mass, frac):
    '''
    Find optimal values of alpha and k0 to fit averaged linear growth
    '''
    
    a_today          = a_of_z(redshift)
    k_range          = np.logspace(np.log10(kmin),np.log10(kmax),500)
    a_initial        = find_a_initial(redshift, mass, frac) 
    data_to_fit      = D_axion(redshift, k_range, a_initial, mass, frac)
    data_to_fit      /= D_axion(redshift, k_range, a_initial, mass, frac)[0] # normalization
    selected_indexes = k_range <= 0.75 * kJeans(a_today, mass, frac)
    params           = curve_fit(smoothed_axion_growth, k_range[selected_indexes], 
                                 data_to_fit[selected_indexes], maxfev=5000000)[0]
    
    return params
