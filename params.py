# Cosmological parameters

H0       = 70 # km/s/Mpc
Omega_M0 = 0.3
Omega_L  = 0.7

# Simulations parameters

boxsize  = 256 # Mpc
grid     = 512

# Axion parameters

ax_mass  = 1e-6 # in units of 10^{-22} eV
ax_frac  = 0.1  # between 0 and 1

# Wavenumber parameters

kmin     = 1e-2 # in Mpc^{-1}
kmax     = 1e2  # in Mpc^{-1}

# Optimal growth paramters

find_optimal = True # use the find_optimal_parameters routine (based on fsolve, can be slow)

# Redshift of interest

redshift = 1.

# Input file

input_f  = '256Mpc_n512_nb40_nt1_1_28_10'

# Output file name

output_f = '256Mpc_n512_nb40_nt1_1_28_10_fdm_smoothed'
