import numpy as np
from params import ax_mass, ax_frac, redshift, find_optimal, input_f, output_f
from axion_growth import find_optimal_parameters, smoothed_axion_growth


# Number of particles
N = len(np.fromfile('displacements_fields/fields/etax1_' + input_f, dtype=np.float32))**(1./3)
N = int(round(N))

# Import displacement fields
disp_cdm_x = np.fromfile('displacements_fields/fields/etax1_' + input_f, dtype=np.float32,count=N**3)
disp_cdm_x = np.reshape(disp_cdm_x,(N,N,N))
disp_cdm_y = np.fromfile('displacements_fields/fields/etay1_' + input_f, dtype=np.float32,count=N**3)
disp_cdm_y = np.reshape(disp_cdm_y,(N,N,N))
disp_cdm_z = np.fromfile('displacements_fields/fields/etaz1_' + input_f, dtype=np.float32,count=N**3)
disp_cdm_z = np.reshape(disp_cdm_z,(N,N,N))

# FFT
fft_disp_x = np.fft.fftn(disp_cdm_x)
fft_disp_y = np.fft.fftn(disp_cdm_y)
fft_disp_z = np.fft.fftn(disp_cdm_z)

# Small-scale filtering
L  = 256. # Mpc
dx = L/N

freq_x, freq_y, freq_z = np.meshgrid(np.fft.fftfreq(N),np.fft.fftfreq(N),np.fft.fftfreq(N))
freq_x = freq_x / dx # need to divide to get physical wavenumbers 
freq_y = freq_y / dx
freq_z = freq_z / dx

if find_optimal:
    best_params = find_optimal_parameters(redshift, ax_mass, ax_frac)
else:
    # use scaling relations (much faster)
    k0          = 44.3 * 1./(1+redshift)**(.25) * ax_mass**.5 *ax_frac**(-.25)
    alpha       = 0.0679 * 1./(1+redshift)**(-.25) * ax_mass**(-.5) * ax_frac**.25
    best_params = [k0, alpha]

print(best_params[0],best_params[1])

fft_disp_x *= smoothed_axion_growth(abs(freq_x), *best_params)
fft_disp_y *= smoothed_axion_growth(abs(freq_y), *best_params)
fft_disp_z *= smoothed_axion_growth(abs(freq_z), *best_params)

# iFFT
new_disp_x = np.fft.ifftn(fft_disp_x)
new_disp_y = np.fft.ifftn(fft_disp_y)
new_disp_z = np.fft.ifftn(fft_disp_z)

# Save FDM displacements
redshift_string = '_z' + str(redshift)

np.savez('displacements_fields/fields/etax1_' + output_f + redshift_string, new_disp_x)
np.savez('displacements_fields/fields/etay1_' + output_f + redshift_string, new_disp_y)
np.savez('displacements_fields/fields/etaz1_' + output_f + redshift_string, new_disp_z)


