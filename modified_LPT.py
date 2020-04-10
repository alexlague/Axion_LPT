import numpy as np
from params import ax_mass, ax_frac, redshift, input_f, output_f, n_lpt
from axion_growth import smoothed_axion_growth

# Working directory
path = '/project/r/rbond/alague/axion_runs/modified_LPT/'

# Number of particles
N = len(np.fromfile(path + 'displacements_fields/fields/etax1_' + input_f, dtype=np.float32))**(1./3)
N = int(round(N))

# Need to repeat procedure twice if doing 2LPT
for i in range(n_lpt):
    
    # Set order of current calculation (first 1LPT, then 2LPT)
    n_lpt_current = i + 1
    
    # Import displacement fields
    disp_cdm_x = np.fromfile(path + 'displacements_fields/fields/etax' + str(n_lpt_current)  + '_' + input_f, dtype=np.float32,count=N**3)
    disp_cdm_x = np.reshape(disp_cdm_x,(N,N,N))
    disp_cdm_y = np.fromfile(path + 'displacements_fields/fields/etay' + str(n_lpt_current)  + '_' + input_f, dtype=np.float32,count=N**3)
    disp_cdm_y = np.reshape(disp_cdm_y,(N,N,N))
    disp_cdm_z = np.fromfile(path + 'displacements_fields/fields/etaz' + str(n_lpt_current)  + '_' + input_f, dtype=np.float32,count=N**3)
    disp_cdm_z = np.reshape(disp_cdm_z,(N,N,N))
    
    # FFT
    fft_disp_x = np.fft.fftn(disp_cdm_x)
    fft_disp_y = np.fft.fftn(disp_cdm_y)
    fft_disp_z = np.fft.fftn(disp_cdm_z)
    
    # Small-scale filtering
    L  = 256. # Mpc
    ds = L/N / np.pi # to adjust for Nyquist frequency
    
    freq_x, freq_y, freq_z = np.meshgrid(np.fft.fftfreq(N),np.fft.fftfreq(N),np.fft.fftfreq(N))
    freq_x = freq_x / ds # need to divide to get physical wavenumbers 
    freq_y = freq_y / ds
    freq_z = freq_z / ds
    
    norm_freq    = np.sqrt(abs(freq_x)**2 + abs(freq_y)**2 + abs(freq_z)**2)
    scaling_freq = smoothed_axion_growth(norm_freq, 1./(1+redshift), ax_mass, ax_frac)
    
    fft_disp_x *= scaling_freq
    fft_disp_y *= scaling_freq
    fft_disp_z *= scaling_freq
        
    #fft_disp_x *= smoothed_axion_growth(abs(freq_x), *best_params)
    #fft_disp_y *= smoothed_axion_growth(abs(freq_y), *best_params)
    #fft_disp_z *= smoothed_axion_growth(abs(freq_z), *best_params)
    #print(abs(freq_x[::50]))
    #print(fft_disp_x[::50])
    
    #import matplotlib.pyplot as plt
    #plt.semilogx(norm_freq.ravel()[::15000],smoothed_axion_growth(norm_freq, *best_params).ravel()[::15000],'.')
    #plt.show()
        
    
    # iFFT
    new_disp_x = np.fft.ifftn(fft_disp_x)
    new_disp_y = np.fft.ifftn(fft_disp_y)
    new_disp_z = np.fft.ifftn(fft_disp_z)
        
    # Save FDM displacements
    redshift_string = '_z' + str(redshift)
        
    np.savez(path + 'displacements_fields/fields/etax' + str(n_lpt_current) + '_' + output_f + redshift_string, new_disp_x)
    np.savez(path + 'displacements_fields/fields/etay' + str(n_lpt_current) + '_' + output_f + redshift_string, new_disp_y)
    np.savez(path + 'displacements_fields/fields/etaz' + str(n_lpt_current) + '_' + output_f + redshift_string, new_disp_z)
        
