#===========================================
# GENERATING PARTICLE CATALOGS
# FROM MODIFIED LPT FOR
# MIXED FUZZY DARK MATTER SIMS
#===========================================

import numpy as np
from params import redshift, input_f, output_f, n_lpt

path = '/project/r/rbond/alague/axion_runs/modified_LPT/displacements_fields/fields/'

# Number of particles           

N = len(np.fromfile(path + 'etax1_' + input_f, dtype=np.float32))**(1./3)
N = int(round(N))

# CDM particles displacements with mixed FDM IC

disp_cdm_fdm_x = np.fromfile(path + 'etax1_' + input_f, dtype=np.float32,count=N**3)
disp_cdm_fdm_y = np.fromfile(path + 'etay1_' + input_f, dtype=np.float32,count=N**3)
disp_cdm_fdm_z = np.fromfile(path + 'etaz1_' + input_f, dtype=np.float32,count=N**3)

# Redshift strings

redshift_dict = {'1':'_z1', '2':'_z2', '4':'_z4', '80':'_z80'}
redshift_str  = redshift_dict[str(int(redshift))] 

# FDM particles displacements with mixed FDM IC

disp_fdm_fdm_x = np.load(path + 'etax1_' + input_f + '_fdm_smoothed' + redshift_str + '.0' + '.npz')['arr_0'].real
disp_fdm_fdm_x = disp_fdm_fdm_x.reshape(N**3)
disp_fdm_fdm_y = np.load(path + 'etay1_' + input_f + '_fdm_smoothed' + redshift_str + '.0' + '.npz')['arr_0'].real
disp_fdm_fdm_y = disp_fdm_fdm_y.reshape(N**3)
disp_fdm_fdm_z = np.load(path + 'etaz1_' + input_f + '_fdm_smoothed' + redshift_str + '.0' + '.npz')['arr_0'].real
disp_fdm_fdm_z = disp_fdm_fdm_z.reshape(N**3)

# Import second order displacements if present

if n_lpt == 2:
    disp_cdm_fdm_x_2lpt = np.fromfile(path + 'etax2_' + input_f, dtype=np.float32,count=N**3)
    disp_cdm_fdm_y_2lpt = np.fromfile(path + 'etay2_' + input_f, dtype=np.float32,count=N**3)
    disp_cdm_fdm_z_2lpt = np.fromfile(path + 'etaz2_' + input_f, dtype=np.float32,count=N**3)

    disp_fdm_fdm_x_2lpt = np.load(path + 'etax2_' + input_f + '_fdm_smoothed' + redshift_str + '.0' + '.npz')['arr_0'].real
    disp_fdm_fdm_x_2lpt = disp_fdm_fdm_x_2lpt.reshape(N**3)
    disp_fdm_fdm_y_2lpt = np.load(path + 'etay2_' + input_f + '_fdm_smoothed' + redshift_str + '.0' + '.npz')['arr_0'].real
    disp_fdm_fdm_y_2lpt = disp_fdm_fdm_y_2lpt.reshape(N**3)
    disp_fdm_fdm_z_2lpt = np.load(path + 'etaz2_' + input_f + '_fdm_smoothed' + redshift_str + '.0' + '.npz')['arr_0'].real
    disp_fdm_fdm_z_2lpt = disp_fdm_fdm_z_2lpt.reshape(N**3)

# Box parameters

ncell = N
lbox  = 256.

# Growth factor

if redshift_str=='_z1':
    D = 0.608 #1. / (1+1)
elif redshift_str=='_z2':
    D = 0.418 #1. / (1+2)
elif redshift_str=='_z4':
    D = 0.254 #1. / (1+4)
elif redshift_str=='_z80':
    D = 0.0151 #1. / (1+80)
else:
    print('Redshift unrecognized')

# 2LPT growth factor

OmegaM0 = 0.31
D2      = -3./7 * (OmegaM0/(1+redshift))**(-1./143) * D**2
print(D2)

# Grid points positions

ids_2lpt = np.arange(N**3)
xL2lpt   = ids_2lpt // ncell**2
yL2lpt   = (ids_2lpt % ncell**2) // ncell
zL2lpt   = (ids_2lpt % ncell**2) % ncell

for i in range(n_lpt):

    # Set order of current calculation (first 1LPT, then 2LPT)
    n_lpt_current = i + 1
    print("Calculating %i LPT" %n_lpt_current)
    
    # Final positions
    
    if n_lpt_current == 1:
        
        final_cdm_fdm_x = ( (xL2lpt + lbox/ncell/2) + D*disp_cdm_fdm_x) % lbox
        final_cdm_fdm_y = ( (yL2lpt + lbox/ncell/2) + D*disp_cdm_fdm_y) % lbox
        final_cdm_fdm_z = ( (zL2lpt + lbox/ncell/2) + D*disp_cdm_fdm_z) % lbox
        
        final_fdm_fdm_x = ( (xL2lpt + lbox/ncell/2) + D*disp_fdm_fdm_x) % lbox
        final_fdm_fdm_y = ( (yL2lpt + lbox/ncell/2) + D*disp_fdm_fdm_y) % lbox
        final_fdm_fdm_z = ( (zL2lpt + lbox/ncell/2) + D*disp_fdm_fdm_z) % lbox
    
    elif n_lpt_current == 2:
    
        final_cdm_fdm_x = ( (xL2lpt + lbox/ncell/2) + D*disp_cdm_fdm_x + D2*disp_cdm_fdm_x_2lpt) % lbox
        final_cdm_fdm_y = ( (yL2lpt + lbox/ncell/2) + D*disp_cdm_fdm_y + D2*disp_cdm_fdm_y_2lpt) % lbox
        final_cdm_fdm_z = ( (zL2lpt + lbox/ncell/2) + D*disp_cdm_fdm_z + D2*disp_cdm_fdm_z_2lpt) % lbox
        
        final_fdm_fdm_x = ( (xL2lpt + lbox/ncell/2) + D*disp_fdm_fdm_x + D2*disp_fdm_fdm_x_2lpt) % lbox
        final_fdm_fdm_y = ( (yL2lpt + lbox/ncell/2) + D*disp_fdm_fdm_y + D2*disp_fdm_fdm_y_2lpt) % lbox
        final_fdm_fdm_z = ( (zL2lpt + lbox/ncell/2) + D*disp_fdm_fdm_z + D2*disp_fdm_fdm_z_2lpt) % lbox
        
    # Particle catalog reshaping

    part_cdm_fdm = np.c_[final_cdm_fdm_x, final_cdm_fdm_y, final_cdm_fdm_z]
    part_fdm_fdm = np.c_[final_fdm_fdm_x, final_fdm_fdm_y, final_fdm_fdm_z]

    # Save CDM displaced particles with mixed FDM IC
    
    if n_lpt_current == 1:
        out_dir = '/project/r/rbond/alague/axion_runs/modified_LPT/displacements_fields/particles/'
    elif n_lpt_current == 2:
        out_dir = '/project/r/rbond/alague/axion_runs/modified_LPT/displacements_fields/particles_2LPT/'
        
    np.save(out_dir + 'new_mixed_cdm_particles' + redshift_str, part_cdm_fdm)
    np.save(out_dir + 'new_mixed_fdm_particles' + redshift_str, part_fdm_fdm)
