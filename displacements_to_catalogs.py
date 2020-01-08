import numpy as np
#import matplotlib.pyplot as plt

path = '/project/r/rbond/alague/axion_runs/modified_LPT/displacements_fields/fields/'

# Number of particles           

N = len(np.fromfile(path + 'etax1_256Mpc_n512_nb40_nt1_1_28_10',dtype=np.float32))**(1./3)
N = int(round(N))

# Import CDM particles which are unchanged with redshift (just scale the growth factor)

# CDM particles displacements with CDM IC
#disp_cdm_cdm_x = np.fromfile(path + 'etax1_256Mpc_n512_nb40_nt1_lcdm', dtype=np.float32,count=N**3)
#disp_cdm_cdm_y = np.fromfile(path + 'etay1_256Mpc_n512_nb40_nt1_lcdm', dtype=np.float32,count=N**3)
#disp_cdm_cdm_z = np.fromfile(path + 'etaz1_256Mpc_n512_nb40_nt1_lcdm', dtype=np.float32,count=N**3)

# CDM particles displacements with mixed FDM IC
disp_cdm_fdm_x = np.fromfile(path + 'etax1_256Mpc_n512_nb40_nt1_1_28_10', dtype=np.float32,count=N**3)
disp_cdm_fdm_y = np.fromfile(path + 'etay1_256Mpc_n512_nb40_nt1_1_28_10', dtype=np.float32,count=N**3)
disp_cdm_fdm_z = np.fromfile(path + 'etaz1_256Mpc_n512_nb40_nt1_1_28_10', dtype=np.float32,count=N**3)


# Loop over redshifts
for redshift in ['_z1']:#,'_z1','_z2','_z4','_z80']:
        
    # FDM particles displacements with mixed FDM IC
    
    disp_fdm_fdm_x = np.load(path + 'etax1_256Mpc_n512_nb40_nt1_1_28_10_fdm' + '_smoothed_z1.0' + '.npz')['arr_0'].real
    disp_fdm_fdm_x = disp_fdm_fdm_x.reshape(N**3)
    disp_fdm_fdm_y = np.load(path + 'etay1_256Mpc_n512_nb40_nt1_1_28_10_fdm' + '_smoothed_z1.0' + '.npz')['arr_0'].real
    disp_fdm_fdm_y = disp_fdm_fdm_y.reshape(N**3)
    disp_fdm_fdm_z = np.load(path + 'etaz1_256Mpc_n512_nb40_nt1_1_28_10_fdm' + '_smoothed_z1.0' + '.npz')['arr_0'].real
    disp_fdm_fdm_z = disp_fdm_fdm_z.reshape(N**3)

    # Box parameters
    
    ncell = N
    lbox  = 256.
    
    # Growth factor
    
    if redshift=='_z1':
        D = 1. / (1+1)
    elif redshift=='_z2':
        D = 1. / (1+2)
    elif redshift=='_z4':
        D = 1. / (1+4)
    elif redshift=='_z80':
        D = 1. / (1+80)
    else:
        D = 1.

    ids_2lpt = np.arange(N**3)
    
    xL2lpt = ids_2lpt // ncell**2
    yL2lpt = (ids_2lpt % ncell**2) // ncell
    zL2lpt = (ids_2lpt % ncell**2) % ncell
    
    # Final positions
    
    #final_cdm_cdm_x = ( (xL2lpt + lbox/ncell/2) + D*disp_cdm_cdm_x) % lbox
    #final_cdm_cdm_y = ( (yL2lpt + lbox/ncell/2) + D*disp_cdm_cdm_y) % lbox
    #final_cdm_cdm_z = ( (zL2lpt + lbox/ncell/2) + D*disp_cdm_cdm_z) % lbox
    
    final_cdm_fdm_x = ( (xL2lpt + lbox/ncell/2) + D*disp_cdm_fdm_x) % lbox
    final_cdm_fdm_y = ( (yL2lpt + lbox/ncell/2) + D*disp_cdm_fdm_y) % lbox
    final_cdm_fdm_z = ( (zL2lpt + lbox/ncell/2) + D*disp_cdm_fdm_z) % lbox
    
    final_fdm_fdm_x = ( (xL2lpt + lbox/ncell/2) + D*disp_fdm_fdm_x) % lbox
    final_fdm_fdm_y = ( (yL2lpt + lbox/ncell/2) + D*disp_fdm_fdm_y) % lbox
    final_fdm_fdm_z = ( (zL2lpt + lbox/ncell/2) + D*disp_fdm_fdm_z) % lbox
    
    # Particle catalog reshaping

    #part_cdm_cdm = np.c_[final_cdm_cdm_x, final_cdm_cdm_y, final_cdm_cdm_z]
    part_cdm_fdm = np.c_[final_cdm_fdm_x, final_cdm_fdm_y, final_cdm_fdm_z]
    part_fdm_fdm = np.c_[final_fdm_fdm_x, final_fdm_fdm_y, final_fdm_fdm_z]

    # Save CDM displaced particles with mixed FDM IC

    #np.save('particles/mixed_cdm_particles' + redshift, part_cdm_fdm)
    #np.save('particles/pure_fdm_particles' + redshift, part_fdm_fdm)
    
    np.save('displacements_fields/particles/new_mixed_cdm_particles' + redshift, part_cdm_fdm)
    np.save('displacements_fields/particles/new_mixed_fdm_particles' + redshift, part_fdm_fdm)
    

    '''
    # Select 10% of FDM particles and 90% of CDM particles

    random_10percent = np.random.choice(np.arange(N**3),int(0.10*N**3)) # choose 10% of particles at random
    fdm_indexes      = np.zeros(N**3)
    cdm_indexes      = np.ones(N**3)
    
    fdm_indexes[random_10percent] = 1.
    cdm_indexes[random_10percent] = 0.
    
    part_fdm_fdm = part_fdm_fdm[fdm_indexes.astype(bool)]
    part_cdm_fdm = part_cdm_fdm[cdm_indexes.astype(bool)]
    
    # Combine into new catalog

    part_mixed_cdm_fdm = np.append(part_cdm_fdm,part_fdm_fdm,axis=0)

    # Save to file to use with nbodykit
    
    np.save('particles/pure_cdm_particles' + redshift, part_cdm_cdm)
    np.save('particles/mixed_fdm_particles' + redshift, part_mixed_cdm_fdm)
    '''
    '''
    if redshift == '':
        
        ### CURRENTLY ONLY FOR z=0 ###
        
        # Load masses for particles
        
        cdm_ic = np.fromfile('fields/Fvec_256Mpc_n512_nb40_nt1_lcdm',dtype=np.float32,count=N**3)
        cdm_ic = cdm_ic.reshape((N,N,N))
        fdm_ic = np.fromfile('fields/Fvec_256Mpc_n512_nb40_nt1_1_28_10',dtype=np.float32,count=N**3)
        fdm_ic = fdm_ic.reshape((N,N,N))
        
        # Save masses
        
        mass_pure_cdm  = cdm_ic
        mass_fdm       = fdm_ic
        mass_mixed_cdm = fdm_ic[cdm_indexes.astype(bool)]
        mass_mixed_fdm = fdm_ic[fdm_indexes.astype(bool)]

        mass_mixed_cdm_fdm = np.append(mass_mixed_cdm,mass_mixed_fdm,axis=0)
        
        np.save('particles/mixed_cdm_masses' + redshift, mass_fdm)
        np.save('particles/pure_cdm_masses' + redshift, mass_pure_cdm)
        np.save('particles/mixed_fdm_masses' + redshift, mass_mixed_cdm_fdm)
        
    '''
