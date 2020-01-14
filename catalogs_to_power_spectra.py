import numpy as np

from nbodykit.lab import LinearMesh, cosmology
from nbodykit.lab import ArrayMesh
from nbodykit.lab import FFTPower
from nbodykit.source import catalog

h   = 0.68
box = 256. #* h
path = '/project/r/rbond/alague/axion_runs/modified_LPT/displacements_fields/'

for redshift in ['_z1']:#,'_z1','_z2','_z4','_z80']:
    # Load particles

    #pure_cdm_particles  = np.load('particles/pure_cdm_particles' + redshift + '.npy')  # 100% CDM with CDM IC
    mixed_fdm_particles = np.load(path + 'particles/new_mixed_fdm_particles' + redshift + '.npy') # mixed case 10% FDM and 90% CDM with FDM IC
    mixed_cdm_particles = np.load(path + 'particles/new_mixed_cdm_particles' + redshift + '.npy') # CDM displacements but with FDM IC / equivalent to WDM app
    
    # Initialize dictionaries
    
    #part_dict_pure_cdm  = {}
    part_dict_mixed_fdm = {}
    part_dict_mixed_cdm = {}

    # Fill dictionaries with particle positions
    
    #part_dict_pure_cdm['Position']  = pure_cdm_particles
    part_dict_mixed_fdm['Position'] = mixed_fdm_particles
    part_dict_mixed_cdm['Position'] = mixed_cdm_particles

    # Load density fields
    
    #pure_cdm_masses  = np.load('particles/pure_cdm_masses' + redshift + '.npy') 
    #mixed_fdm_masses = np.load('particles/mixed_fdm_masses' + redshift + '.npy')
    #mixed_cdm_masses = np.load('particles/mixed_cdm_masses' + redshift + '.npy')
    
    # Fill dictionaries with particle masses
    
    #part_dict_pure_cdm['Mass']  = pure_cdm_masses
    #part_dict_mixed_fdm['Mass'] = mixed_fdm_masses
    #part_dict_mixed_cdm['Mass'] = mixed_cdm_masses
    
    # Create array catalogs

    #cdm_cat   = catalog.ArrayCatalog(part_dict_pure_cdm)
    fdm_cat   = catalog.ArrayCatalog(part_dict_mixed_fdm)
    mixed_cat = catalog.ArrayCatalog(part_dict_mixed_cdm)

    # Paint catalogs to mesh
    
    #cdm_mesh   = cdm_cat.to_mesh(Nmesh=128,BoxSize=box,compensated=True,interlaced=True)
    fdm_mesh   = fdm_cat.to_mesh(Nmesh=128,BoxSize=box,compensated=True,interlaced=True)
    mixed_mesh = mixed_cat.to_mesh(Nmesh=128,BoxSize=box,compensated=True,interlaced=True)
    
    # Calculate power spectra

    #cdm_pk   = FFTPower(cdm_mesh, mode='1d')
    fdm_pk   = FFTPower(fdm_mesh, mode='1d')
    mixed_pk = FFTPower(mixed_mesh, mode='1d')
    
    # Save output
    
    k        = fdm_pk.power['k']
    #cdm_pk   = cdm_pk.power['power'].real
    fdm_pk   = fdm_pk.power['power'].real
    mixed_pk = mixed_pk.power['power'].real
    
    #np.save('power_spectra/cdm_pk' + redshift, np.array([k, cdm_pk]))
    np.save(path + 'power_spectra/new_fdm_pk' + redshift, np.array([k, fdm_pk]))
    np.save(path + 'power_spectra/new_cdm_pk' + redshift, np.array([k, mixed_pk]))
    
    print('Redshift ' + redshift + ' complete')
