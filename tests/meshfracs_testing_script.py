import meshfracs
import os

mesh_location = '../Western_US_Dev/meshes/water_only_mesh.npy'
gebco_location = '../abetadatabase/elevation/gebco/GEBCO'
conductance_location = '../abetadatabase/world_conductance/grayver2021/conductance_world.h5'
working_dir = 'testing_dir'
meshfracs.calculate_bulk_fractions.calculate_conductivity(mesh_location,
                                                          gebco_location,
                                                          conductance_location,
                                                          working_dir,debug_mode=True)
print('here')