# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 15:17:42 2018

@author: ge72tih
"""

import amfe
import matplotlib.pyplot as plt

msh_dict = {}
msh_dict[0] = amfe.amfe_dir('meshes/test_meshes/Geom3.msh')
msh_dict[1] = amfe.amfe_dir('meshes/test_meshes/simple_2.msh')
msh_dict[2] = mshfile = amfe.amfe_dir('meshes/test_meshes/3_partition_pressure_corner.msh')
msh_dict[3] = mshfile = amfe.amfe_dir('meshes/test_meshes/geo_hole_quad_part_4.msh')

domain_id = {}
domain_id[0] = 3
domain_id[1] = 3
domain_id[2] = 11
domain_id[3] = 8

# select mesh to be plotted
mesh_id = 3
mshfile = msh_dict[mesh_id]

m = amfe.Mesh()
m.import_msh(mshfile)

# splitting physical grops

m.split_in_groups()

# plotting boundary elements
amfe.plot_boundary_1d(m)



# plotting mesh
amfe.plot_submesh(m.groups[domain_id[mesh_id]])


# setting main domain for FE calculation
domain = m.set_domain('phys_group', domain_id[mesh_id])
domain.split_in_partitions()
amfe.plot_domain(domain)


plt.show()
