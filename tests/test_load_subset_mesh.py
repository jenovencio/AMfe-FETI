# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 14:50:54 2018

@author: ge72tih
"""


import amfe
import unittest

mshfile = amfe.amfe_dir('meshes/test_meshes/3_partition_pressure_corner.msh')



# Load phys_group to a mesh
print('Loading a the entire domain to mesh')
my_mesh = amfe.Mesh()
my_mesh.import_msh(mshfile)
my_mesh.load_group_to_mesh(11,my_material,mesh_prop='phys_group')
print('Number of loaded Elements = %i' %len(my_mesh.connectivity))


# Load partition to a mesh
my_mesh = amfe.Mesh()
my_mesh.import_msh(mshfile)

print('Loading partition 1 to mesh with phys_group=11')
sub_dict ={}
sub_dict['phys_group'] = 11
sub_dict['partition_id'] = 1

my_mesh.load_subset(sub_dict, my_material)
print('Number of loaded Elements = %i' %len(my_mesh.connectivity))
#
#
#
print('Loading partition 2 to mesh with phys_group=11')
sub_dict['phys_group'] = 11
sub_dict['partition_id'] = 2
#
my_mesh.load_subset(sub_dict, my_material)
print('Number of loaded Elements = %i' %len(my_mesh.connectivity))
#
#
print('Loading partition 3 to mesh with phys_group=11')
sub_dict['phys_group'] = 11
sub_dict['partition_id'] = 3

my_mesh.load_subset(sub_dict, my_material)
print('Number of loaded Elements = %i' %len(my_mesh.connectivity))

