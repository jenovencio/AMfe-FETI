# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 14:21:14 2018

@author: ge72tih
"""

import sys
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import copy

import amfe



mshfile = amfe.amfe_dir('meshes/test_meshes/3_partition_pressure_corner.msh')
my_material = amfe.KirchhoffMaterial(E=210E9, nu=0.3, rho=7.86E3, plane_stress=True, thickness=0.1)


# Load phys_group to a mesh
print('Loading a the entire domain to mesh')
my_mesh = amfe.Mesh()
my_mesh.import_msh(mshfile)
my_mesh.load_group_to_mesh(9,my_material,mesh_prop='phys_group')
my_mesh.load_group_to_mesh(10,my_material,mesh_prop='phys_group')
my_mesh.load_group_to_mesh(11,my_material,mesh_prop='phys_group')
my_mesh.load_group_to_mesh(12,my_material,mesh_prop='phys_group')
my_mesh.load_group_to_mesh(13,my_material,mesh_prop='phys_group')
print('Number of loaded Elements = %i' %len(my_mesh.connectivity))


# Load partition to a mesh
my_mesh = amfe.Mesh()
my_mesh.import_msh(mshfile)

print('Loading partition 1 to mesh')
my_mesh.load_group_to_mesh(1,my_material,mesh_prop='partition_id')
print('Number of loaded Elements = %i' %len(my_mesh.connectivity))


print('Loading partition 2 to mesh')
my_mesh.load_group_to_mesh(2,my_material,mesh_prop='partition_id')
print('Number of loaded Elements = %i' %len(my_mesh.connectivity))

print('Loading partition 3 to mesh')
my_mesh.load_group_to_mesh(3,my_material,mesh_prop='partition_id')
print('Number of loaded Elements = %i' %len(my_mesh.connectivity))