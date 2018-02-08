# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 15:40:03 2018

@author: ge72tih
"""
import amfe
import copy
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------------
case_id = 1

msh_dict = {}
msh_dict[0] = amfe.amfe_dir('meshes/test_meshes/simple_2.msh')
msh_dict[1] = amfe.amfe_dir('meshes/test_meshes/Geom3.msh')

load_case = {}
load_case[0] = {'value':1E8, 'direction':'normal'}
load_case[1] = {'value':1E7, 'direction':[0,-1]}

#-----------------------------------------------------------------------------
#                                FEA Setup
#-----------------------------------------------------------------------------

# Load mesh file
print('Loading a the entire domain to mesh')


mshfile = msh_dict[case_id]
m = amfe.Mesh()
m.import_msh(mshfile)
        
# creating material
my_material = amfe.KirchhoffMaterial(E=210E9, nu=0.3, rho=7.86E3, plane_stress=True, thickness=0.1)
#
## creating mechanical system
my_system = amfe.MechanicalSystem()
my_system.set_mesh_obj(m)
my_system.set_domain(3,my_material)
#
#
my_system.apply_dirichlet_boundaries(1, 'x')
my_system.apply_dirichlet_boundaries(1, 'y')
my_system.apply_neumann_boundaries(2, load_case[case_id]['value'], load_case[case_id]['direction'])


#-----------------------------------------------------------------------------
#                        Solving with FETI solver
#-----------------------------------------------------------------------------

residual = amfe.FetiSolver.linear_static(my_system)

#-----------------------------------------------------------------------------
#                 Solving with AMFE linear solver
#-----------------------------------------------------------------------------

my_system2 = copy.deepcopy(my_system)
s = amfe.LinearStaticsSolver(my_system2)
s.solve()


#-----------------------------------------------------------------------------
#                   Ploting and compare results
#-----------------------------------------------------------------------------
connectivity = my_system2.mesh_class.connectivity
nodes = my_system2.mesh_class.nodes
tri, ax = amfe.plotDeformMesh(connectivity,nodes,my_system.u_output[1],1) 
amfe.plotDeformMesh(connectivity,nodes,my_system2.u_output[1],1,ax) 
ax.legend(['FETI solver','Standard AMFE solver'])
plt.show()