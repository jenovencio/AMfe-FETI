# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 16:00:53 2018

@author: ge72tih
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 14:12:05 2018

@author: ge72tih
"""

''' This module intend to test the mesh class
'''

import unittest
import amfe
import dill as pickle
import numpy as np





class test_mechanical_system(unittest.TestCase):
        
    def  test_set_domain(self):
        
        # mesh file
        mshfile = amfe.amfe_dir('meshes/test_meshes/3_partition_pressure_corner.msh')
        
        # material
        my_material = amfe.KirchhoffMaterial(E=210E9, nu=0.3, rho=7.86E3, plane_stress=True, thickness=0.1)
        
        # --------------------------------------------------------------------
        # Old way of solving AMFE case
        my_system1 = amfe.MechanicalSystem()
        
        
        my_system1.load_mesh_from_gmsh(mshfile, 11, my_material)
        my_system1.apply_dirichlet_boundaries(9, 'x')
        my_system1.apply_dirichlet_boundaries(10, 'y')
        

        my_system1.apply_neumann_boundaries(12, 1E6, 'normal', lambda t: t)
        amfe.solve_linear_displacement(my_system1)
        u1 = my_system1.u_output[1]
        
        #--------------------------------------------------------------------
        # New way of solving AMFE case with supported partitoned meshes
        
        my_mesh = amfe.Mesh()
        my_mesh.import_msh(mshfile)

        my_system = amfe.MechanicalSystem()
        my_system.set_mesh_obj(my_mesh)

        my_system.set_domain(11,my_material)
        my_system.apply_dirichlet_boundaries(9, 'x')
        my_system.apply_dirichlet_boundaries(10, 'y')
        my_system.apply_neumann_boundaries(12, 1E6, 'normal', lambda t: t)
        

        amfe.solve_linear_displacement(my_system)
        u2 = my_system.u_output[1]        
        
        error = np.linalg.norm([u2-u1])
        tol = 1E-10
        self.assertLessEqual(error, tol)
        

 
    
if __name__ == '__main__':
    unittest.main()
 