# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 14:37:26 2018

@author: ge72tih
"""

import unittest
import amfe
import copy


class test_feti_solver(unittest.TestCase):
        
    def  test_FetiSolver_linear_static(self):
        ''' This test compare stantand linear solver in AMFE with
        FETI solver
        
        test two different mesh and 6 differents load cases
        
        Please Run it from the command line:
            python test_feti_solver.py
        
        '''
        
        # list of gmsh mesh files and connectivity files
        msh_dict = {}
        msh_dict[0] = amfe.amfe_dir('meshes/test_meshes/Geom3.msh')
        msh_dict[1] = amfe.amfe_dir('meshes/test_meshes/simple_2.msh')
        
        
         # load cases 
        load_case = {}
        load_case[0] = {'value':1E1, 'direction':'normal'}
        load_case[1] = {'value':1E2, 'direction':'normal'}
        load_case[2] = {'value':1E3, 'direction':'normal'}
        load_case[3] = {'value':1E1, 'direction':[0,-1]}
        load_case[4] = {'value':1E2, 'direction':[0,-1]}
        load_case[5] = {'value':5E3, 'direction':[0,-1]}
        
        for key, mshfile in msh_dict.items():
            # Load phys_group to a mesh
            m = amfe.Mesh()
            m.import_msh(mshfile)
            m.split_in_groups()
            #        
            #plot_boundary_1d(m)
            #plt.show()
            
            # creating material
            my_material = amfe.KirchhoffMaterial(E=210E9, nu=0.3, rho=7.86E3, plane_stress=True, thickness=0.1)
            #
            ### creating mechanical system
            my_system = amfe.MechanicalSystem()
            my_system.set_mesh_obj(m)
            my_system.set_domain(3,my_material)
            #
            ##
            my_system.apply_dirichlet_boundaries(1, 'x')
            my_system.apply_dirichlet_boundaries(1, 'y')
            
            for load_case_id in load_case:
                load_case_dict = load_case[load_case_id]
                my_system.apply_neumann_boundaries(2, load_case_dict['value'], load_case_dict['direction'])
                
                amfe.FetiSolver.linear_static(my_system)
                feti_disp = my_system.u_output[1]
        
                my_system2 = copy.deepcopy(my_system)
                s = amfe.LinearStaticsSolver(my_system2)
                amfe.LinearStaticsSolver(my_system)
                s.solve()
                amfe_disp = my_system2.u_output[1]
        
                self.assertAlmostEquals(amfe_disp.any(),feti_disp.any())



if __name__ == '__main__':
    print('''
    Please Run it from the command line:
            python test_feti_solver.py
    ''')
    
    unittest.main()
                 