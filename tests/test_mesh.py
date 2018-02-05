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



def save_obj(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def load_obj(filename):
    with open(filename, 'rb') as input:
        obj = pickle.load(input)
    return obj





class test_mesh(unittest.TestCase):
        
    def  test_import_msh(self):
        
        # list of gmsh mesh files and connectivity files
        
        files_list = ['bar_3d',
                      '2_partitions_mesh',
                      '3_partition_pressure_corner'
                      ]
        
        for file_id in range(len(files_list)):
            #file_id = 1
            filename_path = amfe.amfe_dir('meshes/test_meshes/' + files_list[file_id] + '.msh')
            obj_path = amfe.amfe_dir('meshes/test_meshes/' + files_list[file_id] + '.pkl')
            my_mesh = amfe.Mesh()
            
            
            var = my_mesh.import_msh(filename_path)
            self.assertEqual(var, None)
            base_mesh = load_obj(obj_path)
            
            # compare base_mesh variables with new mesh variables
            self.assertEqual(base_mesh.no_of_nodes,my_mesh.no_of_nodes)
            self.assertEqual(base_mesh.no_of_dofs,my_mesh.no_of_dofs)
            

            for key_id in range(5): 
                self.assertEqual(base_mesh.el_df.keys()[key_id],my_mesh.el_df.keys()[key_id])
                
        
    def test_load_subset_1(self):
        # Load phys_group to a mesh
        print('Loading a the entire domain to mesh')
        
        my_material = amfe.KirchhoffMaterial(E=210E9, nu=0.3, rho=7.86E3, plane_stress=True, thickness=0.1)
        mshfile = amfe.amfe_dir('meshes/test_meshes/3_partition_pressure_corner.msh')
        
        my_mesh = amfe.Mesh()
        my_mesh.import_msh(mshfile)
        my_mesh.load_group_to_mesh(11,my_material,mesh_prop='phys_group')
        
        
        
        n1 = len(my_mesh.connectivity)
        print('Number of loaded Elements = %i' %n1)
        
        
        # Load partition to a mesh
        my_mesh = amfe.Mesh()
        my_mesh.import_msh(mshfile)
        
        print('Loading partition 1 to mesh with phys_group=11')
        sub_dict ={}
        sub_dict['phys_group'] = 11
        
        my_mesh.load_subset(sub_dict, my_material)
        n2 = len(my_mesh.connectivity)
        print('Number of loaded Elements = %i' %n2)
        
        self.assertEqual(n2,n1)
        
    def test_load_subset_2(self):
        
        my_material = amfe.KirchhoffMaterial(E=210E9, nu=0.3, rho=7.86E3, plane_stress=True, thickness=0.1)
        mshfile = amfe.amfe_dir('meshes/test_meshes/3_partition_pressure_corner.msh')
        
        # Load phys_group to a mesh
        print('Loading a the entire domain to mesh')
        my_mesh = amfe.Mesh()
        my_mesh.import_msh(mshfile)
        my_mesh.load_group_to_mesh(11,my_material,mesh_prop='phys_group')
        n1 = len(my_mesh.connectivity)
        print('Number of loaded Elements = %i' %n1)
        
        
        # Load partition to a mesh
        my_mesh = amfe.Mesh()
        my_mesh.import_msh(mshfile)
        
        print('Loading partition 1 to mesh with phys_group=11')
        sub_dict ={}
        sub_dict['phys_group'] = 11
        sub_dict['partition_id'] = 1
        
        my_mesh.load_subset(sub_dict, my_material)
        n2 = len(my_mesh.connectivity)
        print('Number of loaded Elements = %i' %n2)
        
        self.assertLessEqual(n2, n1)
        
    def test_split_in_groups(self):      
        
        mshfile = amfe.amfe_dir('meshes/test_meshes/3_partition_pressure_corner.msh')
        m = amfe.Mesh()
        m.import_msh(mshfile)
        
        m.split_in_groups()
        eval_list =  list(m.groups.keys())
        right_list = [9, 12, 10, 13, 11]
        
        self.assertListEqual(eval_list, right_list)
 
        
        
#t = test_mesh()
#t.test_import_msh()


#-----------------------------------------------------------------------------       
#creating python objection to use in unittest comparison   
#-----------------------------------------------------------------------------

# seeting paths for files
#gmsh_input_file =amfe.amfe_dir('meshes/test_meshes/2_partitions_mesh.msh')
#gmsh_input_file =amfe.amfe_dir('meshes/test_meshes/3_partition_pressure_corner.msh')
#gmsh_connec_file =amfe.amfe_dir('meshes/test_meshes/3_partition_pressure_corner.pkl')
#gmsh_connec_file =amfe.amfe_dir('meshes/test_meshes/2_partitions_mesh.pkl')
#gmsh_input_file =amfe.amfe_dir('meshes/test_meshes/bar_3d.msh')
#gmsh_connec_file =amfe.amfe_dir('meshes/test_meshes/bar_3d.pkl')


#my_mesh = amfe.Mesh()
#my_mesh.import_msh(gmsh_input_file)
#save_obj(my_mesh,gmsh_connec_file)
        
# load the mesh obj
#my_mesh = load_obj(gmsh_connec_file)        
#

if __name__ == '__main__':
    unittest.main()
 