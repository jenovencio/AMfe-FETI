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
 