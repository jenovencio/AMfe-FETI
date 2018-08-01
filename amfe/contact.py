0# Copyright (c) 2017, Lehrstuhl fuer Angewandte Mechanik, Technische
# Universitaet Muenchen.
#
# Distributed under BSD-3-Clause License. See LICENSE-File for more information
#

class Contact_Elem():
    ''' This is a class to hanble contact element
    '''
    def __init__(self, master_submesh, slave_submesh, type = 'node2node', tol_radius = 1e-6):
        
        self.contact_elem_dict = {}
        self.master_normal_dict = {}
        
        if type = 'node2node':
            self.find_node_pairs(self, master_submesh,slave_submesh, tol_radius)
            self.find_master_normal()
        else:
            print('Type of contact not implemented')
            return None
    
    def find_node_pairs(self, master_submesh,slave_submesh, tol_radius = 1e-6):
        ''' find node pairs for contact given two submeshs

        parameters:
            cyclic_top_low : SubMesh
                SubMesh with the Master nodes 

            virtual_cyclic_top_high: SubMesh
                SubMesh with the Slaves nodes 

            tol_radius : float
                tolerance for finding node pairs, if a node pair do not respect the minimum 
                tolerance it will not considered as node pairs

            return : 
                contact_elem_dict : dict
                    dict that poitns master nodes to slaves

        '''

        master_nodes = master_submesh.create_node_list()
        slaves_nodes = slave_submesh.create_node_list()
        
        # master points to slave # master is a key and slave is value
        contact_elem_dict = {}
        for master_node in master_nodes:
            master_coord = cyclic_top_low.get_node_coord( master_node)
            min_dist = 1E8
            for slave_node in slaves_nodes:
                slave_coord = virtual_cyclic_top_high.get_node_coord(slave_node)
                dist = np.linalg.norm(master_coord - slave_coord)
                if dist<min_dist:
                    slave_pair = slave_node
                    min_dist = dist

            if min_dist>tol_radius:
                print('It was not possible to find a slave node for master node %i ')
            else:
                contact_elem_dict[master_node] = slave_node
        
        self.contact_elem_dict = contact_elem_dict
        return self.contact_elem_dict
    
    def find_master_normal(self, method = 'average'):
        ''' Get the normal to a node. Since there is no unique way to define the
        normal to a node, two methods are available:
        
        methods:
            first :
                compute the normal of the first element assossiated with the node
            
            average :
                compute the normal of all the elements associated with the node
                and the compute the average normal vector
        
        paramentes:
            node_id: int
                element identifier
            
            method : str
                string with 'first' or 'average'. Default value is 'average'
            
            orientation : float
                change the orientation of the normal vector either 1.0 or -1.0
        
        return
            normal_vec : np.array
        '''
        
        for master_node in self.contact_elem_dict:
            node_normal_vec = cyclic_top_high.get_normal_to_node( master_node, method)
            self.master_normal_dict[master_node] = node_normal_vec
            
    def write_files(self, filename):
        pass
        
class Cyclic_Contact_Elem(contact_elem):        
    