0# Copyright (c) 2017, Lehrstuhl fuer Angewandte Mechanik, Technische
# Universitaet Muenchen.
#
# Distributed under BSD-3-Clause License. See LICENSE-File for more information
#

import numpy as np
import copy

class Contact():
    ''' This is a class to hanble contact element
    '''
    def __init__(self, master_submesh, slave_submesh, type = 'node2node', tol_radius = 1e-6):
        
        self.master_submesh = master_submesh
        self.slave_submesh =  slave_submesh
        self.contact_elem_dict = {}
        self.master_normal_dict = {}
        self.master_nodes = []
        self.slaves_nodes = []
        
        if type == 'node2node':
            self.find_node_pairs(master_submesh,slave_submesh, tol_radius)
            self. create_master_normal_dict()
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
            master_coord = master_submesh.get_node_coord( master_node)
            min_dist = 1E8
            for slave_node in slaves_nodes:
                slave_coord = slave_submesh.get_node_coord(slave_node)
                dist = np.linalg.norm(master_coord - slave_coord)
                if dist<min_dist:
                    slave_pair = slave_node
                    min_dist = dist

            if min_dist>tol_radius:
                print('It was not possible to find a slave node for master node %i. Minimum distance is %e' %(master_node,min_dist))
            else:
                contact_elem_dict[master_node] = slave_pair
                self.master_nodes.append(master_node)
                self.slaves_nodes.append(slave_pair)
                
        self.contact_elem_dict = contact_elem_dict
        return self.contact_elem_dict
    
    def create_master_normal_dict(self, method = 'average'):
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
            self.master_normal_dict : dict
                dict which maps master nodes to the normal vector
        '''
        
        for master_node in self.contact_elem_dict:
            node_normal_vec = self.get_normal_at_master_node(master_node, method)
            self.master_normal_dict[master_node] = node_normal_vec
        return self.master_normal_dict
    
    def get_normal_at_master_node(self, master_node, method = 'average'):
        ''' get the normal vector of given a node
        
        parameters:
            master_node : int   
               id of the master node
            method: str
                string specifying the method to compute normal at node
        return
            normal_vector : np.array
        '''
        return self.master_submesh.get_normal_to_node( master_node, method)
    
    def write_files(self, filename):
        pass
        
class Cyclic_Contact(Contact):     
    ''' This class intend to handle cyclic contact problem,
    where master and slaves have a angule between them.
    Basically, the slave SubMesh is rotate (Virtual Slave) by the sector angule 
    and node pair are found by the minimum Euclidian distance.
    
    
    '''
    def __init__(self, master_submesh, slave_submesh, sector_angle= 0, unit = 'deg', type = 'node2node', tol_radius = 1e-6 ):
        
        virtual_slave = virtual_cyclic_top_high = copy.deepcopy(slave_submesh)
        virtual_slave.rot_z(sector_angle, unit)
        self.virtual_slave = virtual_slave
        self.sector_angle = sector_angle
        self.unit = unit
        super(Cyclic_Contact,self).__init__(master_submesh,virtual_slave, type, tol_radius)
        
    