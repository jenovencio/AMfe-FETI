# Copyright (c) 2017, Lehrstuhl fuer Angewandte Mechanik, Technische
# Universitaet Muenchen.
#
# Distributed under BSD-3-Clause License. See LICENSE-File for more information
#

import numpy as np
import copy
from scipy import sparse
import numdifftools as nd

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
        
    

def H(x):
    if x>=1.0E-16:
        return 1.0
    elif x<-1.0E-16:
        return 0.0
    else:
        return 0.5

def R(x):
    if x<=0.0:
        return 0.0
    else:
        return x

def df_normf(x): 
    norm_x = np.linalg.norm(x)
    if norm_x>0.0:
        return (1.0/norm_x)*np.eye(len(x)) - (1.0/norm_x**3)*np.outer(x,x)
    else:
        return np.eye(len(x))*0.0


class jenkins():
    
    def __init__(self, X_contact, X_target, normal, ro = 1.0E8 ,k=1.0E7, mu=0.2, N0 = 0.0):
        
        # contact properties
        self.ro = ro
        self.k = k
        
        self.mu = mu
        self.N0 = N0
        
        #contact parameters
        self.dim = 2*int(X_target.shape[0])
        self.X_target = X_target
        self.X_contact = X_contact
        self.initial_relative_displacement = X_contact - X_target
        self.normal = normal

        if np.abs(np.linalg.norm(self.normal) - 1.0)>1.0E-12:
            raise ValueError('Normal vector has norm different than 1.0!')
        
        self.alpha_0 = np.zeros(X_target.shape)
        self.gap_n = np.zeros(X_target.shape)

        self._f = lambda u, x0 : self.compute(u, x0)

        self.n_var = int(self.dim/2)
        self.n_vec = np.concatenate([self.normal,-self.normal])
        I = sparse.eye(self.n_var)
        self.B = sparse.bmat([[I,-I],[-I,I]]).A
        self.Bv = np.outer( self.n_vec, self.n_vec)
        
    def compute_gap_and_tangent(self, u, un):
        
        dim = self.dim
        if dim%2:
            raise ValueError('Displacement vector is not compatible')

        u_target = u[:int(dim/2)]
        u_contact = u[int(dim/2):]
        
        un_target = un[:int(self.dim/2)]
        un_contact = un[int(dim/2):]
        
        x_contact = self.X_contact + u_contact
        x_target = self.X_target + u_target

        xn_contact = self.X_contact + un_contact
        xn_target = self.X_target + un_target
        
        delta = x_contact - x_target
        gap = self.normal.dot(delta)
        delta_gap = delta - xn_contact + xn_target
        u_tangent = delta_gap.dot(self.normal)*self.normal - delta_gap 
        
        
        if u_tangent.dot(self.normal)>1.E-10:
            raise ValueError('Tangent and Normal vector are not orthogonal!')

        return gap, u_tangent        

    def compute_gap_and_tangent_by_relative_displacement(self, delta_u, delta_un):    
        
        delta = -delta_u + self.initial_relative_displacement
        gap = self.normal.dot(delta)
        delta_gap = delta_u - delta_un 
        u_tangent = delta_gap.dot(self.normal)*self.normal - delta_gap 
        
        
        if u_tangent.dot(self.normal)>1.E-10:
            raise ValueError('Tangent and Normal vector are not orthogonal!')

        return gap, u_tangent        

    
    def compute_tangent_force(self,tangent,N):
        
        N += self.N0
        delta_gap = tangent
        self.Ft_trial = Ft_trial = self.alpha_0 - self.k*delta_gap 
        norm_Ft_trial = np.linalg.norm(Ft_trial)
        if norm_Ft_trial>0:
            d = Ft_trial/norm_Ft_trial
        else:
            d = self.normal

        Phi = norm_Ft_trial - self.mu*N

        self.alpha = -R(Phi)*d + Ft_trial
        return self.alpha
        
    def compute(self,u,x):
        '''
        Compute force based on current and previous displacement vector

        parameters:
            u : np.array
                current displacement vector
            x : np.array
                previous displacement vector

        return 
            force : np.array
        '''
        gap, u_tangent = self.compute_gap_and_tangent(u, x)
        self.N = N = -self.ro*R(-gap) 
        normal_force = N*self.normal
        tangent_force = self.compute_tangent_force(u_tangent,N)
        force = normal_force + tangent_force
        return np.concatenate([force,-force])

    def compute_force_by_relative_displacement(self,delta_u,delta_un):
        '''
        Compute force based on current and previous displacement vector

        parameters:
            delta_u : np.array
                current relative displacement vector between to interfaces
            delta_un : np.array
                previous relative displacement vector between to interfaces

        return 
            force : np.array
        '''
        gap, u_tangent = self.compute_gap_and_tangent_by_relative_displacement(delta_u,delta_un)
        self.N = N = -self.ro*R(-gap) 
        normal_force = N*self.normal
        tangent_force = self.compute_tangent_force(u_tangent,N)
        force = normal_force + tangent_force
        return force

    def update_alpha(self):
        self.alpha_0 = self.alpha

    def refresh_alpha(self):
        self.alpha_0 = 0.0*self.alpha

    def compute_jac(self,u,x0,method='central-difference'):

        
        if method=='central-difference':
            Jfun = nd.Jacobian(lambda u : self._f(u,x0),method='forward')
            J_eval = Jfun(u) 
            return  J_eval

        else:
            raise NotImplementedError('Method is not impemented!')

    def Jacobian(self,u, un):

        gap, u_tangent = self.compute_gap_and_tangent(u, un)
        
        n_var = self.n_var
        n_vec = self.n_vec
        
        B = self.B
        Bv = self.Bv
        duTdu = B - Bv
        dRdu = -self.ro*H(-gap)
        dfx = dRdu*Bv
        dftrialdu = -self.k*duTdu

        def direct(u):
            norm_Ft_trial = np.linalg.norm(Ft_trial) 
            if norm_Ft_trial>0.0:
                d = Ft_trial/norm_Ft_trial
            else:
                d =  self.normal
            return np.concatenate([d,-d])
        
        Ft_trial = self.Ft_trial
        N = self.N
        Phi = np.linalg.norm(Ft_trial) - self.mu*N
        Rphi = -R(Phi)
        dPhi_ana = lambda u : dftrialdu[:,:n_var].dot(direct(u)[:n_var]) - self.mu*dfx[:,:n_var].dot(self.normal)
        dd_ana = df_normf(np.concatenate([Ft_trial,-Ft_trial])).dot(dftrialdu)
        dp1_ana = lambda u : np.outer(direct(u),-H(Phi)*dPhi_ana(u)) - Rphi*dd_ana

        if gap>=0:
            JFtdu_eval =  np.zeros((self.dim, self.dim))
        else:
            JFtdu_eval = dp1_ana(u)  + dftrialdu

        JFt = JFtdu_eval
        J_ana = dfx.T + JFt
        return J_ana



class Nonlinear_force_assembler():
    def __init__(self,map_dict,contact_list):
        self.map_dict = map_dict
        self.contact_list = contact_list
        self._jac = None
        self.force = None
        self.global_jac = None
        self.row_indices = None
        self.col_indices = None

    def _prealloc_jac(self,m):

        row_indices = np.array([],dtype=np.int)
        col_indices = np.array([],dtype=np.int)

        for key,global_index in self.map_dict.items():
            #local_jac_ = self.contact_list[key].Jacobian(u[global_index],X0[global_index])
            #rows, cols = np.meshgrid(global_index,global_index)
            cols, rows = np.meshgrid(global_index,global_index)
            row_indices =  np.concatenate((row_indices,rows.flatten()))
            col_indices =  np.concatenate((col_indices,cols.flatten()))
            
        data = np.zeros(len(row_indices))
        self.global_jac = sparse.coo_matrix((data,(row_indices,col_indices)),shape=(m,m))
        #self.global_jac.tocsr()
        self.row_indices = row_indices
        self.col_indices = col_indices
        return self.global_jac

    def Jacobian(self,u=None,X0=None):
                
        
        if self.global_jac is None:
            #self.global_jac = sparse.lil_matrix((u.shape[0], u.shape[0]))
            self._prealloc_jac(u.shape[0])
            
        #global_jac = 0.0*self.global_jac
        data = np.array([])
        for key,global_index in self.map_dict.items():
            local_jac_ = self.contact_list[key].Jacobian(u[global_index],X0[global_index])
            data = np.concatenate((data,local_jac_.flatten()))
            #self.global_jac[np.ix_(global_index,global_index)] = local_jac_
        
        self.global_jac.data = data
        return self.global_jac

    def compute(self,u,X0):
        
        try:
            self.force = 0.0*self.force
        except:
            self.force = np.zeros(u.shape)
        
        v = compute_force(self.force,u,X0,self.map_dict,self.contact_list)
        return v
        
    def update_alpha(self):

        for c in self.contact_list:
            c.update_alpha()

    def refresh_alpha(self):
        for c in self.contact_list:
            c.refresh_alpha()


def compute_force(global_force,u,X0,map_dict,contact_list):
    for key,global_index in map_dict.items():
        local_force = contact_list[key].compute(u[global_index],X0[global_index])
        global_force[global_index] = local_force
   
    return global_force




class Create_node2node_force_object():
    """
    This class provides data and methods to create nonlinear force 
    object

    Properties 
    contact : Contact obj
        object with conctact information
    bodies_contact_id : tuple
        tuple with bodies id (i,j) where i is the master id and j is the slave id
    elem_type : string
        string with the element type, must be implemented in the contact module
    elem_properties : dict
        dict with the properties of the contact element
    dimension: int
        dimension of the problem
    map_local_domain_dofs_dimension : dict
        dict with where key is the body id and value is the number of dofs
    """
    def __init__(self,contact,bodies_contact_id,elem_type,elem_properties,dimension,map_local_domain_dofs_dimension):
        
        self.contact = contact
        self.bodies_contact_id = bodies_contact_id
        self.elem_type = elem_type
        self.elem_properties = elem_properties
        self.dimension = dimension
        self.map_local_domain_dofs_dimension = map_local_domain_dofs_dimension

        self._find_shift()
        
    def _find_shift(self):

        body_id_list = list(self.map_local_domain_dofs_dimension.keys())
        body_id_list.sort()
        self.shift_dict = {}
        shift = 0
        for body_id in body_id_list:
            self.shift_dict[body_id] = shift
            shift+=self.map_local_domain_dofs_dimension[body_id]

        return self.shift_dict
        
    def assemble_nonlinear_force(self):
        
        dimension = self.dimension
        elem_obj_ = globals()[self.elem_type]
        map_dict = {}
        contact_id = 0
        contact_list = []
        for key, value in self.contact.contact_elem_dict.items():
            shift_master = self.shift_dict[self.bodies_contact_id[0]]
            shift_slave = self.shift_dict[self.bodies_contact_id[1]]
            start_master = shift_master + key*dimension
            start_slave = shift_slave + value*dimension
            dof_list = list(range(start_master,start_master+dimension))
            dof_list.extend(list(range(start_slave,start_slave+dimension)))
            map_dict[contact_id] = dof_list
            X_target = self.contact.master_submesh.parent_mesh.nodes[key]
            X_contact =  self.contact.slave_submesh.parent_mesh.nodes[value]
            normal = -self.contact.master_normal_dict[key][:dimension]
            elem_obj = elem_obj_(X_contact, X_target, normal,**self.elem_properties)
            contact_list.append(elem_obj)
            contact_id +=1

        Fnl_obj = Nonlinear_force_assembler(map_dict,contact_list)
        return Fnl_obj

    def get_master_B(self):
        pass

    def get_slave_B(self):
        pass