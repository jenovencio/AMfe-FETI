# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 10:26:53 2018

@author: ge72tih
"""
import sys
import os

# getting amfe folder
if sys.platform[:3]=='win':
    path_list = os.path.dirname(__file__).split('\\')[:-1]
    amfe_dir = '\\'.join(path_list)
    sys.path.append(amfe_dir)
elif sys.platform[:3]=='lin':
    path_list = os.path.dirname(__file__).split('/')[:-1]
    amfe_dir = '/'.join(path_list)
    sys.path.append(amfe_dir)
else :
    raise('Plataform %s is not supported  ' %sys.platform)
    

from linalg.arnoldi import compute_modes
from contact import Contact, Cyclic_Contact
from utils.utils import OrderedSet, get_dofs
import scipy.sparse as sparse
import scipy.linalg as linalg
import numpy as np
import copy
import unittest
import collections
import logging


#setting logging level
#logging.basicConfig(level=logging.DEBUG)

class SelectionOperator():
    def __init__(self,selection_dict,id_matrix=None):
        ''' the selection dict contain labels as key and 
        dofs as values. The idea is to provide a class
        which can apply permutation in matrix and also global to local map
        
        parameters 
            selection_dict : Ordered dict
                dict with string and dofs
        
        '''
        self.selection_dict = selection_dict
        self.local_indexes = []
        self.bounds = {}
        self.length = {}
        self.local_to_global_dof_dict = {} 
        self.global_to_local_dof_dict = {} 
        self.global_id_matrix = id_matrix
        
        count = 0
        local_dof_counter = 0
        for key, dof_list in selection_dict.items():
            self.local_indexes.extend(dof_list)
            length = len(self.local_indexes)
            self.length[key] = len(dof_list)
            self.bounds[key] = [count,length]
            count += length
            for value in dof_list:
                self.local_to_global_dof_dict[local_dof_counter] = value
                self.global_to_local_dof_dict[value] = local_dof_counter
                local_dof_counter += 1
        
        self.P = self.create_permutation_matrix(self.local_indexes)
        self.ndof = max(self.P.shape)
    
    def nodes_to_local_dofs(self):
        pass
    
    def create_permutation_matrix(self,local_indexes):
        ''' create a Permutation matrix based on local id
        
        '''
        ndof = len(local_indexes)
        P = sparse.lil_matrix((ndof, ndof), dtype=np.int8)
        P[local_indexes, np.arange(ndof)] = 1
        return P.T.tocsc()
        
    def create_block_matrix(self,M):
        ''' This function create block matrix with string
        which is useful for applying boundary conditions
        '''
        block_matrix = {}
        for key1, dofs_1 in self.selection_dict.items():
            for key2, dofs_2 in self.selection_dict.items():
                block_matrix[key1,key2] = M[np.ix_(dofs_1, dofs_2)]
        
        return block_matrix
        
    def assemble_matrix(self,M,list_of_strings):
        ''' This method assemble a matrix based on the list of string
        useful for ordering the matrix according to the block string matrix
        paramenter:
            M : np.array
                matrix to be reordered
            list of strings : list
                list with a sequence of string which gives the 
                order of the degrees of freedom associated with M11
            
            return a ordered Matrix
            
            ex. 
                M_block = s.create_block_matrix(M)
                M_row1 = sparse.hstack((M_block['l','l'],M_block['l','h'],M_block['l','i']))
                M_row2 = sparse.hstack((M_block['h','l'],M_block['h','h'],M_block['h','i']))
                M_row3 = sparse.hstack((M_block['i','l'],M_block['i','h'],M_block['i','i']))
                M_sector = sparse.vstack((M_row1,M_row2,M_row3)).tocsc()
            
            
        '''
        M_block = self.create_block_matrix(M)
        
        M_rows = []
        for s_i in list_of_strings:
            M_row_j_list = [] 
            for s_j in list_of_strings:
                M_row_j_list.append(M_block[s_i,s_j])
            M_rows.append(sparse.hstack(M_row_j_list))
        
        return sparse.vstack(M_rows).tocsc()
            
    def select_block(self,M,rows,columns):
        pass
    
    def build_B(self,label):
        ''' Build Boolean selection operator
        
        '''
        
        
        #local_id = []
        #for node_id in node_list:
        #    global_dof_list = self.global_id_matrix[node_id]
        #    for global_dof in global_dof_list:
        #        local_dof = self.global_to_local_dof_dict[global_dof]
        #        local_id.append(local_dof)
        
        local_id = self.selection_dict[label]
        B = sparse.csc_matrix((len(local_id), self.ndof), dtype=np.int8)
        B[np.arange(len(local_id)), local_id ] = 1
        return B

def apply_cyclic_symmetry(M_block,high,low,interior,beta,theta=0,dimension=3):
    ''' 
    parameters:
        M_block : dict
            block matrix with string indexes
        high : string
            string with the indexes of high dofs
        low : string
            string with the indexes of low dofs
        interior : string
            string with the indexes of interior dofs
        beta : float
            node diameter times the sector angule in rad
    '''
    ej_beta_plus = np.exp(1J*beta)
    ej_beta_minus = np.exp(-1J*beta)
    n_dofs = M_block[low,low].shape[0] 
    #theta=0.0
    logging.debug('Number of Low dofs %i' %n_dofs) 
    logging.debug('Number of High dofs %i' %M_block[high,high].shape[0]) 
    
    T = create_voigt_rotation_matrix(n_dofs, theta, dim=dimension, unit='rad', sparse_matrix = True)
    
    M11 = M_block[high,high] + T.T.dot(M_block[low,low]).dot(T) + ej_beta_plus*M_block[high,low].dot(T) + ej_beta_minus*T.T.dot(M_block[low,high])
    M12 = M_block[high,interior] + ej_beta_minus*T.T.dot(M_block[low,interior])
    M21 = M_block[interior,high] + ej_beta_plus*M_block[interior,low].dot(T)
    M22 = M_block[interior,interior]
    
    M_row1 = sparse.hstack((M11,M12))
    M_row2 = sparse.hstack((M21,M22))
    M = sparse.vstack((M_row1,M_row2))
    return M.tocsc()

def get_unit_rotation_matrix(alpha_rad,dim):  
    
    cos_a = np.cos(alpha_rad)
    sin_a = np.sin(alpha_rad)
    
    if dim==3:
        R_i = np.array([[cos_a,-sin_a,0.0],
                        [sin_a, cos_a,0],
                        [0.0, 0.0, 1.0]])
    elif dim==2:
        R_i = np.array([[cos_a, -sin_a],
                       [sin_a, cos_a]])    
    else:
        raise('Dimension not supported')      
        
    return R_i
    
def rotate_u(u,alpha_rad, dim=2):
    
    R_i = get_unit_rotation_matrix(-alpha_rad,dim)    
    n = len(u)
    u_new = np.reshape(u,(int(n/dim),dim))
    u_r = []
    for u_i in u_new:
       u_r.append(u_i.dot(R_i))
    
    u_r = np.array(u_r)
    return u_r.reshape(-1)

def nullspace(B, tol= 10.e-10):
    ''' Compute the null space of A 
    where A is a complex matrix
    and it has dimention mxn where m<n
    
    parameters
        A : np.matrix
            matrix to compute the null space
    return
        R : np.matrix
            a orthogonal bases of the null space
    
    '''
    
    try:
        BTB = B.conj().T.dot(B).todense()
        convert_to_sparse = True
    except:
        BTB = B.conj().T.dot(B)
        
    w, v_null = linalg.eig(BTB)

    w_real = w.real
    w_imag = w.imag
    nnz = (w_real >= tol).sum()
    R = v_null[:,nnz:]
    
    if convert_to_sparse:
        return sparse.csc_matrix(R)
    
    return R
    
def get_all_dofs(id_matrix):
    dofs = []
    for key, value in id_matrix.items():
        dofs.extend(value)
    return dofs
    
def assemble_cyclic_modes(selector_operator,mode_shape,node_id=0,theta=0,compute_left=True,imag=False,dimension=3):
    
    beta = node_id*theta
    ej_beta_plus = np.exp(1J*beta)
    mode_shape_list = []
    s = selector_operator
    n_modes = mode_shape.shape[1]  
     
    for mode_num in range(n_modes):
        mode = mode_shape[:,mode_num]
        

        ud = np.array([0]*s.length['d'])
        
        if compute_left:
            ur = mode[0:s.length['r']]
            R = create_voigt_rotation_matrix(len(ur), theta, dim=dimension, unit='rad')   
            ui = mode[s.length['r']:]
            ul = R.dot(ej_beta_plus*ur)
            u_m = np.hstack((ud,ur,ul,ui))
        
        else:
            ur = mode[0:s.length['r']]
            ul = mode[s.length['r']:s.length['r'] + s.length['l']]
            ui = mode[s.length['r'] + s.length['l']:]

            
            R = create_voigt_rotation_matrix(len(ur), -theta, dim=dimension, unit='rad')   
            
            
            #u_m = np.hstack((ud,ur,ul,ui))
            u_m = np.hstack((ud,mode))
        
        v = s.P.T.dot(u_m)
        if imag:
            v = v.imag
        
        mode_shape_list.append(v)

    return np.array(mode_shape_list).T
    
def set_cyclic_modes_to_component(my_comp,selector_operator,mode_shape,sector_id=0, node_id=0, theta=0, compute_left=True, unit='rad', dimension=3,**kwargs):


    n_modes = mode_shape.shape[1]
    
    rotation = theta*sector_id
    
    if unit[0:3]=='deg':
        rotation = np.deg2rad(rotation)
        unit = 'rad'
     
    mode_shape_list = assemble_cyclic_modes(selector_operator,mode_shape,node_id=node_id,theta=theta,compute_left=compute_left,imag=False,dimension=dimension)
     
    ej_n_theta = np.exp(1J*sector_id*node_id*theta)
    my_comp = copy.deepcopy(my_comp)
    m_i = my_comp.mesh_class.rot_z(rotation,unit)
    my_comp.set_mesh_obj(m_i)
    my_comp.assembly_class.compute_element_indices()
    
    my_comp.u_output = []
    
    ndofs = len(mode_shape_list[:,0])
    R = create_voigt_rotation_matrix(ndofs , rotation, dim=dimension, unit=unit)
    
    for plot_mode_num in range(n_modes):  
        u_i = mode_shape_list[:,plot_mode_num]
        #v_i = rotate_u(u_i,rotation,dim=dimension)        
        v_i = R.dot(ej_n_theta*u_i)
        my_comp.u_output.append(v_i.real)
    

    return my_comp
    
def create_rotated_component(my_comp,selector_operator,sector_id=0, node_id=0, theta=0, compute_left=True, unit='rad', dimension=3,**kwargs):    
    
    mode_shape_list = my_comp.u_output
    mode_shape_list = np.array(mode_shape_list).T
    n_modes = len(my_comp.u_output)
    
    if unit[0:3]=='deg':
        rotation = np.deg2rad(rotation)
        unit = 'rad'
    
    rotation = theta*sector_id
    
    #ej_n_theta = np.exp(-1J*sector_id*node_id*theta)
    ej_n_theta = np.exp(-1J*sector_id*node_id*theta)
    my_comp = copy.deepcopy(my_comp)
    m_i = my_comp.mesh_class.rot_z(rotation,unit)
    my_comp.set_mesh_obj(m_i)
    my_comp.assembly_class.compute_element_indices()
    
    my_comp.u_output = []
    
    ndofs = len(mode_shape_list[:,0])
    R = create_voigt_rotation_matrix(ndofs , rotation, dim=dimension, unit=unit)
    
    for plot_mode_num in range(n_modes):  
        u_i = mode_shape_list[:,plot_mode_num]
        #v_i = rotate_u(u_i,rotation,dim=dimension)        
        v_i = R.dot(ej_n_theta*u_i)
        my_comp.u_output.append(v_i.real)

            
    return my_comp
    
def create_voigt_rotation_matrix(n_dofs,alpha_rad, dim=2, unit='rad', sparse_matrix = True):
    ''' This function creates voigt rotation matrix, which is a block
    rotation which can be applied to a voigt displacement vector
    ''' 
    
    if n_dofs<=0:
        raise('Error!!! None dof was select to apply rotation.')
    
    if unit[0:3]=='deg':
        rotation = np.deg2rad(rotation)
        unit = 'rad'
        
    R_i = get_unit_rotation_matrix(alpha_rad,dim)  
    
    n_blocks = int(n_dofs/dim)
    
    
    if n_blocks*dim != n_dofs:
        raise('Error!!! Rotation matrix is not matching with dimension and dofs.')
    if sparse_matrix:
        R = sparse.block_diag([R_i]*n_blocks)
    else:
        R = linalg.block_diag(*[R_i]*n_blocks)
    return R
    
class RotationTest(unittest.TestCase):
    def setUp(self):
        self.alpha_rad = np.pi/4.0
        nodes = 10
        self.dim = 3
        self.ndofs = self.dim*nodes
        self.u = np.random.rand(self.ndofs)
        self.maxDiff = 10
    
    def test_rotation_func(self):
        u = self.u
        alpha_rad = self.alpha_rad
        dim = self.dim
        ndofs = self.ndofs
        
        R = create_voigt_rotation_matrix(ndofs , alpha_rad, dim=dim, unit='rad')
        u1 = R.dot(u)
        u2 = rotate_u(u,alpha_rad,dim=dim)
        self.assertSequenceEqual(u1.tolist(), u2.tolist())
    

class Cyclic_Symmetry_Modal_Analysis():

    def __init__(self,my_comp,dirsub,cyclic_left,cyclic_right,sector_angle,unit='rad',tol_radius = 1e-3, node_diam_range = (0,10),n_modes=10):
        
        id_matrix = my_comp.assembly_class.id_matrix
        dimension = my_comp.mesh_class.no_of_dofs_per_node
        # get dofs
        all_dofs = get_all_dofs(id_matrix)
        
        if dimension == 2:
            direction ='xy'
            print('xy direction choosen for cyclic symmetry')
        elif dimension == 3:
            #direction ='xyz'
            direction ='xyz'
            print('xyz direction choosen for cyclic symmetry')
        else:
            raise('Dimension is not supported')
        
        dir_dofs = get_dofs(dirsub, direction =direction, id_matrix=id_matrix)

        # defining sector angle
        # set node diamenter
     
        theta = sector_angle

        # creating node pairs
        contact = Cyclic_Contact(cyclic_left,cyclic_right,sector_angle=theta, unit='rad',tol_radius = tol_radius)


        # modifying order of nodes to have the correct node pairs for cyclic symmetry
        cyclic_left.global_node_list = contact.slaves_nodes
        cyclic_right.global_node_list = contact.master_nodes


        superset = OrderedSet(dir_dofs)
        left_dofs = OrderedSet(get_dofs(cyclic_left, direction = direction, id_matrix=id_matrix)) - superset
        right_dofs = OrderedSet(get_dofs(cyclic_right, direction = direction, id_matrix=id_matrix)) - superset

        boundary_dofs = superset | left_dofs | right_dofs
        interior_dofs = list(OrderedSet(all_dofs) - boundary_dofs)
        left_dofs = list(left_dofs)
        right_dofs = list(right_dofs)

        dof_dict = collections.OrderedDict()
        dof_dict['d'] = dir_dofs 
        dof_dict['r'] = left_dofs 
        dof_dict['l'] = right_dofs
        dof_dict['i'] = interior_dofs

        s = SelectionOperator(dof_dict,id_matrix)

        K, f = my_comp.assembly_class.assemble_k_and_f()
        M = my_comp.assembly_class.assemble_m()
        #plt.matshow(K.todense())

        M_block = s.create_block_matrix(M)
        M_sector = s.assemble_matrix(M,['r','l','i'])

        K_block = s.create_block_matrix(K)
        K_sector = s.assemble_matrix(K,['r','l','i'])
        
        #node_diam = 0
        comp_list = []
        eigval_list = []
        for node_diam in range(*node_diam_range):
            beta = node_diam*theta
            ej_beta_plus = np.exp(1J*beta)
            M_beta = apply_cyclic_symmetry(M_block,'r','l','i',beta,theta=theta,dimension=dimension) 
            K_beta = apply_cyclic_symmetry(K_block,'r','l','i',beta,theta=theta,dimension=dimension) 


            eigval, V = compute_modes(K_beta,M_beta,num_of_modes=n_modes)
            my_comp.u_output = []

            my_comp_copy = copy.deepcopy(my_comp)
            ui = assemble_cyclic_modes(s,V,node_id=node_diam,theta=theta,compute_left=True,imag=False,dimension=dimension)
            my_comp_copy.u_output = list(ui.T)
            comp_list.append(my_comp_copy)
            eigval_list.append(eigval)
            
        self.comp_ND_list = comp_list
        self.selector_operator = s
        self.eigval_list = eigval_list

def cyclic_symmetry_modal_analysis(my_comp,dirsub,cyclic_left,cyclic_right,sector_angle,unit='rad',tol_radius = 1e-3, node_diam_range = (0,10),n_modes=10):

    id_matrix = my_comp.assembly_class.id_matrix
    dimension = my_comp.mesh_class.no_of_dofs_per_node
    # get dofs
    all_dofs = get_all_dofs(id_matrix)
    
    if dimension == 2:
        direction ='xy'
        print('xy direction choosen for cyclic symmetry')
    elif dimension == 3:
        direction ='xyz'
        print('xyz direction choosen for cyclic symmetry')
    else:
        raise('Dimension is not supported')
    
    dir_dofs = get_dofs(dirsub, direction =direction, id_matrix=id_matrix)

    # defining sector angle
    # set node diamenter
 
    theta = sector_angle

    # creating node pairs
    contact = Cyclic_Contact(cyclic_left,cyclic_right,sector_angle=theta, unit='rad',tol_radius = tol_radius)


    # modifying order of nodes to have the correct node pairs for cyclic symmetry
    cyclic_left.global_node_list = contact.slaves_nodes
    cyclic_right.global_node_list = contact.master_nodes


    superset = OrderedSet(dir_dofs)
    left_dofs = OrderedSet(get_dofs(cyclic_left, direction =direction, id_matrix=id_matrix)) - superset
    right_dofs = OrderedSet(get_dofs(cyclic_right, direction =direction, id_matrix=id_matrix)) - superset

    boundary_dofs = superset | left_dofs | right_dofs
    interior_dofs = list(OrderedSet(all_dofs) - boundary_dofs)
    left_dofs = list(left_dofs)
    right_dofs = list(right_dofs)

    dof_dict = collections.OrderedDict()
    dof_dict['d'] = dir_dofs 
    dof_dict['r'] = left_dofs 
    dof_dict['l'] = right_dofs
    dof_dict['i'] = interior_dofs

    s = SelectionOperator(dof_dict,id_matrix)

    K, f = my_comp.assembly_class.assemble_k_and_f()
    M = my_comp.assembly_class.assemble_m()
    #plt.matshow(K.todense())

    M_block = s.create_block_matrix(M)
    M_sector = s.assemble_matrix(M,['r','l','i'])

    K_block = s.create_block_matrix(K)
    K_sector = s.assemble_matrix(K,['r','l','i'])
    
    #node_diam = 0
    comp_list = []
    for node_diam in range(*node_diam_range):
        beta = node_diam*theta
        ej_beta_plus = np.exp(1J*beta)
        M_beta = apply_cyclic_symmetry(M_block,'r','l','i',beta,theta=theta,dimension=dimension) 
        K_beta = apply_cyclic_symmetry(K_block,'r','l','i',beta,theta=theta,dimension=dimension) 


        eigval, V = compute_modes(K_beta,M_beta,num_of_modes=n_modes)

        my_comp_copy = copy.deepcopy(my_comp)
        my_comp.u_output = []
        ui = assemble_cyclic_modes(s,V,node_id=node_diam,theta=theta,compute_left=True,imag=False,dimension=dimension)
        my_comp_copy.u_output = list(ui.T)
        comp_list.append(my_comp_copy)
    return comp_list
    

class Cyclic_Component():
    def __init__(self,my_comp,
                      dirsub,
                      cyclic_left,
                      cyclic_right,
                      sector_angle,
                      unit='rad',
                      tol_radius = 1.0e-3):
    
        id_matrix = my_comp.assembly_class.id_matrix
        dimension = my_comp.mesh_class.no_of_dofs_per_node
        self.component = copy.deepcopy(my_comp)
        
        # get dofs
        all_dofs = get_all_dofs(id_matrix)
        
        if dimension == 2:
            direction ='xy'
            print('xy direction choosen for cyclic symmetry')
        elif dimension == 3:
            direction ='xyz'
            print('xyz direction choosen for cyclic symmetry')
        else:
            raise('Dimension is not supported')
        
        dir_dofs = get_dofs(dirsub, direction =direction, id_matrix=id_matrix)

        # defining sector angle
        # set node diamenter
     
        theta = sector_angle

        # creating node pairs
        contact = Cyclic_Contact(cyclic_left,cyclic_right,sector_angle=theta, unit='rad',tol_radius = tol_radius)


        # modifying order of nodes to have the correct node pairs for cyclic symmetry
        cyclic_left.global_node_list = contact.slaves_nodes
        cyclic_right.global_node_list = contact.master_nodes


        superset = OrderedSet(dir_dofs)
        left_dofs = OrderedSet(get_dofs(cyclic_left, direction =direction, id_matrix=id_matrix)) - superset
        right_dofs = OrderedSet(get_dofs(cyclic_right, direction =direction, id_matrix=id_matrix)) - superset

        boundary_dofs = superset | left_dofs | right_dofs
        interior_dofs = list(OrderedSet(all_dofs) - boundary_dofs)
        left_dofs = list(left_dofs)
        right_dofs = list(right_dofs)

        dof_dict = collections.OrderedDict()
        dof_dict['d'] = dir_dofs 
        dof_dict['r'] = right_dofs
        dof_dict['l'] = left_dofs 
        dof_dict['i'] = interior_dofs

        self.dimension = dimension
        self.nc = len(left_dofs)
        self.ndofs = 2*len(left_dofs) + len(interior_dofs)
        self.s = SelectionOperator(dof_dict,id_matrix)
        self.theta = theta
        
        self.id_matrix = id_matrix
    
    def assemble_sector_operators(self):
    
        K, f = self.component.assembly_class.assemble_k_and_f()
        M = self.component.assembly_class.assemble_m()


        self.M_block = self.s.create_block_matrix(M)
        self.M_sector = self.s.assemble_matrix(M,['r','l','i'])

        K_block = self.s.create_block_matrix(K)
        self.K_sector = self.s.assemble_matrix(K,['r','l','i'])
        
        self.K =K
        self.M = M
        return self.K_sector, self.M_sector
        
    def build_complex_contraint(self,node_diam):
    
    
        s = self.s 
        id_matrix = self.id_matrix
        theta = self.theta
        
        dof_dict2 = collections.OrderedDict()
        dof_dict2['r'] = list(range(s.length['r']))
        dof_dict2['l'] = list(range(dof_dict2['r'][-1]+1,dof_dict2['r'][-1] + s.length['l']+1))
        dof_dict2['i'] = list(range(dof_dict2['l'][-1]+1,dof_dict2['l'][-1] + s.length['i']+1))


        # createing selection operator
        s2 = SelectionOperator(dof_dict2,id_matrix)

        # building cyclic matrices
        #theta = -theta
        beta = node_diam*theta
        ej_beta_plus = np.exp(1J*beta)

        #building Boolean matrices
        Bl = s2.build_B('l')
        Br = s2.build_B('r')

        T = create_voigt_rotation_matrix(self.nc, theta, dim=self.dimension)

        # Building the cyclic constraint
        #C_n = T.dot(Bl) - ej_beta_plus*Br
        #C_n = -T.dot(Br) + ej_beta_plus*Bl
        C_n =  - ej_beta_plus*Br  + T.dot(Bl) 
       
        
        self.Bl = Bl
        self.Br = Br
        
        return C_n
        
    def build_complex_projection(self,node_diam):
        
        nc = self.nc
        ndofs = self.ndofs
        C_n = self.build_complex_contraint(node_diam)
        
        P_n = sparse.eye(ndofs) - 0.5*C_n.conj().T.dot(C_n)
        return P_n
        
    def build_contraint_null_space(self,node_diam):
        
        theta = self.theta
        Bl = self.Bl 
        Br = self.Br
        nc = self.nc
        ndofs = self.n_dofs
        nr = ndofs - nc 
        
        beta = node_diam*theta
        ej_beta_plus = np.exp(1J*beta)
        
        R_col1 = (ej_beta_plus.conj()*Br  + T.dot(Bl)).T
        R_col2 = sparse.vstack([0*sparse.eye(2*nc,nr-nc).tocsc(), sparse.eye(ndofs-2*nc).tocsc()]).tocsc()
        R = sparse.hstack([R_col1,R_col2]).tocsc()
        return R
        
if __name__ == '__main__':
    #rt = RotationTest()
    #rt.setUp()
    unittest.main()
