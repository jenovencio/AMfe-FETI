# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 10:26:53 2018

@author: ge72tih
"""

import scipy.sparse as sparse
import scipy.linalg as linalg
import numpy as np
import copy
import unittest

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
        P = sparse.csc_matrix((ndof, ndof), dtype=np.int8)
        P[local_indexes, np.arange(ndof)] = 1
        return P.T
        
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
    
def get_dofs(id_matrix):
    dofs = []
    for key, value in id_matrix.items():
        dofs.extend(value)
    return dofs
    
def assemble_cyclic_modes(selector_operator,mode_shape,beta=0,compute_left=False,imag=False,theta=0,dimension=3):
    
    ej_beta_plus = np.exp(1J*beta)
    v_primal_diri = []
    s = selector_operator
    n_modes = mode_shape.shape[1]  
     
    for mode_num in range(n_modes):
        mode = mode_shape[:,mode_num]
        

        ud = np.array([0]*s.length['d'])
        
        if compute_left:
            uh = mode[0:s.length['h']]
            R = create_voigt_rotation_matrix(len(uh), theta, dim=dimension, unit='rad')   
            ui = mode[s.length['h']:]
            if imag:
                ul = uh
                uh = R.dot(uh)
                
                #R.dot(uh.real)
                #ul = uh.real
                u_m = np.hstack((ud,ul.imag,uh.real,ui.imag))
            else:
                #ul = R.dot(ej_beta_plus*(uh).real)
                #ul = ej_beta_plus*uh
                ul = uh
                #uh = R.dot(ej_beta_plus*(uh).real)
                if beta==0:
                    uh = R.dot(uh)
                else:    
                    uh = R.dot(ej_beta_plus*ul)
                u_m = np.hstack((ud,ul.real,uh.real,ui.real))
        else:
            if imag:
                u_m = np.hstack((ud,mode.imag))
            else:
                u_m = np.hstack((ud,mode.real))
        
        v = s.P.T.dot(u_m)
        v_primal_diri.append(v)


    v_primal_diri = np.array(v_primal_diri).T
    return v_primal_diri
    
def set_cyclic_modes_to_component(my_comp,selector_operator,mode_shape,beta=0,compute_left=False, rotation=0, theta=0, unit='rad', dimension=3,**kwargs):

    
    n_modes = mode_shape.shape[1]    
    
    
    if unit[0:3]=='deg':
        rotation = np.deg2rad(rotation)
        unit = 'rad'
     
    v_primal_diri = assemble_cyclic_modes(selector_operator,mode_shape,beta,compute_left,theta=theta,dimension=dimension,**kwargs)
     
    if rotation>0:
        my_comp = copy.deepcopy(my_comp)
        m_i = my_comp.mesh_class.rot_z(rotation,unit)
        my_comp.set_mesh_obj(m_i)
        my_comp.assembly_class.compute_element_indices()
        
        my_comp.u_output = []
        my_comp.u_output.append(v_primal_diri[:,0]*0.0)

        ndofs = len(v_primal_diri[:,0])
        R = create_voigt_rotation_matrix(ndofs , rotation, dim=dimension, unit=unit)
        
        for plot_mode_num in range(n_modes):  
            u_i = v_primal_diri[:,plot_mode_num]
            #v_i = rotate_u(u_i,rotation,dim=dimension)        
            v_i = R.dot(u_i)
            my_comp.u_output.append(v_i)
    
    else:
        my_comp.u_output = []
        my_comp.u_output.append(v_primal_diri[:,0]*0.0)

        for plot_mode_num in range(n_modes):       
            my_comp.u_output.append(v_primal_diri[:,plot_mode_num])
            
    return my_comp
    
def create_voigt_rotation_matrix(n_dofs,alpha_rad, dim=2, unit='rad', sparse_matrix = True):
    ''' This function creates voigt rotation matrix, which is a block
    rotation which can be applied to a voigt displacement vector
    ''' 
    
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
    

        
if __name__ == '__main__':
    #rt = RotationTest()
    #rt.setUp()
    unittest.main()
