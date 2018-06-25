# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 09:10:00 2018

@author: ge72tih
"""

import numpy as np
import scipy.sparse as sparse
import scipy
import sys, os
import copy
from .amna import *
from .assembly import Assembly
from .mesh import Mesh
import dill as pickle
import subprocess
from .tools import *
import logging
import pandas as pd

#logging.basicConfig(level=logging.DEBUG)

gmsh2amfe_elem_dict = {}
gmsh2amfe_elem_dict[4] = 'Tet4'
gmsh2amfe_elem_dict[11] = 'Tet10'
gmsh2amfe_elem_dict[5] = 'Hexa8'
gmsh2amfe_elem_dict[17] = 'Hexa20'
gmsh2amfe_elem_dict[9] = 'Tri6'
gmsh2amfe_elem_dict[2] = 'Tri3'
gmsh2amfe_elem_dict[21] = 'Tri10'
gmsh2amfe_elem_dict[3] = 'Quad4'
gmsh2amfe_elem_dict[16] = 'Quad8'
gmsh2amfe_elem_dict[6] = 'Prism6'
gmsh2amfe_elem_dict[1] = 'straight_line'
gmsh2amfe_elem_dict[8] = 'quadratic_line'
gmsh2amfe_elem_dict[15] = 'point'
    
elem_dof = {}
elem_dof['Tet4'] = 3
elem_dof['Tet10'] = 3
elem_dof['Hexa8'] = 3
elem_dof['Hexa20'] = 3
elem_dof['Tri6'] = 2
elem_dof['Tri3'] = 2
elem_dof['Tri10'] = 2
elem_dof['Quad4'] = 2
elem_dof['Quad8'] = 2
elem_dof['Prism6'] = 3
elem_dof['straight_line'] = 2
elem_dof['quadratic_line'] = 2
elem_dof['point'] = 1

dirichlet_dict = {}
dirichlet_dict['x'] = [0]
dirichlet_dict['y'] = [1]
dirichlet_dict['z'] = [2]
dirichlet_dict['xy'] = [0,1]
dirichlet_dict['xz'] = [0,2]
dirichlet_dict['yz'] = [1,2]
dirichlet_dict['xyz'] = [0,1,2]


def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def load_obj(filename):
    with open(filename, 'rb') as input:
        obj = pickle.load(input)
    return obj


# geting path of MPI executable
mpi_exec = 'mpiexec'
try:
    mpi_path = os.environ['MPIDIR']
    mpi_exec = os.path.join(mpi_path, mpi_exec).replace('"','')
except:
    print("Warning! Using mpiexec in global path")
    mpi_path = None
    

try:
    python_path = os.environ['AMFE_PYTHON']
    python_exec = os.path.join(python_path,'python').replace('"','')
except:
    print("Warning! Using python in global path")
    python_path = None
    python_exec = 'python'

class FetiSolver():
    
    def linear_static(my_system,log = False, directory='temp'):
        
        domain = my_system.domain
        domain.split_in_partitions()
        problem_type = domain.problem_type
        num_partitions = len(domain.groups)
        partitions_list = np.arange(1,num_partitions+1)
    
        # saving object to pass to MPIfetisolver
        
        #directory = 'temp'
        filename = 'system.aft'
        run_file_path = 'run_mpi_solver.bat'
        file_path = os.path.join(directory,filename)
        try:
            os.stat(directory)
        except:
            os.mkdir(directory)       
        
        
        save_object(my_system, file_path)
        python_solver_file = amfe_dir('amfe\MPIfetisolver.py')

        logging.info('######################################################################')
        logging.info('###################### SOLVER INFO ###################################')
        logging.info('MPI exec path = %s' %mpi_exec )
        logging.info('Python exec path = %s' %python_exec )

        command = '"' + mpi_exec + '" -l -n ' + str(num_partitions+1) + ' "' + python_exec + '"  "' + \
                  python_solver_file + '" ' + filename 
        
        # export results to a log file called amfeti_solver.log
        if log:
            command += '>amfeti_solver.log'
        
        

        # writing bat file with the command line
        local_folder = os.getcwd()
        os.chdir(directory)
        run_file = open(run_file_path,'w')
        run_file.write(command)
        run_file.close()

        logging.info('Run directory = %s' %os.getcwd())
        logging.info('######################################################################')

        # executing bat file
        try:    
            subprocess.call(run_file_path)
            os.chdir(local_folder)
            
        except:
            os.chdir(local_folder)
            logging.error('Error during the simulation.')
            return None

        
        # loading results from subdomain *.are file
        subdomains_dict = {}
        for i in partitions_list:
            try:
                res_path = os.path.join(directory, str(i) + '.are')
                sub_i = load_obj(res_path)
                subdomains_dict[i] = sub_i
            except:
                logging.warning('WARNING! It was not possible to read %i.are file.\n' + \
                'Make sure you have python and mpiexec installed.' %i)
                return None
        
        # loading solver class
        sol_path = os.path.join(directory,'solver.sol')
        sol = load_obj(sol_path)


        subdomains_dict, sol = FetiSolver.reading_results(partitions_list,directory)

        
        # append feti subdomains to system instance
        my_system.feti_subdomain_dict = subdomains_dict
        
        # calculating average displamecent of subdomain
        avg_displacement = FetiSolver.average_displacement_calc(my_system,subdomains_dict)
        
        # updating system displacement
        total_dof = my_system.assembly_class.mesh.no_of_dofs
        my_system.u_output.append(np.zeros(total_dof))
        
        my_system.u_output.append(avg_displacement)
        
        return sol.residual
    
    def reading_results(partitions_list,directory,solver_file = 'solver.sol' ):
        ''' This method loads results from subdomain *.are files

        argument:
            partitions_list : list
                partitions list to be read

            directory : str
                path are the *.are are stored

        return 
            subdomains_dict : dict
                dictionary with FETIsubdomains objecties
            sol : ParallelSolver obj
                object containing solver informations
        '''
        subdomains_dict = {}
        for i in partitions_list:
            try:
                res_path = os.path.join(directory, str(i) + '.are')
                sub_i = load_obj(res_path)
                subdomains_dict[i] = sub_i
            except:
                logging.warning('WARNING! It was not possible to read %i .are file.\n' + \
                'Make sure you have python and mpiexec installed.' %i)
                return None
        
        # loading solver class
        sol_path = os.path.join(directory,solver_file)
        sol = load_obj(sol_path)
        return subdomains_dict, sol

    def average_displacement_calc(my_system,subdomains_dict):
        ''' This function calculates the average displacement of the whole domain
        based on the displacement of local subdomains
        
        Arguments:
            system: mechanical_system instance
                system intance with case information
            subdomains_dict: dict
                dictionary with feti_submains
        
        Returns 
            avg displacement with the size of the total degrees of freedom
        
        '''
        total_dof = my_system.assembly_class.mesh.no_of_dofs
        displacement = np.zeros(total_dof )
        div = np.zeros(total_dof)

        for sub_key, sub in subdomains_dict.items():
            for i,node in enumerate(sub.submesh.global_node_list):
                local_node = sub.submesh.global_to_local_dict[node]
                global_dofs = my_system.assembly_class.id_matrix[node]
                local_dofs = sub.id_matrix[local_node]
                local_disp = sub.displacement[local_dofs]
                #displacement[global_dofs] += local_disp.T[0]
                displacement[global_dofs] += np.array(local_disp).flatten()
                div[global_dofs] += np.ones(len(global_dofs)) 

        displacement = displacement/div
        return displacement

        
        
class FETIsubdomain(Assembly):
    def __init__(self,submesh_obj):
        
        
        self.submesh = submesh_obj
        self.elem_start_index = self.submesh.parent_mesh.node_idx
        self.elem_last_index = len(self.submesh.parent_mesh.el_df.columns)
        self.key = self.submesh.key
        self.global_elem_list = self.submesh.elements_list
        self.neighbor_partitions = self.submesh.neighbor_partitions
        self.calc_local_indices()
        amfe_mesh = self.__set_amfe_mesh__()
        
        # internal variables
        self.total_dof = amfe_mesh.no_of_dofs
        self.displacement = np.zeros(self.total_dof)
        
        # internal operators variables
        self.stiffness = None
        self.total_force = None
        self.dual_force_dict = {}
        self.G_dict = {}
        self.B_dict = {}
        self.neighbor_G_dict = {}
        self.GtG_rows_dict = {}
        
        # null space variables
        self.null_space_force = None
        self.zero_pivot_indexes = []
        self.null_space_size = 0
        self.null_space = None
        
        super(FETIsubdomain, self).__init__(amfe_mesh)
        
        self.__find_shared_nodes__()        
        
        self.preallocate_csr()
        self.compute_element_indices()
        
        
        # internal decomposition variables
        self.solver_opt = 'cholsps'
        self.cholesky_tolerance = 1.0E-6
        self.compute_cholesky_boolean = False
        
        # pseudo inverse variables
        self.pinv = amna.P_inverse()
        self.pinv_tolerance = 1.0E-6
        self.pinv.set_solver(self.solver_opt)
        self.pinv.tolerance(self.pinv_tolerance)
        self.compute_pinv_boolean = False
        
        self.local_interface_nodes_dict = {}
        self.lambda_global_indices = []
        self.local_interface_dofs_dict = {}
        self.local_interface_dofs_list = []
        self.local_interior_dofs_list = []
        self.num_of_interface_dof_dict = {}
        self.num_of_interface_dof = 0
        
        self.create_interface_and_interior_dof_dicts()
        
    def set_cholesky_tolerance(self, tolerance=1.0E-6):
        ''' set cholesky tolerance
        
        argument    
            tolerance : float
        '''
        self.cholesky_tolerance = tolerance      

    def set_solver_option(self,solver_str='cholsps'):
        ''' This function set the solver option 
        for compute internal operator and solve linear systems
        '''
        self.solver_opt = solver_str
        return self.solver_opt
    
    def calc_local_indices(self):        
    
        node_list = self.submesh.parent_mesh.nodes
        connectivity = self.submesh.elem_dataframe.iloc[:,self.elem_start_index:self.elem_last_index]
        connectivity = connectivity.astype(int)
        
        local_connectivity = []
        local_node_list = []
        global_to_local_dict = {}
        local_to_global_dict = {}
        elem_key = 0
        i = 0
        for df_key, elem_connect in connectivity.iterrows():

            elem_type = self.submesh.elem_dataframe['el_type'].iloc[elem_key]
            dof = elem_dof[elem_type]
            
            # mapping global nodes to subdomain local nodes
            # and generate connectivity with local_nodes_id        
            local_elem_connect = []
            for global_node_id in elem_connect:
                if not(global_node_id in global_to_local_dict):
                    global_to_local_dict[global_node_id] = i
                    local_to_global_dict[i] = global_node_id  
                    local_node_list.append(node_list[global_node_id][:dof])    
                    i += 1
                local_elem_connect.append(global_to_local_dict[global_node_id])
                    
                            
            local_connectivity.append(np.array(local_elem_connect))
            elem_key += 1
            
        local_connectivity = np.array(local_connectivity)
        local_node_list = np.array(local_node_list)
        
        # updating subdomain local information
        self.submesh.add_local_mesh(local_connectivity, 
                                    local_node_list, 
                                    global_to_local_dict,
                                    local_to_global_dict)
        
        self.global_interface_nodes_dict = self.submesh.interface_nodes_dict
    
    def __set_amfe_mesh__(self):
        amfe_mesh = Mesh()
        
        amfe_mesh.nodes = self.submesh.local_node_list

                        
        # pass mesh connectivity to AMfe connectivity
        amfe_mesh.connectivity = self.submesh.local_connectivity
        
        
        # create an element object and assign a material to ech        
        my_material = self.submesh.__material__
        
        # assign a material for every element
        object_series = []
        for elem_key in self.submesh.elements_list:
            #elem_gmsh_key = self.submesh.parent_mesh.elements_type_dict[elem_key]
            elem_type = self.submesh.elem_dataframe['el_type'].loc[elem_key]
            amfe_mesh.no_of_dofs_per_node = elem_dof[elem_type]
            elem_obj = copy.deepcopy(amfe_mesh.element_class_dict[elem_type])
            elem_obj.material = my_material
            object_series.append(elem_obj)        
        
        amfe_mesh.ele_obj.extend(object_series)
        amfe_mesh._update_mesh_props()        
        
        

        return amfe_mesh
    
    def __find_shared_nodes__(self):
        
        # shared nodes for neumann boundary contition
        self.neumann_nodes = []
        #elem_start_index = self.submesh.parent_mesh.node_idx
        #elem_last_index = len(self.submesh.elem_dataframe.columns)
        elem_connec = self.submesh.parent_mesh.el_df.iloc[:,self.elem_start_index:self.elem_last_index]
        #elem_connec = elem_connec.dropna(1) # removing columns with NaN
        #elem_connec = elem_connec.astype(int) # converting all to int
        
        for sub_obj in self.submesh.neumann_submesh:
            for i,elem_key in enumerate(sub_obj.submesh.elements_list):
                local_connectivity = []
                bool_elem = True
                for node_id in elem_connec.loc[elem_key].dropna().astype(int):
                    if node_id in self.submesh.global_node_list:
                        # map global connectivity to local connectivity
                        local_node_id = self.submesh.global_to_local_dict[node_id]
                        local_connectivity.append(local_node_id)
                    else:
                        bool_elem = False 
                        break
                
                if len(local_connectivity)>1 and bool_elem:   
                    self.mesh.neumann_connectivity.append(np.array(local_connectivity))
                    self.mesh.neumann_obj.extend([sub_obj.neumann_obj[i]])
    
    def create_interface_and_interior_dof_dicts(self):
        ''' This function read the subdomain information and creates dictionaries
            with local interface nodes, local interface dof,
            local interior dofs and list with global lambda indexes
            also compute the total number of interface dofs
            and the number of dofs per neighbor
            
            update instance variables
            self.local_interface_nodes_dict as dict
            self.local_interface_dofs_dict as dict
            self.local_interior_dofs_dict as dict
            self.num_of_interface_dof_dict as dict
            self.lambda_global_indices as list
            self.num_of_interface_dof as int
        '''
        total_dof = self.total_dof
        all_dofs = set(np.arange(total_dof))
        node_set = set()
        
        for neighbor_subdomain_key in self.submesh.interface_nodes_dict:
            num_interface_nodes = len(self.submesh.interface_nodes_dict[neighbor_subdomain_key])

            self.local_interface_nodes_dict[neighbor_subdomain_key] = []
            self.local_interface_dofs_dict[neighbor_subdomain_key] = []
            self.num_of_interface_dof_dict[neighbor_subdomain_key] = 0
            for node_id in self.submesh.interface_nodes_dict[neighbor_subdomain_key]:
                # mapping global dof to local dofs
                local_node_id = self.submesh.global_to_local_dict[node_id]
                self.local_interface_nodes_dict[neighbor_subdomain_key].append(local_node_id)
                node_dof_list = self.id_matrix[local_node_id]
                self.local_interface_dofs_dict[neighbor_subdomain_key].extend(node_dof_list)
                self.num_of_interface_dof_dict[neighbor_subdomain_key] += len(node_dof_list)
                
                if node_id not in node_set:
                    node_set.add(node_id)
                    self.local_interface_dofs_list.extend(node_dof_list)        
                    # create map from local dofs to global dofs
                    init_index = node_id*self.submesh.problem_type
                    last_index = init_index + len(node_dof_list)
                    lambda_indice = list(range(init_index,last_index))
                    self.lambda_global_indices.extend(lambda_indice)
                
            self.num_of_interface_dof += self.num_of_interface_dof_dict[neighbor_subdomain_key]
            
        self.local_interior_dofs_list = list(all_dofs.difference(self.local_interface_dofs_list))
        
        return None

    def assemble_interface_boolean_matrix(self):
        ''' This function computes dictionaries of
        Boolean Matrix per every neighbor
        
        return
            B_dict : dict
                Dictionary with Boolean matrices
        
        '''

        total_dof = self.total_dof

        B_dict = self.B_dict
        num_of_neighbor = 0
        for neighbor_key in self.local_interface_dofs_dict:
            bool_sign = np.sign(neighbor_key - self.key)
            total_int_dof = self.num_of_interface_dof_dict[neighbor_key]
            data = bool_sign*np.ones(total_int_dof,dtype=int)
            columns = self.local_interface_dofs_dict[neighbor_key]
            rows = list(range(total_int_dof))
            B_i = sparse.coo_matrix((data,(rows,columns)),shape=(total_int_dof,total_dof)) 
            B_dict[self.key,neighbor_key] = B_i

        return self.B_dict   
                
    def assemble_K_and_total_force(self):            
        ''' This function assemble the stiffness matrix
        and the total force vector fint+fext
        
        also update internal variables self.stiffness and 
        self.force 
        
        return 
            K : np.matrix
                stiffness matrix operator
            total_force : np.array
                total force fint + fext
        '''
        
        K, fext = self.assemble_k_and_f_neumann()
        K, fint = self.assemble_k_and_f()
        # getting force vector
        fexti = self.force.copy()
        finti = self.internal_force.copy()
        force = fexti + finti
        self.total_force = fexti + finti
        return K, self.total_force
    
    def solve_local_displacement(self, force=None, global_lambda=None, lambda_dict={}):
        ''' Solve local displacement given a gloval lambda vector
        and the indexation matrix that maps global to local dofs
        
        arguments
            global_lambda : np.array
                global lambda
                
            lambda_dict : dict
                dictionary which maps global lambda to
                local lambda
                
            see set_solver_option to modify solver type
                solve_opt : str
                    solver option default = 'cholsps'
        
        return 
            displacement : np.array
                displacement of the subdomain
        '''
        
        solver_opt = self.solver_opt
        
        if force is None:
            force = self.force
        
        # convert force to the right format
        if force.shape != (self.total_dof,):
            force = np.array(force).flatten()
            
        # convert global_lambda to the right format
        if len(global_lambda.shape) > 1:
            global_lambda = np.array(global_lambda).flatten()

            
        sub_id = self.submesh.key
        b_bar = np.zeros(self.total_dof)
        for nei_id in self.submesh.neighbor_partitions:
            if sub_id<nei_id:
                lambda_id = (sub_id,nei_id)
            else:    
                lambda_id = (nei_id,sub_id)
                       
            local_id = lambda_dict[lambda_id]
            Bi = self.B_dict[sub_id,nei_id]   
            local_lambda = global_lambda[local_id]
            
            b_hati = Bi.T.dot(local_lambda)
            b_bar += b_hati
                
        # solving K(s)u_bar(s) = f - b(s) in order to calculte the action of F(s)
        b = force - b_bar
        
        if solver_opt=='cholsps':
            if self.compute_cholesky_boolean:
                Ui = self.full_rank_upper_cholesky.todense()
                idf = self.zero_pivot_indexes
            else:
                Ui,idf,R = self.compute_cholesky_decomposition()

            
            b[idf] = 0.0
            u_bar = scipy.linalg.cho_solve((Ui,False),b)  
        
        elif solver_opt=='svd':
            try:
               Kinv = self.psedoinverse
            
            except:
                Kinv,R = self.calc_pinv_and_null_space('svd')
            
            u_bar = Kinv.dot(b)
            
        else:
            print('Solve option not implement yet')
        
        self.u_bar = np.matrix(u_bar).T
        self.displacement = np.array(u_bar).flatten()
        return self.displacement
        
    def apply_rigid_body_correction(self,global_alpha, alpha_dict):
        
        sub_id = self.submesh.key
        try:
            u_bar = self.displacement
        except:
            print('No displacement is defined, then no corretion can be applied. \n' \
                  'please call solve_local_displacement method before rigig correction')

        if self.null_space_size>0:
            R = self.null_space
            local_id = alpha_dict[sub_id]
            local_alpha =  global_alpha[local_id]
            u_bar += np.array(R.dot(local_alpha)).flatten()
        
        self.displacement = u_bar
        return self.displacement

    def compute_cholesky_decomposition(self):
        ''' Compute cholesky decomposition
        '''
        # without boundary conditions
        K, total_force = self.assemble_K_and_total_force()

        # with boundary conditions
        K, total_force = self.insert_dirichlet_boundary_cond(K,total_force)
        tol = self.cholesky_tolerance
        U, idf, R = cholsps(K,tol)            

        # store cholesky with free pivots = 0
        self.upper_cholesky = U

        if idf:
            # set free pivots to zero
            
            Ui = U.copy()
            Ui[idf,:] = 0.0
            Ui[:,idf] = 0.0
            Ui[idf,idf] = 1.0
            #self.total_force[idf] = 0.0
            self.zero_pivot_indexes = idf
            self.full_rank_upper_cholesky = scipy.sparse.csr_matrix(Ui)
    
        else:
            self.full_rank_upper_cholesky = scipy.sparse.csr_matrix(U)
            Ui = U
    
        self.compute_cholesky_boolean = True
        return Ui,idf,R
        
    def calc_null_space(self):
        ''' compute null space null(K)
        
        '''
        # getting local solver from obj variable
        solver_opt = self.solver_opt
        if solver_opt=='cholsps':
            Ui,idf,R = self.compute_cholesky_decomposition()
            
        elif solver_opt=='svd': 
            Kinv,R = self.calc_pinv_and_null_space()
          
        else:
            print('Not implemented')
            return mp.matrix([])
        
        self.set_null_space(R)
        return R

    def set_null_space(self,R):
        ''' This function sets the subdomain null_space
        
        
        arguments
            R : np.matrix
                matrix with subdomain null space
            
            return None
        '''
        
        if R is not None:
            try:
            
                force = np.array(self.total_force).flatten()
                self.null_space_force = -R.T.dot(force)
                
            except:
                pass
        else:
            R = np.matrix([])
        
        self.null_space_size  = R.shape[1]
        self.null_space = R
        
        return R
        
    def get_null_space(self):
        ''' this function return
        subdomain null space
        '''
        return self.null_space
        
    def calc_dual_force_dict(self):
        ''' Compute local dual force 
        which is equivalent to a displacement 
        in the interface
        
        K_i u_i = f_i
        
        d_{(i,j)} = B_{(i,j)}*u_i
        '''
        
        solver_opt = self.solver_opt
        
        # getting force vector
        force = self.total_force
        if force.shape != (self.total_dof,):
            force = np.array(force).flatten()
        
        if solver_opt=='cholsps':
            if self.compute_cholesky_boolean:
                Ui = self.full_rank_upper_cholesky.todense()
            else:
                Ui,idf,R = self.compute_cholesky_decomposition()
            
            # calculate the dual force B*Kpinv*f            
            u_hat = scipy.linalg.cho_solve((Ui,False),force)

            for (sub_id,nei_id) in self.B_dict: 
                Bi = self.B_dict[sub_id,nei_id]
                self.dual_force_dict[sub_id,nei_id] = Bi.dot(u_hat)

        else:
            logging.error('Solver type not implemented')
            return None
            
        return self.dual_force_dict
            
    def calc_pinv_and_null_space(self,solver_opt='svd',tol=1.0E-8):
        
        solver_opt = self.solver_opt
        tol = self.pinv_tolerance
        if solver_opt=='svd':
            # without boundary conditions
            K, total_force = self.assemble_K_and_total_force()

            # with boundary conditions
            K, total_force = self.insert_dirichlet_boundary_cond(K,total_force)
            K = K.todense()
            V,val,U = np.linalg.svd(K)
            
            total_var = np.sum(val)
            
            norm_eigval = val/val[0]
            idx = [i for i,val in enumerate(norm_eigval) if val>tol]
            val = val[idx]
            
            
            invval = 1.0/val[idx]

            subV = V[:,idx]
            
            Kinv =  np.matmul( subV,np.matmul(np.diag(invval),subV.T))
            
            last_idx = idx[-1]
            R = V[:,last_idx+1:]
            self.psedoinverse = Kinv
            #self.null_space = R 
            #self.null_space_size  = R.shape[1]
        else:
            print('Solver option not implemented')
            return None
        
        return Kinv,R
        
    def insert_dirichlet_boundary_cond(self,K=None,f=None):
        
        if K is None:
            K = self.stiffness
        
        if f is None:
            f = self.force
            
        self.dirichlet_dof = []
        dirichlet_stiffness = 1.0E10
                
        for sub_obj_dir in self.submesh.dirichlet_submesh:
            if sub_obj_dir.value == 0.0:
                # modify K and fext
                elem_start_index = sub_obj_dir.submesh.parent_mesh.node_idx
                elem_last_index = len(sub_obj_dir.submesh.elem_dataframe.columns)
                elem_connec = sub_obj_dir.submesh.elem_dataframe.iloc[:,elem_start_index:elem_last_index]
                elem_connec = elem_connec.dropna(1) # removing columns with NaN
                elem_connec = elem_connec.astype(int) # convert all to int
            
                
                for i, elem in elem_connec.iterrows():
                    local_connectivity = []
                    for global_node_id in elem:                     
                        if global_node_id in self.submesh.global_node_list:
                            # map global connectivity to local connectivity
                            local_node_id = self.submesh.global_to_local_dict[global_node_id]
                            local_connectivity.append(local_node_id)
                
                    if local_connectivity:   
                        for local_node_id in local_connectivity:
                            dofs = np.array(self.id_matrix[local_node_id])
                            
                            pick_dofs = dofs[dirichlet_dict[sub_obj_dir.direction]]
                            self.dirichlet_dof.extend(pick_dofs)
                            
                            for dof in pick_dofs:
                                K[dof,:] = 0.0
                                K[:,dof] = 0.0
                                K[dof,dof] = dirichlet_stiffness
                                f[dof] = 0.0
                            
                
            else:
                print('Dirichlet boundary condition >0 is not yet support!')
                return None
                
        return K,f    
    
    def calc_G_dict(self):
        ''' this function assembles G_dict 
        
        return 
         G_dict : dict
            dictionary with local G matrix
        '''
        # geting subdomain B_dict
        if not self.B_dict: 
            B_dict = self.assemble_interface_boolean_matrix()
        else:
            B_dict = self.B_dict 
        
        # compute null space based on self.solve_opt
        R = self.calc_null_space()

        # store all G in G_dict
        for key in B_dict:
            if self.null_space_size>0:
                Bi = B_dict[key]
                self.G_dict[key] = -Bi.dot(R)
                n_rows,n_cols = self.G_dict[key].shape
                logging.info('Creating G_dict(%i,%i) with interface dof = %i and null space size = %i' %(key[0],key[1],n_rows,n_cols))
            else:
                self.G_dict[key] = None
            
        return self.G_dict

    def assemble_primal_schur_complement(self,type='schur'):
        
        try:
            ii_id = self.local_interior_dofs_list
            bb_id = self.local_interface_dofs_list
        except:
            self.create_interface_and_interior_dof_dicts()
            ii_id = self.local_interior_dofs_list
            bb_id = self.local_interface_dofs_list
            
        num_intetior_dof = len(ii_id)
        K = self.stiffness
        Block_zero = np.zeros([num_intetior_dof,num_intetior_dof])
        Kbb = K[bb_id,:][:,bb_id].todense()
        Kii = K[ii_id,:][:,ii_id].todense()
        Kbi = K[bb_id,:][:,ii_id].todense()
        Kib = Kbi.T
        
        if type=='lumped':
            Sbb = Kbb
        
        elif type=='superlumped':
            Sbb = np.diag(Kbb.diagonal().A1)
            
        elif type=='schur': 
            Kii_inv = np.linalg.inv(Kii)
            Sbb = Kbb - Kbi.dot(Kii_inv).dot(Kib)
        
        elif type=='lumpedschur':
            diag_inv = 1.0/Kii.diagonal()
            Kii_inv = np.diag(diag_inv .A1)
            Sbb = Kbb - Kbi.dot(Kii_inv).dot(Kib)
        
        n_dof = self.mesh.no_of_dofs
        C = np.matrix(np.zeros([n_dof,n_dof]))
        #C = sparse.bmat([[Block_zero, None], [None, Sbb]])
        C[np.ix_(bb_id, bb_id)] = Sbb
        return C

    def append_neighbor_G_dict(self, nei_key, G_matrix):
        ''' This function append neighbor_G_dict
        to neighbor_G_dict internal variables
        
        argument:
            nei_key : int
                neighbor id
            G_matrix : np.matrix
                neighbor G matrix
        
        return 
            self.neighbor_G_dict : dict
        '''
        if G_matrix is not None:
            self.neighbor_G_dict[nei_key,self.key] = G_matrix
        
        return self.neighbor_G_dict
        
    def calc_GtG_row(self):
        ''' This function calculate GtG row
        for the given FETI domain
    
        before using this method please append G_dict from neighbors
    
        this method will update the internal variable
        self.GtG_rows_dict = {}
        '''
        GtG_dict = self.GtG_rows_dict
        Gj_dict = self.neighbor_G_dict
        Gi_dict = self.G_dict
        n = self.null_space_size
        if n>0:
            GiGi = np.zeros([n,n])
            for sub_id, nei_id in Gi_dict:
                    
                    Gi = Gi_dict[sub_id,nei_id]
                    
                    # internal product which is a diagonal 
                    # block entry for the global assembly
                    GiGi += Gi.T.dot(Gi)
                    GtG_dict[sub_id,sub_id] = GiGi
                    
                    try:
                        Gj = Gj_dict[nei_id,sub_id]
                    except:
                        continue

                    # product with the neighbor
                    GiGj = Gi.T.dot(Gj)
                    GtG_dict[sub_id,nei_id] = GiGj
            
            self.G_dict = GtG_dict
        return GtG_dict 
    
    def apply_local_F(self, global_lambda, lambda_dict):
        ''' This function applies local F operator in order to assemble
        global operator F
        
        Let`s define global F as the Dual Interface Flexibility mathematically 
        written as:
        
        F\lambda = d
        
        Where d is the interface displacement gap
        
        The goal of this function is to apply locally the product 
        F\lambda
        
        The interface force lambda is always a interface pair force
        than we can write F\lambda as
        
        F_{(i,j)}\lambda_{(i,j)}
        
        Writting the above equation using local matrix operators
        we have 
        
        B_{(i,j)}*u_i + B_{(j,i)}*u_j
        
        where u_i is the solution of the following linear system
        
        K_i*u_i = \sum_{j=1}^{nei} B{(i,j)}\lambda_{(i,j)}
        
        then the local operator f^local{(i,j)} is defined as:
        
        f^local{(i,j)} = B_{(i,j)}*K_i^{*}*B_{(i,j)}}
        
        then F\lambda can be as :
        [f^local{(i,j)} + f^local{(j,i)}]\lambda_{(i,j)}
        
        the it`s application h_local_({i,j}) is defined as
        h_local_({i,j}) = f^local{(i,j)}\lambda_{(i,j)}
        
        arguments
            global_lambda : np.array
                global lambda
                
            lambda_dict : dict
                dictionary which maps global lambda to
                local lambda
            solve_opt : str
                solver option default = 'cholsps'
        
        return 
            local_h_dict : dict
                dict with local f(i,j)*\lambda(i,j)
        '''
        
        #solving K(i)u(i) = b(i,j) in order to calculte the action of f(i,j)
        force = np.zeros(self.total_dof)
        u_i = self.solve_local_displacement(force,
                                            global_lambda, 
                                            lambda_dict)
                                            
        sub_id = self.key
        local_h_dict = {}                              
        for nei_id in self.submesh.neighbor_partitions:
            Bi = self.B_dict[sub_id,nei_id]  
            local_h_dict[sub_id,nei_id] = Bi.dot(u_i)
        
        return local_h_dict
    
    
class Master():
    def __init__(self,no_of_dofs_per_node=2):
    
        self.GtG_row_dict = {}
        self.course_grid_size = 0
        self.null_space_force_dict = {}
        self.alpha_dict = {}
        self.lambda_dict = {}
        self.lambda_im_dict = {}
        self.G_dict = {}
        self.total_interface_dof = 0
        self.total_nullspace_dof = 0
        self.total_dof = 0
        self.interface_pair_list = []
        self.null_space_force = []
        self.lambda_im = np.array([])
        self.lambda_ker = np.array([])
        self.d_hat_dict = {}
        self.h_dict = {}
        self.subdomain_keys = []
        self.Bi_dict = {}
        self.displacement_dict = {}
        self.__subdomain_dof_info_dict = {} 
        self.no_of_dofs_per_node = no_of_dofs_per_node
        self.G = None
        self.e = None
        self.GGT = None

    def append_partition_dof_info_dicts(self,subdomain_key,subdomain_interface_dofs_dict,subdomain_null_space_size):
        ''' This function append subdomain information about the interface and null space size
        in order to build the global index matrix
        
        argument:
            subdomain_key : int
                subdomain key
            subdomain_interface_dofs_dict : dict
                subdomain dict with interface pairs as keys and num of
                dofs as values
            subdomain_null_space_size : int
                subdomain null space size
        '''
        self.__subdomain_dof_info_dict[subdomain_key] = {}
        self.__subdomain_dof_info_dict[subdomain_key]['num_interface_dof_dict']= subdomain_interface_dofs_dict
        self.__subdomain_dof_info_dict[subdomain_key]['null_space_size'] = subdomain_null_space_size
        return self.__subdomain_dof_info_dict
    
    def append_partition_tuple_info(self, tuple_info):
        subdomain_key = tuple_info[0]
        subdomain_interface_dofs_dict = tuple_info[1]
        subdomain_null_space_size = tuple_info[2]
        self.append_partition_dof_info_dicts(subdomain_key,subdomain_interface_dofs_dict,subdomain_null_space_size)
        
    def append_subdomain_keys(self,sub_key):
        
        key_list = self.subdomain_keys
        key_list.append(sub_key)
        try:
            key_list = list(set(key_list))
        except:
            pass
        
        self.subdomain_keys.sort()
    
    def appendGtG_row(self,G_row,key): 
    
        self.GtG_row_dict[key] = G_row
    
    def append_G_dict(self,G_dict):
        ''' This function appends subdomains
        G_dict to Master in order to build global G_dict
        
        arguments
            G_dict : dict
                dict with subdomain rigig traces
           
        returns
            self.G_dict : dict
                dict with all G_dict
        '''
        
        for key in G_dict:
            Gi = G_dict[key]
            if Gi is not None:
                self.G_dict[key] = Gi
  
        return self.G_dict
    
    def append_h(self,local_h_dict):
        for key in local_h_dict:
            self.h_dict[key] = local_h_dict[key]
    
    def append_null_space_force(self,e,sub_key):
        '''
        '''
        
        if e is not None:
            self.null_space_force_dict[sub_key] = e

    def assemble_G_and_e(self):
        
        key_list = self.subdomain_keys
        
        G = np.zeros([self.total_interface_dof,self.total_nullspace_dof])
        e = np.zeros(self.total_nullspace_dof)
        
        for sub_key in key_list:
            i_index = self.alpha_dict[sub_key]
            
            if not i_index:
                continue
            # assemble null force
            ei = self.null_space_force_dict[sub_key]
            e[i_index] = np.array(ei).flatten()
            
            for nei_key in key_list:
                if sub_key!=nei_key:
                    if sub_key<nei_key:
                        interface_pair = (sub_key,nei_key)
                    else:
                        interface_pair = (nei_key,sub_key)
                        
                    j_index = self.lambda_dict[interface_pair]
                    if not j_index:
                        continue
                    
                    Gij = self.G_dict[sub_key,nei_key]
                    G[np.ix_(j_index,i_index)] = Gij
        
        self.G = G.T
        self.e = e
        return self.G,self.e

    def append_local_B(self,Bi):
        
        flag = 0
        for key in Bi:
            B = Bi[key]
            self.Bi_dict[key] = B
            n, m = np.shape(B)  
            
            # only half because dof are shared      
            if key[0]<key[1]:
                self.total_interface_dof += n 
            if flag == 0:
                self.total_dof += m     
                flag = 1
    
    def assemble_global_B(self):

        key_list = self.subdomain_keys
        
        bool_var = 0
        count_dof = 0
        count_null = 0
        #null_space_size = 0
        #B = np.zeros([self.total_dof,int(self.total_interface_dof)])
        B = np.zeros([self.total_dof,int(self.total_interface_dof)])
        
        for i,sub_id in enumerate(key_list):
            for j,nei_id in enumerate(key_list):

                if (sub_id,nei_id) in self.Bi_dict:
                    
                    Bij = self.Bi_dict[sub_id,nei_id]
                    dof_interface, local_dof = np.shape(Bij)
                    
                    #----------------------------------------------------------
                    # update alpha and rows
                    # has keys -> no do someting -> yes do something else
                    alpha_key = (sub_id,nei_id)
                    #alpha_key_nei = (nei_id,sub_id)
                    
                    # update alpha
                    rows = np.arange(count_null,count_null+local_dof)
                    self.displacement_dict[alpha_key] = rows

                    
                    #----------------------------------------------------------
                    # update lambda and columns
                    # has keys -> no do someting -> yes do something else
                    lambda_key = (sub_id,nei_id)
                    lambda_key_nei = (nei_id,sub_id)
                    
                    if lambda_key in self.lambda_dict:
                        
                        columns = self.lambda_dict[lambda_key]
                    
                    elif lambda_key_nei in self.lambda_dict:
                        columns = self.lambda_dict[lambda_key_nei]
                        
                    else:
                        # update lambda
                        end_columns = count_dof+dof_interface
                        columns = np.arange(count_dof,end_columns)
                        self.lambda_dict[lambda_key] = columns
                        self.lambda_dict[lambda_key_nei] = columns
                        self.interface_pair_list.append(lambda_key)
                        count_dof += dof_interface
                    
                    
                    
                    B[rows,columns.min():columns.max()+1] = Bij.T.todense()
                    
            count_null += local_dof
        return B.T
    
    def assemble_global_F_action(self):
        ''' This function receive local f actions
        and assemble them in a global F actions
        F\lambda can be as :
        [f^local{(i,j)} + f^local{(j,i)}]\lambda_{(i,j)}
        h^local{(i,j)} + h^local{(j,i)}

        '''
        
        h_v = np.zeros(self.total_interface_dof)
        for interface_pair in self.interface_pair_list:
            i_index = self.lambda_dict[interface_pair]
            hij = self.h_dict[interface_pair]
            hji = self.h_dict[interface_pair[::-1]]
            h_v[i_index] = hij + hji
                    
        return h_v
    
    def assemble_GGT(self):
        '''Assemble global GtG based on self.GtG_row_dict
        before with this method you must have indexation matrix 
        see build_local_to_global_mapping method and also you need
        and dict with GtG_row_dict which containts GiGj dictionaties
        
        return
            GtG : np.matrix
            
        '''
        # initalize global G'G
        n = self.total_nullspace_dof
        GtG = np.zeros([n,n])
        key_list = self.subdomain_keys

        for sub_key in key_list:
            
            if sub_key not in self.alpha_dict:
                continue
            
            i_index = self.alpha_dict[sub_key]
            if not i_index:
                continue
            
            Gij = self.GtG_row_dict[sub_key][sub_key,sub_key]
            # add G_row to GtG
            GtG[np.ix_(i_index,i_index)] = Gij
            
            for nei_key in key_list:
                if nei_key!=sub_key:
                    
                    if nei_key not in self.alpha_dict:
                        continue                        
                    
                    j_index = self.alpha_dict[nei_key]
                    if not j_index:
                        continue
                        
                    # add G_row to GtG
                    Gij = self.GtG_row_dict[sub_key][sub_key,nei_key]
                    GtG[np.ix_(i_index,j_index)] = Gij
        self.GGT = GtG
        return self.GGT
    
    def append_d_hat(self,d_hat_dict):
        
        for key in d_hat_dict:
            if key is not None:
                self.d_hat_dict[key] = d_hat_dict[key]

    def assemble_global_d_hat(self):
        ''' Assemble global d_hat where each subdomain ith 
        has local dict as d_hat(i,j) = B(i,j)*u_hat(i)
        where u_hat is defined as: u_hat = inv(K)*[f + B*lambda]
        '''
        
        d_hat = np.zeros(self.total_interface_dof)
        for interface_pair in self.interface_pair_list:
            i_index = self.lambda_dict[interface_pair]
            dij = self.d_hat_dict[interface_pair]
            dji = self.d_hat_dict[interface_pair[::-1]]
            d_hat[i_index] = dij + dji
                                            
        return d_hat
    
    def solve_lambda_im(self):
        ''' This methods solves lambda_im
        it depend on other methods in order to build
        the GtG and G matrices as well the the null space force 
        vector "e"
        
        return 
            lampda_im : np.array
                lampda im which solver the self equilibrium problem
            
        '''
        
        # solve lambda im
        # lambda_im = G'*(G'*G)^-1*e
        GGT = self.assemble_GGT()
        if len(GGT)==0:
            logging.warning('Course Grid size equal 0!')
            return self.lambda_im
            
        logging.info('Course Grid size equal %i by %i!' %GGT.shape)
            
        #logging.debug('GtG')
        #logging.debug(GtG)

        G, e = self.assemble_G_and_e()
        logging.debug('G')
        logging.debug(pd.DataFrame(G))
        logging.debug('e')
        logging.debug(pd.DataFrame(e))
        
        Ug, idf, R = cholsps(GGT)        
        Ug[idf,:] = 0.0
        Ug[:,idf] = 0.0
        Ug[idf,idf] = 1.0
        e[idf] = 0.0
        
        # solving lambda im
        aux1 = scipy.linalg.cho_solve((Ug,False),e)
        lambda_im = G.T.dot(aux1)
        self.lambda_im = lambda_im
        
        return self.lambda_im
    
    def solve_corse_grid(self,r = None):
        ''' This function computes the Projection P in r
        where P = I - G(G'G)G'
        returns w = Pr
        '''
        GGT = self.GGT
        G = self.G
        
        Gd_hat = np.matmul(G,r)
        
        Ug, idf, R = cholsps(GGT)        
        Ug[idf,:] = 0.0
        Ug[:,idf] = 0.0
        Ug[idf,idf] = 1.0
        Gd_hat[idf] = 0.0
        
        alpha_hat = scipy.linalg.cho_solve((Ug,False),Gd_hat)
        
        w = r - np.matmul(G.T,alpha_hat)
        
        self.project_residual = w
        
        return w, alpha_hat
    
    def assemble_dual_force(self,subdomains_dict):
        
        key_list = self.subdomain_keys
        
        i = 0
        for sub_id in key_list:
            sub_i = subdomains_dict[sub_id]
            for nei_id in key_list:
                if sub_id<nei_id:
                    sub_j = subdomains_dict[nei_id]
                    flag = 0
                    if (sub_id,nei_id) in sub_i.dual_force_dict:
                        dij = sub_i.dual_force_dict[sub_id,nei_id]
                        flag = 1
                        
                    if (nei_id,sub_id) in sub_j.dual_force_dict and flag ==1:
                        dji = sub_j.dual_force_dict[nei_id,sub_id]
                        d = dij + dji
                    
                    elif (nei_id,sub_id) in sub_j.dual_force_dict and flag ==0:
                        d = sub_j.dual_force_dict[nei_id,sub_id]
                        flag = 1
                    
                    if flag == 1:
                        if i>0:
                            d_hat = np.vstack([d_hat,d])
                        else:
                            d_hat = d
                            i = 1
                
        return d_hat

    def build_local_to_global_mapping(self):
        ''' This function build global interface and null space
        indexation matrices based on internal variable self.__subdomain_dof_info_dict
        
        before using this method please set the self.__subdomain_dof_info_dict
        using the self.append_partition_dof_info_dicts() method
        
        '''
        key_list = self.subdomain_keys
        dof_per_node = self.no_of_dofs_per_node
        lambda_dof = self.total_interface_dof
        alpha_dof = self.total_nullspace_dof
        
        for sub_id in key_list:
            sub_dict =  self.__subdomain_dof_info_dict[sub_id]
            sub_null_space_size = sub_dict['null_space_size']
            list_alpha = list(range(alpha_dof,alpha_dof + sub_null_space_size))
            self.alpha_dict[sub_id] = list_alpha
            alpha_dof += sub_null_space_size
            
            for nei_id in key_list:
                nei_dict = sub_dict['num_interface_dof_dict']
                if nei_id in nei_dict:
                    if (nei_id,sub_id) not in self.interface_pair_list:
                        self.interface_pair_list.append((sub_id,nei_id))
                        interface_size = nei_dict[nei_id]
                        lampda_list = list(range(lambda_dof,lambda_dof + interface_size))
                        self.lambda_dict[sub_id,nei_id] = lampda_list
                        lambda_dof += interface_size
                else:
                    continue

                       
        self.lambda_im = np.zeros(lambda_dof)
        self.lambda_ker = np.zeros(lambda_dof)
        self.total_interface_dof = lambda_dof
        self.total_nullspace_dof = alpha_dof
        return self.lambda_dict, self.alpha_dict
     
     
class SuperDomain():
    def __init__(self,submesh_dict=None,method='svd'):
        ''' This class is a special class to handle 
        serial Domain Decomposition Solvers
        '''

        self.feti_subdomains_dict = {}
        self.K_list = []
        self.R_list = []
        self.fext_list = []
        self.fint_list = []
        self.B_list = []
        self.displacement_global_dict = {}
        self.lambda_global_dict = {}
        self.lambda_key_list = []
        self.alpha_global_dict = {}
        self.alpha_key_list = []
        self.domains_key_list = []
        self.total_displacement_dofs = 0
        self.total_lambda_dofs = 0
        self.total_alpha_dofs = 0
        self.global_B = None
        self.block_stiffness = None
        self.block_force = None
        self._subdomain_displacement_dict = {}
        self.G_dict = {}
        self.master = Master()
        self.method = method
        if submesh_dict is not None:
            self.create_feti_subdomains_dict(submesh_dict)
            
    def create_feti_subdomains_dict(self, submesh_dict):
        self.domains_key_list = np.sort(list(submesh_dict.keys()))
        self.master.subdomain_keys = self.domains_key_list
        for sub_key in self.domains_key_list:
            sub_i = FETIsubdomain(submesh_dict[sub_key])
            sub_i.solver_opt = self.method
            G_dict = sub_i.calc_G_dict()
            subdomain_interface_dofs_dict = sub_i.num_of_interface_dof_dict
            subdomain_null_space_size = sub_i.null_space_size
            self.master.append_G_dict(G_dict)
            self.master.append_partition_dof_info_dicts(sub_key,subdomain_interface_dofs_dict,subdomain_null_space_size)
            sub_i.calc_dual_force_dict()
            self.master.append_null_space_force(sub_i.null_space_force,sub_key)
            
            # add feti domain to super_domain dict
            self.feti_subdomains_dict[sub_key] = sub_i
         
        
        self.build_local_to_global_mapping()

        return self.feti_subdomains_dict

    def get_feti_subdomains(self,sub_key):
        return self.feti_subdomains_dict[sub_key]

    def build_local_to_global_mapping(self):

        dof_init = 0
        dof_lambda_init = 0
        dof_alpha_init = 0
        for sub_key in self.domains_key_list:
            sub = self.get_feti_subdomains(sub_key)
            
            num_local_dofs = sub.total_dof
            last_dof = dof_init + num_local_dofs
            self.displacement_global_dict[sub_key] = np.arange(dof_init,last_dof)
            dof_init = last_dof
            
            if self.method == 'cholsps':
                Ui, idf, R = sub.compute_cholesky_decomposition()
            elif self.method == 'svd':
                Kinv, R = sub.calc_pinv_and_null_space()
                idf = R.shape[1]
            else:
                logging.error('Method = %s is not defined' %self.method)
                
            sub.set_null_space(R)
            
            if idf:
                last_alpha = dof_alpha_init  + R.shape[1]
                self.alpha_global_dict[sub_key] = np.arange(dof_alpha_init,last_alpha)
                self.alpha_key_list.append(sub_key)
                self.R_list.append(R)
                dof_alpha_init = last_alpha

            sub.create_interface_and_interior_dof_dicts()
            for nei_key in sub.submesh.neighbor_partitions:
                if sub_key<nei_key:
                    last_lambda_dof = dof_lambda_init + len(sub.local_interface_dofs_dict[nei_key])
                    self.lambda_key_list.append((sub_key, nei_key))
                    self.lambda_global_dict[sub_key, nei_key] = np.arange(dof_lambda_init,last_lambda_dof)
                    self.lambda_global_dict[nei_key,sub_key] = self.lambda_global_dict[sub_key, nei_key]
                    dof_lambda_init = last_lambda_dof
                    
        self.total_displacement_dofs = dof_init
        self.total_lambda_dofs = dof_lambda_init
        self.total_alpha_dofs = dof_alpha_init

    def create_K_and_f_list(self):
        
        for sub_key in self.domains_key_list:
            sub = self.get_feti_subdomains(sub_key)
            K1, fint = sub.assemble_k_and_f()
            K1 = K1.todense()

            K, fext1 = sub.assemble_k_and_f_neumann()
            K1, fext1 = sub.insert_dirichlet_boundary_cond(K1,fext1)
            self.K_list.append(K1)
            self.fext_list.append(fext1)
            self.fint_list.append(fint)
                
    def assemble_block_stiffness_and_force(self):

        if not self.K_list:
            self.create_K_and_f_list()


        Kd = scipy.linalg.block_diag(*self.K_list)
        fd = np.hstack(self.fext_list)
        
        self.block_stiffness = Kd
        self.block_force = fd

        return Kd, fd 

    def assemble_global_B(self):

        B = np.zeros([self.total_lambda_dofs,self.total_displacement_dofs])


        for sub_key in self.domains_key_list:
            sub = self.get_feti_subdomains(sub_key)
            Bi_dict = sub.assemble_interface_boolean_matrix()
            idx = self.displacement_global_dict[sub_key]
            for nei_key in sub.submesh.neighbor_partitions:
                idy = self.lambda_global_dict[sub_key, nei_key]
                Bij = Bi_dict[sub_key, nei_key].todense()
                for local_j,global_j in enumerate(idx):
                    for local_i,global_i in enumerate(idy):
                        B[global_i,global_j] = Bij[local_i,local_j]

        self.global_B = B
        return self.global_B

    def assemble_global_G_and_e(self):

        if self.global_B is None:
            B = self.assemble_global_B()
        else:
            B = self.global_B

        if self.block_force is None:
            Kd, fd = self.assemble_block_stiffness_and_force()
        else:
            fd = self.block_force
        
        
        #sending info to master
        for sub_id in self.domains_key_list:
            sub_i = self.feti_subdomains_dict[sub_id]
            for nei_id in sub_i.neighbor_partitions:
                sub_j = self.feti_subdomains_dict[nei_id]
                Gj = sub_j.G_dict[nei_id,sub_id]
                sub_i.append_neighbor_G_dict(nei_id,Gj)
        
        G = np.zeros([self.total_alpha_dofs,self.total_lambda_dofs])
        e = np.zeros(self.total_alpha_dofs)

        for i, sub_key in enumerate(self.alpha_key_list):
            Ri = self.R_list[i]
            idx = self.displacement_global_dict[sub_key]
            Bij = B[:,idx]
            fi = fd[idx]
            ei = - Ri.T.dot(fi)
            Gij = - Ri.T.dot(Bij.T)
            idy = self.alpha_global_dict[sub_key]
            G[idy,:] = Gij
            e[idy] = ei

        return G, e

    def compute_pseudoinverse(self,K):
        Kinv = np.linalg.pinv(K)
        return Kinv

    def assemble_F_and_d(self):

        if self.global_B is None:
            B = self.assemble_global_B()
        else:
            B = self.global_B

        Kd, fd = self.assemble_block_stiffness_and_force()
        Kinv = self.compute_pseudoinverse(Kd)

        F = B.dot(Kinv).dot(B.T)
        d = B.dot(Kinv).dot(fd)

        return F, d

    def eval_subdomain_displacement(self,global_lambda,global_alpha):

        u = {}
        lambda_dict = self.lambda_global_dict
        alpha_dict = self.alpha_global_dict
        method = self.method
        for sub_key in self.domains_key_list:
            sub = self.get_feti_subdomains(sub_key)
            old_method = sub.set_solver_option()
            sub.set_solver_option(method)
            sub.solve_local_displacement(sub.force,global_lambda,lambda_dict)
            u1 = sub.apply_rigid_body_correction(global_alpha,alpha_dict)
            u1 = np.array(u1).flatten()
            u[sub_key] = u1

            sub.set_solver_option(old_method)
        self._subdomain_displacement_dict = u
        return u

    def create_subdomain_primal_schur_complement_list(self,type='schur'):
        
        self.s_list = []
        for sub_key in self.domains_key_list:
            sub = self.get_feti_subdomains(sub_key)
            s = sub.assemble_primal_schur_complement(type)
            self.s_list.append(s)

        return self.s_list

    def assemble_block_primal_schur_complement(self,type='schur'):

        try:
            S_list = self.S_list
        except:
            S_list = self.create_subdomain_primal_schur_complement_list(type)

        S_block = scipy.linalg.block_diag(*S_list)
        return S_block


class Boundary():
    def __init__(self,submesh_obj,val = 0, direction = 'normal', typeBC = 'neumann'):
        
        amfe_mesh = Mesh()
        self.submesh = submesh_obj 
        self.elements_list = submesh_obj.elements_list
        self.neumann_obj = []
        self.value = val
        self.direction = direction
        self.type = typeBC
        # make a deep copy of the element class dict and apply the material
        # then add the element objects to the ele_obj list
        
        self.connectivity = []
        object_series = []
        
        if typeBC == 'neumann':
            for elem_key in self.elements_list: 
                
                self.connectivity.append(np.array(self.submesh.parent_mesh.elements_dict[elem_key]))
                elem_gmsh_key = self.submesh.parent_mesh.elements_type_dict[elem_key]
                elem_type = gmsh2amfe_elem_dict[elem_gmsh_key]
                
                elem_neumann_class_dict = copy.deepcopy(amfe_mesh.element_boundary_class_dict[elem_type])
                elem_neumann_class_dict.__init__(val, direction)
                
                object_series.append(elem_neumann_class_dict)
            #object_series = elements_df['el_type'].map(ele_class_dict)
            self.neumann_obj.extend(object_series)            
            
if __name__ == "__main__":
    # execute only if run as a script
    None    
