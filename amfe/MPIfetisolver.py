# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 08:35:26 2017

@author: ge72tih
"""

# importing mpy4py
from mpi4py import MPI
import os

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

import subprocess
import sys
import dill as pickle
import amfe
import numpy as np
import scipy.sparse as sparse
import scipy

#from .amna import *
#from .feti_solver import *

def run_command(cmd):
    """given shell command, returns communication tuple of stdout and stderr"""
    return subprocess.Popen(cmd, 
                            stdout=subprocess.PIPE, 
                            stderr=subprocess.PIPE, 
                            stdin=subprocess.PIPE).communicate()


# Subdomain level
def subdomain_i(sub_id):    
    submesh_i = domain.groups[sub_id]
    sub_i = amfe.FETIsubdomain(submesh_i)
    B_dict = sub_i.assemble_interface_boolean_matrix()
    sub_i.calc_null_space()
    R = sub_i.null_space
    U = sub_i.upper_cholesky
    
    # checking cholesky decompostion
    K = sub_i.stiffness
    Knew = np.matmul(U.T,U)
    
    chol_error = np.linalg.norm(Knew - K)
    
    Fext = sub_i.force
    #e = Fext.dot(R)
    
    # store all G in G_dict
    Gi_dict = {}
    for key in B_dict:
        if sub_i.zero_pivot_indexes:
            Gi_dict[key] = -B_dict[key].dot(R)
        else:
            Gi_dict[key] = None
        
    return sub_i, Gi_dict


def create_GtG_rows(Gi_dict,Gj_dict,sub_id):
        

    GtG_dict = {}

    for local_id, nei_id in Gi_dict:
        if local_id == sub_id:
                                    
            try:
                Gi = Gi_dict[sub_id,nei_id]
                GiGi = Gi.T.dot(Gi)
                GtG_dict[sub_id,sub_id] = GiGi
                
                Gj = Gj_dict[nei_id,sub_id]
                GiGj = Gi.T.dot(Gj)
                GtG_dict[sub_id,nei_id] = GiGj
                
            except:
                pass
             
            
    return GtG_dict 


def exchange_info(sub_id,master,master_append_func,var,partitions_list):
    ''' This function exchange info (lists, dicts, arrays, etc) with the 
    neighbors subdomains. Every subdomain has a master objective which receives 
    the info and to some calculations based on it.
    
    Inpus:
        sub_id: id of the subdomain
        master: object with global information
        master_append_func: master function to append the exchanged var
        var: variable to be exchanged among neighbors
        partitions_list: list of subdomain neighbors
    
    '''
    
    # forcing list as inputs
    if type(var)!=list:
        var_list = [var]
    else:        
        var_list = var
        
    if type(master_append_func)!=list:
        master_append_func_list = [master_append_func]
    else:
        master_append_func_list = master_append_func
    
    if len(var)!=len(master_append_func):
        print('Error exchaning information among subdomains')
        return None
    
    for var,master_append_func in zip(var_list,master_append_func_list):
        for partition_id in partitions_list:
            if partition_id != sub_id:
                #send to neighbors local h 
                comm.send(var, dest=partition_id)
		
                # receive local h from neighbors
                nei_var = comm.recv(source=partition_id)

                try:
                    master_append_func(nei_var)
             
                except:       
                    master_append_func(nei_var,partition_id)            		        		
            else:
                try:   
                    master_append_func(var)
             
                except:
                    master_append_func(var,sub_id)
            		
    
    return master


def subdomain_apply_F(sub_i,lambda_id_dict,pk):
    ''' This step is to calculate a_hat in every domain
    '''
        
    i = 0
    sub_id = sub_i.submesh.key
    for nei_id in sub_i.submesh.neighbor_partitions:
        local_id = master.lambda_id_dict[sub_id,nei_id]
        Bi = sub_i.B_dict[sub_id,nei_id].todense()    
        local_pk = pk[local_id]
        
        b_hati = np.matmul(Bi.T,local_pk)
        if i == 0:
            b_hat = b_hati
            i = 1
        else:
            b_hat = b_hat + b_hati
            
    # solving K(s)a(s) = b(s) in order to calculte the action of F(s)
    Ui = sub_i.full_rank_upper_cholesky.todense()
    idf = sub_i.zero_pivot_indexes
    b_hat[idf] = 0.0
    
    a_hat = scipy.linalg.cho_solve((Ui,False),b_hat)  
    

    # build local h with local F(v)    
    local_h_dict = {}    
    for nei_id in sub_i.submesh.neighbor_partitions:
        Bi = sub_i.B_dict[sub_id,nei_id].todense()    
        local_h_dict[sub_id,nei_id] = Bi.dot(a_hat)
       
    # sending local h for master 
    return local_h_dict
        


def global_apply_F(master,sub_i,lambda_id_dict,pk):
    
    sub_id = sub_i.submesh.key
    local_h_dict = subdomain_apply_F(sub_i,lambda_id_dict,pk)
    
    
    master_func_list = [master.append_h,master.append_d_hat]
    var_list = [local_h_dict,sub_i.dual_force_dict]
    
    master = exchange_info(sub_id,master,master_func_list,var_list,partitions_list)
    
        
    return master.assemble_h()


def subdomain_step4(sub_i,lambda_sol,alpha):
            
    i = 0
    for nei_id in sub_i.submesh.neighbor_partitions:
        local_id = master.lambda_id_dict[sub_id,nei_id]
        Bi = sub_i.B_dict[sub_id,nei_id].todense()    
        local_lambda = lambda_sol[local_id]
        
        if (sub_id,nei_id) in master.alpha_dict:
            alpha_id = master.alpha_dict[sub_id,nei_id]
        
        
        b_hati = np.matmul(Bi.T,local_lambda)
        if i == 0:
            b_hat = b_hati
            i = 1
        else:
            b_hat = b_hat + b_hati

    
    
    f = sub_i.force
    b = f - b_hat
    
    
    # solving K(s)a(s) = b(s) in order to calculte the action of F(s)
    Ui = sub_i.full_rank_upper_cholesky.todense()
    idf = sub_i.zero_pivot_indexes
    b[idf] = 0.0
    u_hat = scipy.linalg.cho_solve((Ui,False),b)  
    
    if idf:
        R = sub_i.null_space
        local_alpha = alpha[alpha_id]
        u_bar = u_hat + np.matmul(R,local_alpha)
        sub_i.displacement = u_bar
    else:    
        sub_i.displacement = u_hat
            
    return sub_i.displacement


def beta_calc(k,y_dict=None,w_dict=None):
    
    if k == 0:
        return 0.0
    
    else:
        yk1 = y_dict[0]
        yk2 = y_dict[1]
        wk1 = w_dict[0]
        wk2 = w_dict[1]
        
        aux1 = float(yk1.T.dot(wk1))
        aux2 = float(yk2.T.dot(wk2))
        
        beta = aux1/aux2
        
        return beta
        
        
def alpha_calc(yk,wk,pk,h):
    aux1 = yk.T.dot(wk)
    aux2 = pk.T.dot(h)
    
    alpha = float(aux1/aux2)
    
    return alpha


class ParallelSolver():
    def __init__(self):
        
        self.residual = []
        
    def mpi_solver(self,sub_domain,num_partitions,n_int = 500, tol = 1E-10):
        ''' solve linear FETI problem with PCGP with parcial reorthogonalization
        '''
        
        if rank<=num_partitions and rank>0:
            
            global sub_id
            sub_id = rank
            print("%%%%%%%%%%%%%%%%%%% START %%%%%%%%%%%%%%%%%%%%%%%%%%%%")    
            print("Solving domain %i from size %i" %(rank,num_partitions))   
            sub_i, Gi_dict = subdomain_i(sub_id) 
            print(Gi_dict.keys())  
            
             
            # sending message for neighbors
            for nei_id in sub_i.submesh.neighbor_partitions:
                print("\nSending message from %i to neighbor %i" %(sub_id,nei_id))
                if (sub_id,nei_id) in Gi_dict: 
                    comm.send(Gi_dict[sub_id, nei_id], dest=nei_id)
            
                    
            master.subdomain_keys = partitions_list
            master.appendG(Gi_dict,sub_id)
            
            # reciving message for neighbors
            Gj_dict = {}
            for nei_id in sub_i.submesh.neighbor_partitions:
                print("\nReceiving message at subdomain %i from neighbor %i" %(sub_id,nei_id))
                Gj_dict[nei_id,sub_id]= comm.recv(source=nei_id)
                master.appendG(Gj_dict,nei_id)
                        
            print(Gj_dict.keys())   
            
            # creating GtG rows
            GtG_rows_dict = create_GtG_rows(Gi_dict,Gj_dict,sub_id)
            null_space_size = sub_i.null_space_size
            
            print('Local GtG rows')
            print(GtG_rows_dict.keys())
                    
            master_func_list = [master.appendGtG_row, master.append_local_B,
                                master.append_null_space_force]
            
            var_list = [GtG_rows_dict,sub_i.B_dict,sub_i.null_space_force]
            
            exchange_info(sub_id,master,master_func_list,var_list,partitions_list)
            
        
            print('Master GtG keys')
            print(master.GtG_row_dict.keys())    
            
            # print local rows
            print('Local rows for each subdomain')
            for key in master.GtG_row_dict:
                print(master.GtG_row_dict[key])   
        
            print('B dict')
            print(master.Bi_dict.keys())   
            for key in master.Bi_dict:
                print(master.Bi_dict[key].todense())
            
            print('G dict')
            print(master.G_dict.keys())   
            for key in master.G_dict:
                print(master.G_dict[key])
                
            print('total interface dof %i' %master.total_interface_dof)
            print('total dof %i' %master.total_dof)
            print('Null space size %i' %master.total_nullspace_dof)
            print('Null space size %i' %master.course_grid_size)
            
            
            lambda_im_dict = master.solve_lambda_im()
            lambda_im = master.lambda_im
            lambda_ker = master.lambda_ker
            
            print('master.lambda_im')
            print(master.lambda_im)
            
            print('lambda_im_dict')
            print(lambda_im_dict)
            
            lambda_id_dict = master.lambda_id_dict
            print('lambda_id_dict')
            print(lambda_id_dict)
            
            # apply local F operations
            local_h_dict = subdomain_apply_F(sub_i,lambda_id_dict,lambda_im)
            print('local_h_dict.keys()')
            print(local_h_dict.keys())
            for key in local_h_dict:
                print(local_h_dict[key])
            
            
            
            master_func_list = [master.append_h, master.append_d_hat]
            
            var_list = [local_h_dict, sub_i.dual_force_dict]
            
            exchange_info(sub_id,master,master_func_list,var_list,partitions_list)
            
            Fim = master.assemble_h() # Fpk = B*Kpinv*B'*pk
            print('Fim')
            print(Fim)
            
            d = master.assemble_global_d_hat() # dual force global assemble
            print('d')
            print(d)
            
            # init precond
            n = len(lambda_ker)
            precond = np.eye(n,n)
            
            # dicts to story past iterations
            w_dict = {}
            y_dict = {}
            pk1 = np.zeros([n,1])
            
            
            # initial residual
            r0 = d - Fim
            rk = r0
            
            #---------------------------------------------------------------------
            # PCPG algorithm        
            for i in range(n_int):
            
                # check norm of rk
                norm_rk = np.linalg.norm(rk)
                print('iteration %i, norm rk = %f' %(i,norm_rk))
                
                # solve course grid
                wk, alpha_hat = master.solve_corse_grid(rk) # assemble local d_hats and solve P*F*d_hat
                norm_wk = np.linalg.norm(wk)
                self.residual.append(norm_wk)
                #print('wk =', wk.T)
                
                print('iteration %i, norm wk = %0.2e /n' %(i,norm_wk))
                if norm_wk < tol:
                    break
                
                #------------------------------------------------------------------
                # calculation for the preconditioning
                
                sub_i.solve_local_displacement(lambda_im, lambda_id_dict)
                print('u_bar')
                print(sub_i.u_bar)
                
                #------------------------------------------------------------------
                
                yk = np.matmul(precond,wk)
                
                # story past iterations
                if i>0:
                    # append w and y to dict
                    w_dict[1] = w_dict[0].copy()
                    y_dict[1] = w_dict[0].copy()
                    w_dict[0] = wk
                    y_dict[0] = yk
                    
                else:
                    # append w and y to dict
                    w_dict[0] = wk 
                    y_dict[0] = yk
                
                # calc beta
                beta = beta_calc(i,y_dict,w_dict)
                print('beta = ', beta)
                
                # calc pk
                pk = yk + beta*pk1
                print('pk', pk.T)
                
                # this step depends on comunication of subdomains
                # calc local Fpk
                local_h_dict = subdomain_apply_F(sub_i,lambda_id_dict,pk)
                # append local F operation into master instance
                
                master_func_list = [master.append_h, master.append_d_hat]
            
                var_list = [local_h_dict, sub_i.dual_force_dict]
            
                exchange_info(sub_id,master,master_func_list,var_list,partitions_list)
                
                            
                h = master.assemble_h() # Fpk = B*Kpinv*B'*pk
                print('h = ', h.T)
                
                # calc alpha k # do it in parallel
                alpha = alpha_calc(yk,wk,pk,h)
                print('alpha =', alpha)
            
                # update lambda_ker and r
                delta_rk = alpha*h
                lambda_ker = lambda_ker - alpha*pk
                rk = rk - delta_rk
                
                # update variables
                pk1 = pk
                
            # lagrange multiplier solution
            lambda_sol =  lambda_im + lambda_ker
            print('lambda_sol')
            print(lambda_sol.T)
            
            # compute global error
            d = master.assemble_global_d_hat() # dual force global assemble
            
            # calc Global F_lambda
            F_lambda = global_apply_F(master,sub_i,lambda_id_dict,lambda_sol)
            
            print('F_lambda =', F_lambda.T)
            
            d_hat = d
            
            d_hat = d - F_lambda
        
            wk, alpha_hat = master.solve_corse_grid(d_hat)
            
            u = subdomain_step4(sub_i, lambda_sol, alpha_hat)
            print('u =', u.T)
            
            res_path = os.path.join(directory,str(sub_id) + '.are')
            amfe.save_object(sub_i, res_path)
            return sub_i
    
            print("%%%%%%%%%%%%%%%%%%% END %%%%%%%%%%%%%%%%%%%%%%%%%%%%")    
            
        else:
            print("Nothing to do on from process %i " %rank)   
            return None





if __name__ == "__main__":
    # execute only if run as a script
    domain_pkl_path = sys.argv        

    args = []
    for s in sys.argv:
        args.append(s)    
        
    # load FEA case
    case_obj = args[1]
    directory = args[2]
    case_path = os.path.join(directory,case_obj)
    
    print('########################################')
    print(case_obj)
    print(directory)
    print(case_path)
    print('########################################')
    
    my_system = amfe.load_obj(case_path)
    domain = my_system.domain
    # Instanciating Global Master class to handle Coarse problem
    master = amfe.Master()
    
    num_partitions = len(domain.groups)
    partitions_list = np.arange(1,num_partitions+1)
    parsolver = ParallelSolver()
    sub_i = parsolver.mpi_solver(domain,num_partitions)
    
    if rank == 1:
        solver_path = os.path.join(directory, 'solver.sol')
        amfe.save_object(parsolver, solver_path)




