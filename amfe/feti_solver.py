# -*- coding: utf-8 -*-
"""
Created on Tue Feb	6 09:10:00 2018

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


try:
	mpi_path = os.environ['MPIDIR']
except:
	mpi_path = '""'

class FetiSolver():
	
	def linear_static(my_system,log = False):
		
		domain = my_system.domain
		domain.split_in_partitions()
		problem_type = domain.problem_type
		num_partitions = len(domain.groups)
		partitions_list = np.arange(1,num_partitions+1)
	
		# saving object to pass to MPIfetisolver
		
		directory = 'temp'
		filename = 'system.aft'
		file_path = os.path.join(directory,filename)
		try:
			os.stat(directory)
		except:
			os.mkdir(directory)		  
		
		
		save_object(my_system, file_path)
		python_solver_file = amfe_dir('amfe\MPIfetisolver.py')
		mpi_exec = os.path.join(mpi_path,'mpiexec').replace('"','')
		
		command = 'cmd /c "'+ mpi_exec + '" -l -n ' + str(num_partitions+1) + ' python '+ python_solver_file + ' ' + \
					filename + ' ' + directory 
		
		# export results to a log file called amfeti_solver.log
		if log:
			command += '>amfeti_solver.log'
		
		print(command)
		output = subprocess.run(command, shell=False, stdout=subprocess.PIPE, 
								universal_newlines=True)
	
		print(os.getcwd())
		
		# loading results from subdomain *.are file
		subdomains_dict = {}
		for i in partitions_list:
			res_path = os.path.join(directory, str(i) + '.are')
			sub_i = load_obj(res_path)
			subdomains_dict[i] = sub_i
		
		# loading solver class
		sol_path = os.path.join(directory,'solver.sol')
		sol = load_obj(sol_path)

		
		# append feti subdomains to system instance
		my_system.feti_subdomain_dict = subdomains_dict
		
		# calculating average displamecent of subdomain
		avg_displacement = FetiSolver.average_displacement_calc(my_system,subdomains_dict)
		
		# updating system displacement
		total_dof = my_system.assembly_class.mesh.no_of_dofs
		my_system.u_output.append(np.zeros(total_dof))
		
		my_system.u_output.append(avg_displacement)
		
		return sol.residual
	
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
		
		self.null_space_size = 0
		self.submesh = submesh_obj
		self.elem_start_index = self.submesh.parent_mesh.node_idx
		self.elem_last_index = len(self.submesh.parent_mesh.el_df.columns)
		self.key = self.submesh.key
		self.global_elem_list = self.submesh.elements_list
		
		self.calc_local_indices()
		amfe_mesh = self.__set_amfe_mesh__()
		self.dual_force_dict = {}
		self.G_dict = {}
		self.total_dof = amfe_mesh.no_of_dofs
		self.null_space_force = None
		
		self.displacement = np.zeros(self.total_dof)
		
		super(FETIsubdomain, self).__init__(amfe_mesh)
		
		self.__find_shared_nodes__()		
		
		self.preallocate_csr()
		self.compute_element_indices()

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
		elem_connec = elem_connec.dropna(1) # removing columns with NaN
		elem_connec = elem_connec.astype(int) # converting all to int
		
		for sub_obj in self.submesh.neumann_submesh:
			for i,elem_key in enumerate(sub_obj.submesh.elements_list):
				local_connectivity = []
				bool_elem = True
				for node_id in elem_connec.loc[elem_key]:
					if node_id in self.submesh.global_node_list:
						# map global connectivity to local connectivity
						local_node_id = self.submesh.global_to_local_dict[node_id]
						local_connectivity.append(local_node_id)
					else:
						bool_elem = False 
						break
				
				if local_connectivity and bool_elem:   
					self.mesh.neumann_connectivity.append(np.array(local_connectivity))
					self.mesh.neumann_obj.extend([sub_obj.neumann_obj[i]])
				
				
	def create_interface_and_interior_dof_dicts(self):
		''' This function read the subdomain information and creates dictionaries
			with local interface nodes, local interface dof,
			local interior dofs and list with global lambda indexes
			also compute the number of interface dofs
			
			create instance variables
			self.local_interface_nodes_dict as dict
			self.local_interface_dofs_dict as dict
			self.local_interior_dofs_dict as dict
			self.lambda_global_indices as list
			self.num_of_interface_dof as int
		'''
		
		total_dof = self.mesh.no_of_dofs
		all_dofs = set(np.arange(total_dof))
		self.local_interface_nodes_dict = {}
		self.lambda_global_indices = []
		self.local_interface_dofs_dict = {}
		self.local_interface_dofs_list = []
		self.local_interior_dofs_list = {}
		self.num_of_interface_dof = 0
		
		for neighbor_subdomain_key in self.submesh.interface_nodes_dict:
			bool_sign = np.sign(neighbor_subdomain_key - self.submesh.key)
			num_interface_nodes = len(self.submesh.interface_nodes_dict[neighbor_subdomain_key])
			total_int_dof = self.mesh.no_of_dofs_per_node*num_interface_nodes
				
			count = 0
			
			self.local_interface_nodes_dict[neighbor_subdomain_key] = []
			self.local_interface_dofs_dict[neighbor_subdomain_key] = []
			
			
			for node_id in self.submesh.interface_nodes_dict[neighbor_subdomain_key]:
				# mapping global dof to local dofs
				local_node_id = self.submesh.global_to_local_dict[node_id]
				self.local_interface_nodes_dict[neighbor_subdomain_key].append(local_node_id)
				node_dof_list = self.id_matrix[local_node_id]
				self.local_interface_dofs_dict[neighbor_subdomain_key].extend(node_dof_list)
				self.local_interface_dofs_list.extend(node_dof_list)		

				for j,dof in enumerate(node_dof_list):					  
					lambda_indice = node_id*self.submesh.problem_type+j
					self.lambda_global_indices.append(lambda_indice)
					count += 1
				
				self.num_of_interface_dof += count
			
			interface_dofs = set(self.local_interface_dofs_list)
			self.local_interface_dofs_list = list(interface_dofs)
			self.local_interior_dofs_list = list(all_dofs.difference(interface_dofs))
		
		return None

				
	def assemble_interface_boolean_matrix(self):
		
		
		total_dof = self.mesh.no_of_dofs
		self.local_interface_nodes_dict = {}
		self.lambda_global_indices = []
		self.local_interface_dofs_dict = {}
		B_dict = {}
		num_of_neighbor = 0
		for neighbor_subdomain_key in self.submesh.interface_nodes_dict:
			bool_sign = np.sign(neighbor_subdomain_key - self.submesh.key)
			num_interface_nodes = len(self.submesh.interface_nodes_dict[neighbor_subdomain_key])
			total_int_dof = self.mesh.no_of_dofs_per_node*num_interface_nodes
				
			B_i = sparse.lil_matrix((total_int_dof,total_dof),dtype=int) 
			count = 0
			
			self.local_interface_nodes_dict[neighbor_subdomain_key] = []
			self.local_interface_dofs_dict[neighbor_subdomain_key] = []
			for node_id in self.submesh.interface_nodes_dict[neighbor_subdomain_key]:
				# mapping global dof to local dofs
				local_node_id = self.submesh.global_to_local_dict[node_id]
				self.local_interface_nodes_dict[neighbor_subdomain_key].append(local_node_id)
				node_dof_list = self.id_matrix[local_node_id]
				self.local_interface_dofs_dict[neighbor_subdomain_key].extend(node_dof_list)
						
				for j,dof in enumerate(node_dof_list):					  
					B_i[count,dof] = 1
					lambda_indice = node_id*self.submesh.problem_type+j
					self.lambda_global_indices.append(lambda_indice)
					count += 1
					
			B_i = bool_sign*B_i
			B_dict[self.submesh.key,neighbor_subdomain_key] = B_i
			if num_of_neighbor>0:
				B = sparse.vstack([B,B_i])
			else:
				B = B_i				   
	
			num_of_neighbor += 1

		n, m = np.shape(B)	
		self.num_of_interface_dof = n			  
		self.B_dict = B_dict	
		return B_dict	 
				
	def solve_local_displacement(self, global_lambda, lambda_id_dict,solve_opt='cholsps'):
		i = 0
		sub_id = self.submesh.key
		for nei_id in self.submesh.neighbor_partitions:
			local_id = lambda_id_dict[sub_id,nei_id]
			Bi = self.B_dict[sub_id,nei_id].todense()	 
			local_lambda = global_lambda[local_id]
			
			b_hati = np.matmul(Bi.T,local_lambda)
			if i == 0:
				b_bar = b_hati
				i = 1
			else:
				b_bar = b_bar + b_hati
				
		# solving K(s)u_bar(s) = f + b(s) in order to calculte the action of F(s)
		b = self.force - b_bar
		
		if solve_opt=='cholsps':
			try:
				Ui = self.full_rank_upper_cholesky.todense()
			except:
				Ui,idf,R = self.compute_cholesky_decomposition()

			idf = self.zero_pivot_indexes
			b[idf] = 0.0
			u_bar = scipy.linalg.cho_solve((Ui,False),b)  
		
		elif solve_opt=='svd':
			try:
			   Kinv = self.psedoinverse
			
			except:
				Kinv,R = self.calc_pinv_and_null_space('svd')
			
			u_bar = Kinv.dot(b)
			
		else:
			print('Solve optiton not implement yet')
		
		self.u_bar = u_bar
		self.displacement =np.array(u_bar).flatten()
		return self.displacement
		
		
	def apply_rigid_body_correction(self,global_alpha, alpha_dict):
		
		sub_id = self.submesh.key
		try:
			u_bar = self.u_bar
		except:
			print('No displacement is defined, then no corretion can be applied. \n' \
				  'please call solve_local_displacement method before rigig correction')

		if self.null_space_size>0:
			R = self.null_space
			local_id = alpha_dict[sub_id]
			local_alpha =  global_alpha[local_id]
			u_bar += R.dot(local_alpha)
		
		self.displacement = np.array(u_bar).flatten()
		return self.displacement

	def compute_cholesky_decomposition(self):
		K, fext = self.assemble_k_and_f_neumann()
		K, fint = self.assemble_k_and_f()
			
		self.insert_dirichlet_boundary_cond()
			
		U, idf, R = cholsps(K)			  
		self.null_space = R 
			
		# store cholesky with free pivots = 0
		self.zero_pivot_indexes = idf
		self.upper_cholesky = U
		fexti = fext.copy()
		self.null_space_size  = len(idf)
		if idf:
			# set free pivots to zero
			
			self.null_space_force = np.matrix(-fext.T.dot(R)).T
																
			Ui = U.copy()
			Ui[idf,:] = 0.0
			Ui[:,idf] = 0.0
			Ui[idf,idf] = 1.0
			fexti[idf] = 0.0

			self.full_rank_upper_cholesky = scipy.sparse.csr_matrix(Ui)
	
		else:
			self.full_rank_upper_cholesky = scipy.sparse.csr_matrix(U)
			self.null_space_force = None
			Ui = U
	
		return Ui,idf,R
		
		
	def calc_null_space(self,solver_opt = 'cholsps'):

		
		if solver_opt=='cholsps':
			
			Ui,idf,R = self.compute_cholesky_decomposition()
			self.null_space_size  = len(idf)
			self.null_space = R
			K, fext = self.assemble_k_and_f_neumann()
			K, fint = self.assemble_k_and_f()			 
			fexti = fext.copy()
			
			if idf:
				# store all G in G_dict
				for key in self.B_dict:
					self.G_dict[key] = -self.B_dict[key].dot(R)

			# remove this command in the future
			self.calc_dual_force(Ui,fexti)
							
		elif solver_opt=='svd': 
			
			self.calc_pinv_and_null_space()
			
		else:
			print('Not implemented')
			return None
			
		return self.null_space


	def calc_dual_force(self,Ui,fexti):
		# calculate the dual force B*Kpinv*f			
		u_hat = scipy.linalg.cho_solve((Ui,False),fexti)
		self.u_hat = np.matrix(u_hat).T
			
		for (sub_id,nei_id) in self.B_dict: 
			Bi = self.B_dict[sub_id,nei_id].todense()
			self.dual_force_dict[sub_id,nei_id] = np.matmul(Bi,self.u_hat)

	def calc_pinv_and_null_space(self,solver_opt='svd',tol=1.0E-8):
		
		if solver_opt=='svd':
			K, fext = self.assemble_k_and_f_neumann()
			K, fint = self.assemble_k_and_f()
			self.insert_dirichlet_boundary_cond()
			
			K = K.todense()
			V,val,U = np.linalg.svd(K)
			
			total_var = np.sum(val)
			
			norm_eigval = val/val[0]
			idx = [i for i,val in enumerate(norm_eigval) if val>tol]
			val = val[idx]
			
			
			invval = 1.0/val[idx]

			subV = V[:,idx]
			
			Kinv =	np.matmul( subV,np.matmul(np.diag(invval),subV.T))
			
			last_idx = idx[-1]
			R = V[:,last_idx+1:]
			self.psedoinverse = Kinv
			self.null_space = R 
			self.null_space_size  = R.shape[1]
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
		
class Master():
	def __init__(self):
	
		self.GtG_row_dict = {}
		self.course_grid_size = 0
		self.id_dict = {} # list if sizes of local GtG 
		self.null_space_force_dict = {}
		self.alpha_dict = {}
		self.lambda_id_dict = {}
		self.lambda_im_dict = {}
		self.G_dict = {}
		self.total_interface_dof = 0
		self.total_nullspace_dof = 0
		self.total_dof = 0
		self.interface_pair_list = []
		self.null_space_force = []
		self.lambda_im = []
		self.lambda_ker = []
		self.d_hat_dict = {}
		self.h_dict = {}
		self.subdomain_keys = []
		self.Bi_dict = {}
		self.displacement_dict = {}
	
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
			
	def add_total_interface_dof(self, sub_dof):
		
		self.total_interface_dof += sub_dof/2.0
	
	def appendG(self,Gi,key):
						
		self.G_dict[key] = Gi
		
	def append_h(self,local_h_dict):
		
		for key in local_h_dict:
			self.h_dict[key] = local_h_dict[key]
		
	
	def append_null_space_force(self,e,sub_key):
		
		
		try:
			sub_null_space_size = len(e)
			self.course_grid_size +=sub_null_space_size
			self.id_dict[sub_key] = sub_null_space_size
			self.null_space_force_dict[sub_key] = e
		except:
			self.id_dict[sub_key] = 0
			pass
	
	def assemble_G(self):
		
		key_list = self.subdomain_keys
		
		bool_var = 0
		count_dof = 0
		count_null = 0
		null_space_size = 0
		G = np.zeros([self.course_grid_size,int(self.total_interface_dof)])
		lambda_id_dict = {}
		self.alpha_dict = {}
		
		for i,sub_id in enumerate(key_list):
			flag = 0
			for j,nei_id in enumerate(key_list):

				try:
					
					Gij = self.G_dict[sub_id][sub_id,nei_id]
					dof_int, null_space_size = np.shape(Gij)
					
					#----------------------------------------------------------
					# update alpha and rows
					# has keys -> no do someting -> yes do something else
					alpha_key = (sub_id,nei_id)
					#alpha_key_nei = (nei_id,sub_id)
					
					# update alpha
					rows = np.arange(count_null,count_null+null_space_size)
					self.alpha_dict[alpha_key] = rows

					
					#----------------------------------------------------------
					# update lambda and columns
					# has keys -> no do someting -> yes do something else
					lambda_key = (sub_id,nei_id)
					lambda_key_nei = (nei_id,sub_id)
					
					if lambda_key in self.lambda_id_dict:
						
						columns = self.lambda_id_dict[lambda_key]
					
					elif lambda_key_nei in self.lambda_id_dict:
						columns = self.lambda_id_dict[lambda_key_nei]
						
					else:
						# update lambda
						end_columns = count_dof+dof_int
						columns = np.arange(count_dof,end_columns)
						self.lambda_id_dict[lambda_key] = columns
						self.lambda_id_dict[lambda_key_nei] = columns
						self.interface_pair_list.append(lambda_key)
						count_dof += dof_int
					
					
					
					G[rows,columns.min():columns.max()+1] = Gij.T			
					flag = 1
					#count_null += null_space_size
					
				except:
					pass
				
				
				
			#count_dof += dof_int	 
			if flag == 1:
				count_null += null_space_size
			
			if sub_id in self.null_space_force_dict:
				ei = self.null_space_force_dict[sub_id]
				if bool_var==1:
					e = np.vstack([e,ei])
					
				else:
					e = ei
					bool_var = 1
			
		self.null_space_force = e

				

			
		return G.T

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
					
					if lambda_key in self.lambda_id_dict:
						
						columns = self.lambda_id_dict[lambda_key]
					
					elif lambda_key_nei in self.lambda_id_dict:
						columns = self.lambda_id_dict[lambda_key_nei]
						
					else:
						# update lambda
						end_columns = count_dof+dof_interface
						columns = np.arange(count_dof,end_columns)
						self.lambda_id_dict[lambda_key] = columns
						self.lambda_id_dict[lambda_key_nei] = columns
						self.interface_pair_list.append(lambda_key)
						count_dof += dof_interface
					
					
					
					B[rows,columns.min():columns.max()+1] = Bij.T.todense()
					
			count_null += local_dof
		return B.T
		
	def assemble_h(self):
		
		# list of all domains
		key_list = self.subdomain_keys
		
		i = 0
		for sub_id in key_list:
			for nei_id in key_list:
				if sub_id<nei_id:
					try:
						hij = self.h_dict[sub_id,nei_id]
						hji = self.h_dict[nei_id,sub_id]
						h = hij + hji
						
						if i>0:
							h_v = np.vstack([h_v,h])
						else:
							h_v = h
							i = 1
		
					except:
						pass
					
		return h_v
		
	
	def assemble_GtG(self):
	# solve course grid
	
		# initalize global G'G
		n = self.course_grid_size
		GtG = np.zeros([n,n])
		# assemple G'G
		key_list = self.subdomain_keys
		
				
		offset = 0
		offset_list = []
		for i,key in enumerate(key_list):
			try:
				Gij = self.GtG_row_dict[key][key,key]
				nlocal = self.id_dict[key]
				init = offset
				end = offset+nlocal 
				GtG[init:end,init:end] = Gij
				offset_list.append(offset)
				offset = end
				self.total_nullspace_dof += nlocal
				count = 0
				for j,nei in enumerate(key_list):
					if nei!=key:
						try:
							Gij = self.GtG_row_dict[key][key,nei]
							GtG[init:end,count:count+nlocal] = Gij
						
						except:
							pass
						
					count +=  self.id_dict[nei]
			except:
				pass
				
		return GtG
	
	
	def append_d_hat(self,d_hat_dict):
		
		for key in d_hat_dict:
			self.d_hat_dict[key] = d_hat_dict[key]
			
	def assemble_global_d_hat(self):
		
		key_list = self.subdomain_keys
		
		i = 0
		for sub_id in key_list:
			for nei_id in key_list:
				if sub_id<nei_id:
					try:
						dij = self.d_hat_dict[sub_id,nei_id]
						dji = self.d_hat_dict[nei_id,sub_id]
						d = dij + dji
						
						if i>0:
							d_hat = np.vstack([d_hat,d])
						else:
							d_hat = d
							i = 1
		
					except:
						pass
					
		
		return d_hat
	
	def solve_lambda_im(self):
		
		# solve lambda im
		# lambda_im = G'*(G'*G)^-1*e
		GtG = self.assemble_GtG()
		print('GtG')
		print(GtG)
		
		B = self.assemble_global_B()
		print('B')
		print(B)
		
		G = self.assemble_G()
		print('G')
		print(G)
		
		e = self.null_space_force
		print('e')
		print(e)
		
		Ug, idf, R = cholsps(GtG)		 
		Ug[idf,:] = 0.0
		Ug[:,idf] = 0.0
		Ug[idf,idf] = 1.0
		e[idf] = 0.0
		
		aux1 = scipy.linalg.cho_solve((Ug,False),e)
		
		lambda_im = np.matmul(G,aux1)
		
		
		#GtGinv = np.linalg.inv(GtG)
		#lambda_im = np.matmul(G,np.matmul(GtGinv,e))
		
		self.lambda_im = lambda_im
		self.lambda_ker = np.matrix(np.zeros(len(lambda_im))).T
		
		for (sub_id,nei_id) in self.lambda_id_dict:
			local_lambda = lambda_im[self.lambda_id_dict[sub_id,nei_id]]
			self.lambda_im_dict[sub_id,nei_id] = local_lambda
			
		self.G = G
		self.GtG = GtG
			
		return self.lambda_im_dict
	
	
	def solve_corse_grid(self,r = None):
		''' This function computes de Projection P in r
		where P = I - G(G'G)G'
		returns w = Pr
		'''
		
		
		GtG = self.GtG
		G = self.G
		
		Gd_hat = np.matmul(G.T,r)
		
		Ug, idf, R = cholsps(GtG)		 
		Ug[idf,:] = 0.0
		Ug[:,idf] = 0.0
		Ug[idf,idf] = 1.0
		Gd_hat[idf] = 0.0
		
		alpha_hat = scipy.linalg.cho_solve((Ug,False),Gd_hat)
		
		#alpha_hat = np.linalg.solve(GtG,Gd_hat)
		
		w = r - np.matmul(G,alpha_hat)
		
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
			
class Boundary():
	def __init__(self,submesh_obj,val = 0, direction = 'normal', typeBC = 'neumann'):
		
		amfe_mesh = amfe.Mesh()
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