# Copyright (c) 2017, Lehrstuhl fuer Angewandte Mechanik, Technische Universitaet Muenchen.
#
# Distributed under BSD-3-Clause License. See LICENSE-File for more information.
#
"""
Module handling the whole mechanical system, no matter if it's a finite element system, defined by certain parameters
or a multibody system.
"""


import time
import os
import copy
import h5py
import numpy as np
import pandas as pd
from scipy.sparse import bmat

from .mesh import Mesh, SubMesh
from .assembly import Assembly
from .boundary import DirichletBoundary
from .boundary import Boundary
from .solver import *
from scipy.sparse import linalg as splinalg
import scipy.sparse as sparse


__all__ = [
    'MechanicalSystem',
    'MechanicalSystemStateSpace',
    'ReducedSystem',
    'ReducedSystemStateSpace',
    'reduce_mechanical_system',
    'convert_mechanical_system_to_state_space',
    'reduce_mechanical_system_state_space',
    'MechanicalAssembly',
    'CraigBamptonComponent',
    'get_dirichlet_dofs'
]


class MechanicalSystem():
    '''
    Master class for mechanical systems with the goal to black-box the routines of assembly and element selection.

    Attributes
    ----------
    mesh_class : instance of Mesh()
        Class handling the mesh.
    assembly_class : instance of Assembly()
        Class handling the assembly.
    dirichlet_class : instance of DirichletBoundary
        Class handling the Dirichlet boundary conditions.
    domain : instance of SubMesh class
        Class handling the Subdomains for FETI solver.
    T_output : list of floats
        List of timesteps saved.
    u_output : list of ndarrays
        List of unconstrained displacement arrays corresponding to the timesteps in T_output.
    S_output : list of ndarrays
        List of stress arrays corresponding to the timesteps in T_output.
    E_output : list of ndarrays
        List of strain arrays corresponding to the timesteps in T_output.
    stress : ndarray
        Array of nodal stress of the last assembly run. Shape is (no_of_nodes, 6).
    strain : ndarray
        Array of nodal strain of the last assembly run. Shape is (no_of_nodes, 6).
    stress_recovery : bool
        Flag for option stress_recovery.
    iteration_info : ndarray
        Array containing the information of an iterative solution procedure. Eteration_info[:,0] is the time
        information, iteration_info[:,1] is the number of iterations, iteration_info[:,3] is the residual.
    M_constr : ?
        Mass matrix
    D_constr : ?
        Damping matrix
    '''

    def __init__(self, stress_recovery=False):
        '''
        Parameters
        ----------
        stress_recovery : bool, optional
            Flag, for setting stress recovery option. Default is False.

        '''
        self.stress_recovery = stress_recovery
        self.T_output = []
        self.u_output = []
        self.S_output = []
        self.E_output = []
        self.stress = None
        self.strain = None
        self.iteration_info = np.array([])
        self.domain = []
        # instantiate the important classes needed for the system:
        self.dirichlet_class = DirichletBoundary(np.nan)
        
        # old implementation
        #self.mesh_class = Mesh()
        #self.assembly_class = Assembly(self.mesh_class)
        
        # new implementation using the set_mesh_obj
        self.set_mesh_obj()

        # make syntax a little bit leaner
        # !Christian Meyer: ! careful: This prohibits to easily change dirichlet_class instance, because old instance
        # still will be referenced!
        self.unconstrain_vec = self.dirichlet_class.unconstrain_vec
        self.constrain_vec = self.dirichlet_class.constrain_vec
        self.constrain_matrix = self.dirichlet_class.constrain_matrix

        # initializations to be overwritten by loading functions
        self.M_constr = None
        self.D_constr = None
        self.no_of_dofs_per_node = None

        # external force to be overwritten by user-defined external forces
        # self._f_ext_unconstr = lambda t: np.zeros(self.mesh_class.no_of_dofs)


    def set_mesh_obj(self, mesh_obj = None):
        ''' This method sets the mesh object (mesh_obj) in the mesh_class variable
        The user can load the mesh from a file using the load_mesh_from_gmsh method,
        but if the user already have a mesh_obj instance then, he can use the
        set_mesh_obj function in order to set the variable mesh_class.
        If the mesh_obj has boundaries conditions, the mechanical_system will NOT
        inherit the boundaries conditions from the mesh_obj
        
        Parameters
        ----------
            mesh_obj:
                instance of amfe.mesh class
        
        Returns
        -------
        None
        '''
        
        if mesh_obj is None:
            self.mesh_class = Mesh()
            self.assembly_class = Assembly(self.mesh_class)
        else:     
            self.mesh_class = mesh_obj
            self.assembly_class = Assembly(mesh_obj)
            

        return None


    def set_domain(self, key, material, mesh_prop = 'phys_group'):
        ''' this function sets a domain after the mesh_class is intantiated
        then use set_mesh_obj before use this methods.
        You also can load the mesh file direct using the load_mesh_from_gmsh 
        method.
        
        Parameters
        ----------
        Key : int
            Number of the mesh propertie
        
        material : Material Class instance
            Material class which will be assigned to the elements
        mesh_prop : str
            mesh_prop : {'phys_group', 'geom_entity', 'el_type', 'partition_id'}
            optional label of which the element should be chosen from. Standard is
            physical group.
            
          
        
        ex.: self.set_domain(11)

        Returns
        -------
        None     
        
        
        '''       
        try:
            self.mesh_class.load_group_to_mesh(key, material, mesh_prop)
            submesh = self.mesh_class.set_domain(mesh_prop,key)
            submesh.set_material(material)
            
            self.no_of_dofs_per_node =self.mesh_class.no_of_dofs_per_node
            try:
                self.dirichlet_class.no_of_unconstrained_dofs = self.mesh_class.no_of_dofs
            except:
                print('No Dirichlet Boundary conditions was found')
                pass
                
            self.assembly_class.preallocate_csr()
            self.domain = submesh
        except:
            raise('Please make sure use have a mesh object in the .mesh_class variable')
            
        
        return submesh


    def load_mesh_from_gmsh(self, msh_file, phys_group, material, scale_factor=1):

        '''
        Load the mesh from a msh-file generated by gmsh.

        Parameters
        ----------
        msh_file : str
            File name to an existing .msh file.
        phys_group : int
            Integer key of the physical group which is considered as the mesh part.
        material : amfe.Material
            Material associated with the physical group to be computed.
        scale_factor : float, optional
            Scale factor for the mesh to adjust the units. The default value is 1, i.e. no scaling is done.
        '''

        self.mesh_class.import_msh(msh_file, scale_factor=scale_factor)
        self.mesh_class.load_group_to_mesh(phys_group, material)
        self.no_of_dofs_per_node = self.mesh_class.no_of_dofs_per_node

        self.assembly_class.preallocate_csr()
        self.dirichlet_class.no_of_unconstrained_dofs = self.mesh_class.no_of_dofs
        self.dirichlet_class.update()

    def deflate_mesh(self):
        '''
        Remove free floating nodes not connected to a selected element from the mesh.
        '''

        self.mesh_class.deflate_mesh()
        self.assembly_class.preallocate_csr()
        self.dirichlet_class.no_of_unconstrained_dofs = self.mesh_class.no_of_dofs
        self.dirichlet_class.update()

    def load_mesh_from_csv(self, node_list_csv, element_list_csv, no_of_dofs_per_node=2,
                           explicit_node_numbering=False,
                           ele_type=False):
        '''
        Loads the mesh from two csv-files containing the node and the element list.

        Parameters
        ----------
        node_list_csv: str
            Filename of the csv-file containing the coordinates of the nodes (x, y, z)
        element_list_csv: str
            Filename of the csv-file containing the nodes which belong to one element
        no_of_dofs_per_node: int, optional
            Degree of freedom per node as saved in the csv-file
        explicit_node_numbering : bool, optional
            Flag stating, if the node numbers are explcitly numbered in the csv file, i.e. if the first column gives
            the numbers of the nodes.
        ele_type: str
            Spezifiy elements type of the mesh (e.g. for a Tri-Mesh different elements types as Tri3, Tri4, Tri6 can be
            used). If not spezified value is set to 'False'.
        '''

        self.mesh_class.import_csv(node_list_csv, element_list_csv,
                                   explicit_node_numbering=explicit_node_numbering,
                                   ele_type=ele_type)
        self.no_of_dofs_per_node = no_of_dofs_per_node
        self.assembly_class.preallocate_csr()
        return

    def tie_mesh(self, master_key, slave_key,
                 master_prop='phys_group',
                 slave_prop='phys_group',
                 tying_type='fixed',
                 verbose=False,
                 conform_slave_mesh=False,
                 fix_mesh_dist=1E-3):
        '''
        Tie nonconforming meshes for a given master and slave side.

        Parameters
        ----------
        master_key : int or string
            Mesh key of the master face mesh. The master face mesh has to be at least the size of the slave mesh. It is
            better, when the master mesh is larger than the slave mesh.
        slave_key : int or string
            Mesh key of the slave face mesh or point cloud
        master_prop : string, optional
            Mesh property for which master_key is specified. Default value: 'phys_group'.
        slave_prop : string, optional
            Mesh property for which slave_key is specified. Default value: 'phys_group'
        tying_type : string {'fixed', 'slide'}
            Mesh tying type. 'fixed' glues the meshes together while 'slide' allows for a sliding motion between the
            meshes.

        Notes
        -----
        The master mesh has to embrace the full slave mesh. If this is not the case, the routine will fail, a slave
        point outside the master mesh cannot be addressed to a specific element.
        '''

        vals = self.mesh_class.tie_mesh(master_key=master_key,
                                        slave_key=slave_key,
                                        master_prop=master_prop,
                                        slave_prop=slave_prop,
                                        tying_type=tying_type,
                                        verbose=verbose,
                                        fix_mesh_dist=fix_mesh_dist)

        self.dirichlet_class.add_constraints(*vals)
        self.dirichlet_class.update()
        return

    def apply_dirichlet_boundaries(self, key, coord, mesh_prop='phys_group'):
        '''
        Apply dirichlet-boundaries to the system.

        Parameters
        ----------
        key : int
            Key for mesh property which is to be chosen. Matches the group given in the gmsh file. For help, the
            function mesh_information or boundary_information gives the groups.
        coord : str {'x', 'y', 'z', 'xy', 'xz', 'yz', 'xyz'}
            Coordinates which should be fixed.
        mesh_prop : str {'phys_group', 'geom_entity', 'el_type'}, optional
            Label of which the element should be chosen from. Default is 'phys_group'.
        '''

        id_matrix = self.assembly_class.id_matrix
        self.mesh_class.set_dirichlet_bc(key, coord, mesh_prop, id_matrix=id_matrix)
        self.dirichlet_class.constrain_dofs(self.mesh_class.dofs_dirichlet)
        return

    def apply_neumann_boundaries(self, key, val, direct,
                                 time_func=None,
                                 shadow_area=False,
                                 mesh_prop='phys_group'):
        '''
        Apply neumann boundaries to the system via skin elements.

        Parameters
        ----------
        key : int
            Key of the physical domain to be chosen for the neumann bc.
        val : float
            Value for the pressure/traction onto the element.
        direct : ndarray or str 'normal'
            Direction, in which force should act at.
        time_func : function object
            Function object returning a value between -1 and 1 given the input t:

            >>> val = time_func(t)

        shadow_area : bool, optional
            Flag, if force should be proportional to shadow area of surface with respect to direction. Default: False.
        mesh_prop : str {'phys_group', 'geom_entity', 'el_type'}, optional
            Label of which the element should be chosen from. Default is phys_group.
        '''

        self.mesh_class.set_neumann_bc(key=key,
                                       val=val,
                                       direct=direct,
                                       time_func=time_func,
                                       shadow_area=shadow_area,
                                       mesh_prop=mesh_prop)
        self.assembly_class.compute_element_indices()
        return

    def export_paraview(self, filename, field_list=None):
        '''
        Export the system with the given information to paraview.

        Parameters
        ----------
        filename : str
            Filename to which the xdmf file and the hdf5 file will be saved.
        field_list : list, optional
            List of tuples containing a field to be exported as well as a dictionary with the attribute information of
            the hdf5 file. Example:

                >>> # example field list with reduced displacement not to export
                >>> # ParaView and strain epsilon to be exported to ParaView
                >>> field_list = [(q_red, {'ParaView':False, 'Name':'q_red'}),
                                  (eps, {'ParaView':True,
                                         'Name':'epsilon',
                                         'AttributeType':'Tensor6',
                                         'Center':'Node',
                                         'NoOfComponents':6})]
        '''

        if field_list is None:
            field_list = []
        t1 = time.time()
        if len(self.T_output) is 0:
            self.T_output.append(0)
            self.u_output.append(np.zeros(self.mesh_class.no_of_dofs))
            if self.stress_recovery:
                self.S_output.append(np.zeros((self.mesh_class.no_of_nodes, 6)))
                self.E_output.append(np.zeros((self.mesh_class.no_of_nodes, 6)))
        print('Start exporting mesh for paraview to:\n    ', filename)

        if self.stress_recovery and len(self.S_output) > 0 \
           and len(self.E_output) > 0:
            no_of_timesteps = len(self.T_output)
            S_array = np.array(self.S_output).reshape((no_of_timesteps, -1))
            E_array = np.array(self.E_output).reshape((no_of_timesteps, -1))
            S_export = (S_array.T, {'ParaView':True,
                                  'Name':'stress',
                                  'AttributeType':'Tensor6',
                                  'Center':'Node',
                                  'NoOfComponents':6})
            E_export = (E_array.T, {'ParaView':True,
                                  'Name':'strain',
                                  'AttributeType':'Tensor6',
                                  'Center':'Node',
                                  'NoOfComponents':6})
            field_list.append(S_export)
            field_list.append(E_export)

        bmat = self.dirichlet_class.b_matrix()
        self.mesh_class.save_mesh_xdmf(filename, field_list, bmat, u=self.u_output, timesteps=self.T_output)
        t2 = time.time()
        print('Mesh for paraview successfully exported in ' +
              '{0:4.2f} seconds.'.format(t2 - t1))
        return

    def M(self, u=None, t=0):
        '''
        Compute the Mass matrix of the dynamical system.

        Parameters
        ----------
        u : ndarray, optional
            Array of the displacement.
        t : float
            Time.

        Returns
        -------
        M : sp.sparse.sparse_matrix
            Mass matrix with applied constraints in sparse csr-format.
        '''

        if u is not None:
            u_unconstr = self.unconstrain_vec(u)
        else:
            u_unconstr = None

        M_unconstr = self.assembly_class.assemble_m(u_unconstr, t)
        self.M_constr = self.constrain_matrix(M_unconstr)
        return self.M_constr

    def K(self, u=None, t=0):
        '''
        Compute the stiffness matrix of the mechanical system.

        Parameters
        ----------
        u : ndarray, optional
            Displacement field in voigt notation.
        t : float, optional
            Time.

        Returns
        -------
        K : sp.sparse.sparse_matrix
            Stiffness matrix with applied constraints in sparse csr-format.
        '''

        if u is None:
            u = np.zeros(self.dirichlet_class.no_of_constrained_dofs)

        K_unconstr = \
            self.assembly_class.assemble_k_and_f(self.unconstrain_vec(u), t)[0]

        return self.constrain_matrix(K_unconstr)

    def D(self, u=None, t=0):
        '''
        Return the damping matrix of the mechanical system.

        Parameters
        ----------
        u : ndarray, optional
            Displacement field in voigt notation.
        t : float, optional
            Time.

        Returns
        -------
        D : sp.sparse.sparse_matrix
            Damping matrix with applied constraints in sparse csr-format.
        '''

        if self.D_constr is None:
            return self.K()*0
        else:
            return self.D_constr

    def f_int(self, u, t=0):
        '''
        Return the elastic restoring force of the system.
        '''
        f_unconstr = \
            self.assembly_class.assemble_k_and_f(self.unconstrain_vec(u), t)[1]
        return self.constrain_vec(f_unconstr)

    def _f_ext_unconstr(self, u, t):
        '''
        Return the unconstrained external force coming from the Neumann BCs. This function is just a placeholder if you
        want to change the behavior of f_ext: This function may be monkeypatched if necessary, for instance, when a
        global external force, e.g. gravity, should be applied.
        '''

        f_unconstr = \
            self.assembly_class.assemble_k_and_f_neumann(self.unconstrain_vec(u), t)[1]
        return f_unconstr

    def f_ext(self, u, du, t):
        '''
        Return the nonlinear external force of the right hand side of the equation, i.e. the excitation.
        '''

        if u is None:
            u = np.zeros(self.dirichlet_class.no_of_constrained_dofs)
        return self.constrain_vec(self._f_ext_unconstr(u, t))

    def K_and_f(self, u=None, t=0):
        '''
        Compute tangential stiffness matrix and nonlinear force vector in one assembly run.
        '''

        if u is None:
            u = np.zeros(self.dirichlet_class.no_of_constrained_dofs)
        if self.stress_recovery: # make sure, that current stress / strain is exported
            K_unconstr, f_unconstr, self.stress, self.strain = \
                self.assembly_class.assemble_k_f_S_E(self.unconstrain_vec(u), t)
        else:
            K_unconstr, f_unconstr = \
                self.assembly_class.assemble_k_and_f(self.unconstrain_vec(u), t)
        K = self.constrain_matrix(K_unconstr)
        f = self.constrain_vec(f_unconstr)
        return K, f

    def apply_rayleigh_damping(self, alpha, beta):
        '''
        Apply Rayleigh damping to the system. The damping matrix D is defined as D = alpha*M + beta*K(0). Thus, it is
        Rayleigh Damping applied to the linearized system around zero deformation.

        Parameters
        ----------
        alpha : float
            Damping coefficient for the mass matrix.
        beta : float
            Damping coefficient for the stiffness matrix.
        '''

        if self.M_constr is None:
            self.M()
        self.D_constr = alpha*self.M_constr + beta*self.K()
        return

    def write_timestep(self, t, u):
        '''
        Write the timestep to the mechanical_system class.
        '''

        self.T_output.append(t)
        self.u_output.append(self.unconstrain_vec(u))
        # Check both, if stress recovery and if stress and strain is there
        if self.stress_recovery:
            # catch the case when no stress was computed, for instance in time
            # integration
            if self.stress is None and self.strain is None:
                self.stress = np.zeros((self.mesh_class.no_of_nodes,6))
                self.strain = np.zeros((self.mesh_class.no_of_nodes,6))
            self.S_output.append(self.stress.copy())
            self.E_output.append(self.strain.copy())

    def clear_timesteps(self):
        '''
        Clear the timesteps gathered internally.
        '''

        self.T_output = []
        self.u_output = []
        self.S_output = []
        self.E_output = []
        self.stress = None
        self.strain = None
        self.iteration_info = np.array([])
        return

    def set_solver(self, solver, **solver_options):
        '''
        Set solver to be able to use shortcut my_system.solve() for solving the system.
        '''

        self.solver = solver(mechanical_system=self, **solver_options)
        return

    def solve(self):
        '''
        Shortcut to solve system.
        '''

        if not hasattr(self, 'solver'):
            raise ValueError('No solver set. Use my_system.set_solver(solver, **solver_options) to set solver first.')

        self.solver.solve()
        return


class MechanicalSystemStateSpace(MechanicalSystem):
    def __init__(self, regular_matrix=None, **kwargs):
        MechanicalSystem.__init__(self, **kwargs)
        if regular_matrix is None:
            self.R_constr = self.K()
        else:
            self.R_constr = regular_matrix
        self.x_red_output = []
        self.R_constr = regular_matrix
        self.E_constr = None

    def M(self, x=None, t=0):
        if x is not None:
            self.M_constr = MechanicalSystem.M(self, x[0:int(x.size/2)], t)
        else:
            self.M_constr = MechanicalSystem.M(self, None, t)
        return self.M_constr

    def E(self, x=None, t=0):
        if self.M_constr is None:
            self.M(x, t)
        self.E_constr = bmat([[self.R_constr, None],
                              [None, self.M_constr]])
        return self.E_constr

    def D(self, x=None, t=0):
        if x is not None:
            self.D_constr = MechanicalSystem.D(self, x[0:int(x.size/2)], t)
        else:
            self.D_constr = MechanicalSystem.D(self, None, t)
        return self.D_constr

    def K(self, x=None, t=0):
        if x is not None:
            K = MechanicalSystem.K(self, x[0:int(x.size/2)], t)
        else:
            K = MechanicalSystem.K(self, None, t)
        return K

    def A(self, x=None, t=0):
        if self.D_constr is None:
            A = bmat([[None, self.R_constr], [-self.K(x, t), None]])
        else:
            A = bmat([[None, self.R_constr], [-self.K(x, t), -self.D_constr]])
        return A

    def f_int(self, x=None, t=0):
        if x is None:
            x = np.zeros(2*self.dirichlet_class.no_of_constrained_dofs)
        f_int = MechanicalSystem.f_int(self, x[0:int(x.size/2)], t)
        return f_int

    def F_int(self, x=None, t=0):
        if x is None:
            x = np.zeros(2*self.dirichlet_class.no_of_constrained_dofs)
        if self.D_constr is None:
            F_int = np.concatenate((self.R_constr@x[int(x.size/2):],
                                    -self.f_int(x, t)), axis=0)
        else:
            F_int = np.concatenate((self.R_constr@x[int(x.size/2):],
                                    -self.D_constr@x[int(x.size/2):]
                                     - self.f_int(x, t)), axis=0)
        return F_int

    def f_ext(self, x, t):
        if x is None:
            f_ext = MechanicalSystem.f_ext(self, None, None, t)
        else:
            f_ext = MechanicalSystem.f_ext(self, x[0:int(x.size/2)],
                                           x[int(x.size/2):], t)
        return f_ext

    def F_ext(self, x, t):
        F_ext = np.concatenate((np.zeros(self.dirichlet_class.no_of_constrained_dofs),
                                self.f_ext(x, t)), axis=0)
        return F_ext

    def K_and_f(self, x=None, t=0):
        if x is not None:
            K, f_int = MechanicalSystem.K_and_f(self, x[0:int(x.size/2)], t)
        else:
            K, f_int = MechanicalSystem.K_and_f(self, None, t)
        return K, f_int

    def A_and_F(self, x=None, t=0):
        if x is None:
            x = np.zeros(2*self.dirichlet_class.no_of_constrained_dofs)
        K, f_int = self.K_and_f(x, t)
        if self.D_constr is None:
            A = bmat([[None, self.R_constr], [-K, None]])
            F_int = np.concatenate((self.R_constr@x[int(x.size/2):], -f_int), axis=0)
        else:
            A = bmat([[None, self.R_constr], [-K, -self.D_constr]])
            F_int = np.concatenate((self.R_constr@x[int(x.size/2):],
                                    -self.D_constr@x[int(x.size/2):] - f_int), axis=0)
        return A, F_int

    def write_timestep(self, t, x):
        MechanicalSystem.write_timestep(self, t, x[0:int(x.size/2)])
        self.x_output.append(x.copy())
        return

    def export_paraview(self, filename, field_list=None):
        x_export = np.array(self.x_output).T
        x_dict = {'ParaView':'False', 'Name':'x'}
        if field_list is None:
            new_field_list = []
        else:
            new_field_list = field_list.copy()
        new_field_list.append((x_export, x_dict))
        MechanicalSystem.export_paraview(self, filename, new_field_list)
        return

    def clear_timesteps(self):
        MechanicalSystem.clear_timesteps(self)
        self.x_output = []
        return


class ReducedSystem(MechanicalSystem):
    '''
    Class for reduced systems. It is directly inherited from MechanicalSystem. Provides the interface for an
    integration scheme and so on where a basis vector is to be chosen...

    Notes
    -----
    The Basis V is a Matrix with x = V*q mapping the reduced set of coordinates q onto the physical coordinates x. The
    coordinates x are constrained, i.e. the x denotes the full system in the sense of the problem set and not of the
    pure finite element set.

    The system runs without providing a V_basis when constructing the method only for the unreduced routines.

    Attributes
    ----------
    V : ?
        Set of basis vectors the system has been reduced with u_constr = V*q.
    V_unconstr : ?
        Extended reduction basis that is extended by the displacement coordinates of the constrained degrees of freedom.
    u_red_output : ?
        Stores the timeseries of the generalized coordinates (similar to u_output).
    assembly_type : {'indirect', 'direct'}
        Stores the type of assembly method how the reduced system is computed.
    
    Examples
    --------
    my_system = amfe.MechanicalSystem()
    V = vibration_modes(my_system, n=20)
    my_reduced_system = amfe.reduce_mechanical_system(my_system, V)
    '''

    def __init__(self, V_basis=None, assembly='indirect', **kwargs):
        '''
        Parameters
        ----------
        V_basis : ndarray, optional
            Basis onto which the problem will be projected with an Galerkin-Projection.
        assembly : str {'direct', 'indirect'}
            flag setting, if direct or indirect assembly is done. For larger reduction bases, the indirect method is
            much faster.
        **kwargs : dict, optional
            Keyword arguments to be passed to the mother class MechanicalSystem.
        '''

        MechanicalSystem.__init__(self, **kwargs)
        self.V = V_basis
        self.u_red_output = []
        self.V_unconstr = self.dirichlet_class.unconstrain_vec(V_basis)
        self.assembly_type = assembly

    def K_and_f(self, u=None, t=0):
        if u is None:
            u = np.zeros(self.V.shape[1])
        if self.assembly_type == 'direct':
            # this is really slow! So this is why the assembly is done diretly
            K, f_int = self.assembly_class.assemble_k_and_f_red(self.V_unconstr,
                                                                u, t)
        elif self.assembly_type == 'indirect':
            K_raw, f_raw = self.assembly_class.assemble_k_and_f(self.V_unconstr @ u,
                                                                t)
            K = self.V_unconstr.T @ K_raw @ self.V_unconstr
            f_int = self.V_unconstr.T @ f_raw
        else:
            raise ValueError('The given assembly type for a reduced system '
                             + 'is not valid.')
        return K, f_int

    def K(self, u=None, t=0):
        if u is None:
            u = np.zeros(self.V.shape[1])

        if self.assembly_type == 'direct':
            # this is really slow! So this is why the assembly is done diretly
            K, f_int = self.assembly_class.assemble_k_and_f_red(self.V_unconstr,
                                                                u, t)
        elif self.assembly_type == 'indirect':
            K_raw, f_raw = self.assembly_class.assemble_k_and_f(self.V_unconstr @ u,
                                                                t)
            K = self.V_unconstr.T @ K_raw @ self.V_unconstr
        else:
            raise ValueError('The given assembly type for a reduced system '
                             + 'is not valid.')
        return K

    def f_ext(self, u, du, t):
        return self.V.T @ MechanicalSystem.f_ext(self, self.V @ u, du, t)

    def f_int(self, u, t=0):

        if self.assembly_type == 'direct':
            # this is really slow! So this is why the assembly is done diretly
            K, f_int = self.assembly_class.assemble_k_and_f_red(self.V_unconstr,
                                                                u, t)
        elif self.assembly_type == 'indirect':
            K_raw, f_raw = self.assembly_class.assemble_k_and_f(self.V_unconstr @ u,
                                                                t)
            f_int = self.V_unconstr.T @ f_raw
        else:
            raise ValueError('The given assembly type for a reduced system '
                             + 'is not valid.')

        return f_int

    def D(self, u=None, t=0):
        if self.assembly_type == 'direct':
            raise NotImplementedError('The direct method is note implemented yet for damping matrices')
        elif self.assembly_type == 'indirect':
            self.D_constr = self.V.T @ MechanicalSystem.D(self, self.V @ u, t) @ self.V
        else:
            raise ValueError('The given assembly type for a reduced system '
                             + 'is not valid.')

        return self.D_constr

    def M(self, u=None, t=0):
        # Just a plain projection
        # not so well but works...
        self.M_constr = self.V.T @ MechanicalSystem.M(self, u, t) @ self.V
        return self.M_constr

    def write_timestep(self, t, u):
        MechanicalSystem.write_timestep(self, t, self.V @ u)
        self.u_red_output.append(u.copy())

    def K_unreduced(self, u=None, t=0):
        '''
        Unreduced Stiffness Matrix.

        Parameters
        ----------
        u : ndarray, optional
            Displacement of constrained system. Default is zero vector.
        t : float, optionial
            Time. Default is 0.

        Returns
        -------
        K : sparse csr matrix
            Stiffness matrix.
        '''

        return MechanicalSystem.K(self, u, t)

    def f_int_unreduced(self, u, t=0):
        '''
        Internal nonlinear force of the unreduced system.

        Parameters
        ----------
        u : ndarray
            Displacement of unreduces system.
        t : float, optional
            Time, default value: 0.

        Returns
        -------
        f_nl : ndarray
            nonlinear force of unreduced system.
        '''

        return MechanicalSystem.f_int(self, u, t)

    def M_unreduced(self):
        '''
        Unreduced mass matrix.
        '''

        return MechanicalSystem.M(self)

    def export_paraview(self, filename, field_list=None):
        '''
        Export the produced results to ParaView via XDMF format.
        '''

        u_red_export = np.array(self.u_red_output).T
        u_red_dict = {'ParaView':'False', 'Name':'q_red'}

        if field_list is None:
            new_field_list = []
        else:
            new_field_list = field_list.copy()

        new_field_list.append((u_red_export, u_red_dict))

        MechanicalSystem.export_paraview(self, filename, new_field_list)

        # add V and Theta to the hdf5 file
        filename_no_ext, _ = os.path.splitext(filename)
        with h5py.File(filename_no_ext + '.hdf5', 'r+') as f:
            f.create_dataset('reduction/V', data=self.V)

        return

    def clear_timesteps(self):
        MechanicalSystem.clear_timesteps(self)
        self.u_red_output = []


class ReducedSystemStateSpace(MechanicalSystemStateSpace):
    def __init__(self, right_basis=None, left_basis=None, **kwargs):
        MechanicalSystemStateSpace.__init__(self, **kwargs)
        self.V = right_basis
        self.W = left_basis
        self.x_red_output = []

    def E(self, x=None, t=0):
        if x is not None:
            self.E_constr = self.W.T@MechanicalSystemStateSpace.E(self, self.V@x, \
                                                                  t)@self.V
        else:
            self.E_constr = self.W.T@MechanicalSystemStateSpace.E(self, None, t)@self.V
        return self.E_constr

    def E_unreduced(self, x_unreduced=None, t=0):
        return MechanicalSystemStateSpace.E(self, x_unreduced, t)

    def A(self, x=None, t=0):
        if x is not None:
            A = self.W.T@MechanicalSystemStateSpace.A(self, self.V@x, t)@self.V
        else:
            A = self.W.T@MechanicalSystemStateSpace.A(self, None, t)@self.V
        return A

    def A_unreduced(self, x_unreduced=None, t=0):
        return MechanicalSystemStateSpace.A(self, x_unreduced, t)

    def F_int(self, x=None, t=0):
        if x is not None:
            F_int = self.W.T@MechanicalSystemStateSpace.F_int(self, self.V@x, t)
        else:
            F_int = self.W.T@MechanicalSystemStateSpace.F_int(self, None, t)
        return F_int

    def F_int_unreduced(self, x_unreduced=None, t=0):
        return MechanicalSystemStateSpace.F_int(self, x_unreduced, t)

    def F_ext(self, x, t):
        if x is not None:
            F_ext = self.W.T@MechanicalSystemStateSpace.F_ext(self, self.V@x, t)
        else:
            F_ext = self.W.T@MechanicalSystemStateSpace.F_ext(self, None, t)
        return F_ext

    def F_ext_unreduced(self, x_unreduced, t):
        return MechanicalSystemStateSpace.F_ext(self, x_unreduced, t)

    def A_and_F(self, x=None, t=0):
        if x is not None:
            A_, F_int_ = MechanicalSystemStateSpace.A_and_F(self, self.V@x, t)
        else:
            A_, F_int_ = MechanicalSystemStateSpace.A_and_F(self, None, t)
        A = self.W.T@A_@self.V
        F_int = self.W.T@F_int_
        return A, F_int

    def A_and_F_unreduced(self, x_unreduced=None, t=0):
        return MechanicalSystemStateSpace.A_and_F(self, x_unreduced, t)

    def write_timestep(self, t, x):
        MechanicalSystemStateSpace.write_timestep(self, t, self.V@x)
        self.x_red_output.append(x.copy())
        return

    def export_paraview(self, filename, field_list=None):
        x_red_export = np.array(self.x_red_output).T
        x_red_dict = {'ParaView':'False', 'Name':'x_red'}
        if field_list is None:
            new_field_list = []
        else:
            new_field_list = field_list.copy()
        new_field_list.append((x_red_export, x_red_dict))
        MechanicalSystemStateSpace.export_paraview(self, filename, new_field_list)
        return

    def clear_timesteps(self):
        MechanicalSystemStateSpace.clear_timesteps(self)
        self.x_red_output = []
        return


def reduce_mechanical_system(
        mechanical_system, V,
        overwrite=False,
        assembly='indirect'):
    '''
    Reduce the given mechanical system with the linear basis V.

    Parameters
    ----------
    mechanical_system : instance of MechanicalSystem
        Mechanical system which will be transformed to a ReducedSystem.
    V : ndarray
        Reduction Basis for the reduced system
    overwrite : bool, optional
        Switch, if mechanical system should be overwritten (is less memory intensive for large systems) or not.
    assembly : str {'direct', 'indirect'}
        flag setting, if direct or indirect assembly is done. For larger reduction bases, the indirect method is much
        faster.

    Returns
    -------
    reduced_system : instance of ReducedSystem
        Reduced system with same properties of the mechanical system and reduction basis V.
    '''

    if overwrite:
        reduced_sys = mechanical_system
    else:
        reduced_sys = copy.deepcopy(mechanical_system)
    reduced_sys.__class__ = ReducedSystem
    reduced_sys.V = V.copy()
    reduced_sys.V_unconstr = reduced_sys.dirichlet_class.unconstrain_vec(V)
    reduced_sys.u_red_output = []
    reduced_sys.M_constr = None
    # reduce Rayleigh damping matrix
    if reduced_sys.D_constr is not None:
        reduced_sys.D_constr = V.T @ reduced_sys.D_constr @ V
    reduced_sys.assembly_type = assembly
    return reduced_sys


def convert_mechanical_system_to_state_space(
        mechanical_system,
        regular_matrix=None,
        overwrite=False):
    if overwrite:
        sys = mechanical_system
    else:
        sys = copy.deepcopy(mechanical_system)
    sys.__class__ = MechanicalSystemStateSpace
    sys.x_output = []
    if regular_matrix is None:
        sys.R_constr = sys.K()
    else:
        sys.R_constr = regular_matrix
    sys.E()
    return sys


def reduce_mechanical_system_state_space(
        mechanical_system_state_space, right_basis,
        left_basis=None,
        overwrite=False):
    if overwrite:
        red_sys = mechanical_system_state_space
    else:
        red_sys = copy.deepcopy(mechanical_system_state_space)
    red_sys.__class__ = ReducedSystemStateSpace
    red_sys.V = right_basis.copy()
    if left_basis is None:
        red_sys.W = right_basis.copy()
    else:
        red_sys.W = left_basis.sopy()
    red_sys.x_red_output = []
    red_sys.E_constr = None
    return red_sys


class MechanicalAssembly(MechanicalSystem):
    '''
    Master class is inherited from mechanical systems.
    The goal is to handle multidomain "multiple meshes"
    with multiple constraints intra and extra domaina
   
    '''

    domain_counter = 1

    def __init__(self):
        super().__init__()

        self.domain_dict = {}
        self.assembly_dict = {}
        self.no_of_dofs_per_node_dict = {}
        self.domain_key_list = []
        self.bonded_interface_constraint_list = []
        self.global_to_local_node_dict = {}
        self.local_to_global_node_dict = {}
        self.last_column_id_in_dataframe = None
        self.neumann_submesh = []
        self.dirichlet_submesh = []
        self.domain_key_already_appended_list = []
        self.max_node_id = 0
        self.max_num_of_partitions = 0
        self.max_num_of_elem = 0
        self.node_idx = 0

    def append_domain(self, submesh, material, key=None):

        if key is None:
            key =  MechanicalAssembly.domain_counter 
            MechanicalAssembly.domain_counter += 1

        self.domain_key_list.append(key)
        submesh.set_material(material)
        self.domain_dict[key] = copy.deepcopy(submesh)

        return self.domain_dict
    
    def get_domain(self, key):
        return self.domain_dict[key]

    def get_submesh(self,domain_key,group_key,group_tag='phys_group'):
        ''' This function return a submesh from a domain and a phys_group

        parameters:
            key : int
                Domain key
            key_group : int
                key of a mesh group
            group_tag : str
                string with the name of the group to be selected

        return 
            submesh : SubMesh obj
                submesh of a domain and group

        '''
        domain = self.get_domain(domain_key)
        submesh = domain.parent_mesh.get_submesh(group_tag,group_key)
        submesh.split_in_partitions('domain')

        return submesh.groups[domain_key]

    def update_global_system(self):
        ''' This function create a new data frame
        with the multiple domains, updates domains
        meshes and submeshes which can be useful for
        global assembly methods.
        '''
        self.create_assembly_dataframe()
        self.create_mesh_obj()
        self.update_submesh_with_global_mesh()
        self.assembly_class = Assembly(self.mesh_class)


    def create_mesh_obj(self):
        ''' this method create a global mesh object based on 
        self.domain_dict

        return
            mesh_obj : Mesh object
        '''
        self.mesh_class = Mesh()
        self.mesh_class.el_df = self.el_df
        self.mesh_class.node_idx = self.node_idx
        self.mesh_class.no_of_dofs_per_node = self.no_of_dofs_per_node 
        self.mesh_class.nodes = self.nodes
        
        return self.mesh_class

    def update_submesh_with_global_mesh(self):
        ''' This method update the submesh in domain_dict
        with the new global mesh generate by create_mesh_obj
        '''
        global_index = self.el_df.index.tolist()
        for key in self.domain_key_list:
            domain = self.get_domain(key)
            domain.parent_mesh = copy.deepcopy(self.mesh_class)
            try:
                subset = self.el_df['domain'] == key
                global_index = np.array([i for i,bool in enumerate(subset) if bool])
                domain.elements_list = list(global_index[domain.elements_list])
            except:
                None
            domain.subset_list() # update elements
            domain.create_node_list() # updates nodes

    def get_global_mesh(self):
        self.mesh_class._update_mesh_props()
        return self.mesh_class

    def apply_dirichlet_boundaries(self, submesh, value, direction='xyz'):
        ''' Function that apply Dirichlet boundary condition to a specific
        domain based on the Key of the domain.

        parameters:
            key : int
                key of the domain_dict which contains multiple domains
            submesh : SubMesh obj
                SubMesh object with elements where the boundary cond. will be applied
            value : float
                value of the boundary condition, only 0.0 is availible
            dir : str
                direction of the boundary condition

        return:
            new domain_dict with boundary condition in the domain key


        '''

        if abs(value) == 0.0:
            # Objecs for FETI method
            dir_sub = Boundary(submesh,value,direction,'dirichlet')
            #domain.append_bondary_condition(dir_sub)

            self.dirichlet_submesh.append(dir_sub)

            return self.dirichlet_submesh

        else:
            raise('Dirichlet boundary cond. difference from 0.0 is not supported')


    def apply_neumann_boundaries(self, submesh, value, direction='normal'):
            
            #domain = self.get_domain(key)
            new_sub = Boundary(submesh,value,direction)
            #domain.append_bondary_condition(new_sub)

            self.neumann_submesh.append(new_sub)
            return self.neumann_submesh
    
    def create_assembly_dataframe(self):
        ''' This function create a dataframe including all
        domain in self.domain_dict
        '''
                
        count = 0
        max_node_id = self.max_node_id
        max_num_of_partitions = self.max_num_of_partitions
        max_num_of_elem = self.max_num_of_elem

        # create list of a difference of already append domains
        s = set(self.domain_key_already_appended_list)
        t = set(self.domain_key_list)
        new_domain_list = list(t.difference(s))
        
        for key in new_domain_list:
            domain = self.get_domain(key)
            df = copy.deepcopy(domain.parent_mesh.el_df)
            self.node_idx = domain.parent_mesh.node_idx      
            node_idx = self.node_idx
            # convert all node numbers to integers
            df.iloc[:,node_idx :] = df.iloc[:,node_idx :].fillna(-1)
            df.iloc[:,node_idx :] = df.iloc[:,node_idx  :].astype(np.int64)

            # renumbering nodes
            df.iloc[:,node_idx  :] += max_node_id
            df.iloc[:,node_idx  :] = df.iloc[:,node_idx :].replace(-1 + max_node_id, np.nan)

            max_node_id += domain.parent_mesh.no_of_nodes
            if 'partition_id' in df.columns:
                # renumbering partitions and it neighbors
                num_of_partitions = df['partition_id'].max()
                df['partition_id'] += int(max_num_of_partitions)

                # apply renumbering in neighbors list
                for j,nei_list in enumerate(df['partitions_neighbors']):
                    if isinstance(nei_list,list) and max_num_of_partitions>0:
                        #df['partitions_neighbors'].iloc[j] = list(np.array(nei_list) - max_num_of_partitions)
                        #df['partitions_neighbors'][j] = list(np.array(nei_list) - max_num_of_partitions)
                        df.at[j,('partitions_neighbors')] = list(np.array(nei_list) - max_num_of_partitions)
            else:
                num_of_partitions = 1
                df.insert(node_idx ,'no_of_mesh_partitions', num_of_partitions)
                df.insert(node_idx + 1,'partition_id', max_num_of_partitions + num_of_partitions)
                df.insert(node_idx + 2 ,'partitions_neighbors',None)
                node_idx += 3
                
            max_num_of_partitions += num_of_partitions
            
            if 'domain' not in df.columns:
                # add extra column with domain
                df.insert(node_idx ,'domain',key)
                self.node_idx = node_idx + 1
            else:
                df['domain'] = key

            
            if 'local_idx' not in df.columns:
                df.insert(1 ,'local_idx', df.iloc[:,0].values)
                self.node_idx = self.node_idx + 1

            # renumbering elem
            num_of_elem = df.iloc[:,0].max()
            df.iloc[:,0] += max_num_of_elem
            
            # update max number of elements
            max_num_of_elem +=num_of_elem

            if count==0:
                self.el_df = copy.deepcopy(df)
                self.nodes = copy.deepcopy(domain.parent_mesh.nodes)
                count = 1
            else:
                self.el_df = self.el_df.append(df, ignore_index=True)
                self.nodes = np.vstack([self.nodes, copy.deepcopy(domain.parent_mesh.nodes)])

        if not self.domain_key_already_appended_list:
            self.no_of_dofs_per_node = domain.parent_mesh.no_of_dofs_per_node
            self.last_column_id_in_dataframe = len(self.el_df.columns)

        # update global variables
        self.max_node_id = max_node_id
        self.max_num_of_partitions = max_num_of_partitions
        self.max_num_of_elem = max_num_of_elem

        self.domain_key_already_appended_list.extend(new_domain_list)
        return self.el_df
        
    def split_in_partitions(self,group_tag='partition_id'):
        ''' This methods applies partition to mesh
        it uses the split_partitions_implemented in Submesh
        '''

        try:
            domain = self.domain
        except:
            domain = self.update_domain()

        self.groups = {}
        for key in self.domain_key_list:
            domain = self.get_domain(key)
            domain.split_in_partitions(group_tag)
            for domain_key in domain.groups:
                if domain_key not in self.groups:
                    self.groups.update({domain_key: domain.groups[domain_key]})

        return self.groups

    def update_domain(self, preallocate=False):
        ''' create the self.domain variables
        '''
        self.update_submesh_with_global_mesh()
        
        count = 0
        for key in self.domain_key_list:
            domain = self.get_domain(key)
            if count > 0:
                elem_dataframe = elem_dataframe.append(domain.elem_dataframe)
            else:    
                elem_dataframe = copy.deepcopy(domain.elem_dataframe)
                count += 1
        
        for key in self.domain_key_list:
            domain = self.get_domain(key)
            domain.elem_dataframe = elem_dataframe 

            for neu_sub in self.neumann_submesh:
                domain.append_bondary_condition(neu_sub)

            for dir_sub in self.dirichlet_submesh:
                domain.append_bondary_condition(dir_sub)
        
            self.mesh_class.compute_connectivity_and_add_material(domain.elem_dataframe, domain.__material__)
            self.no_of_dofs_per_node = self.mesh_class.no_of_dofs_per_node

            if preallocate:
                self.assembly_class.compute_element_indices()
            else:
                self.assembly_class.preallocate_csr()

        #elem_dataframe = elem_dataframe.reset_index(drop=True)
        #elem_dataframe = self.el_df
        
        num_of_elem = len(self.el_df)

        for neu_sub in self.neumann_submesh:
            domain.append_bondary_condition(neu_sub)

        for dir_sub in self.dirichlet_submesh:
            domain.append_bondary_condition(dir_sub)

        self.update_boundary_submesh()

        self.domain = self
        return self

    def update_boundary_submesh(self):
        ''' This method updates boundary submeshes based on 
        new el_df because to renumbering of nodes
        '''

        
        elem_start_index = self.node_idx
        elem_last_index = self.last_column_id_in_dataframe
        bc_list = [self.neumann_submesh, self.dirichlet_submesh]

        for submesh_item in bc_list:
            for i,submesh_neu in enumerate(submesh_item):
                nodes = []
                for elem in submesh_neu.elements_list:
                    nodes.extend(list(self.el_df.iloc[elem,elem_start_index:elem_last_index].dropna(0).astype(int)))

                submesh_neu.submesh.parent_mesh = self.mesh_class
                submesh_neu.submesh.elem_dataframe = self.el_df.iloc[submesh_neu.elements_list,:]
                submesh_neu.submesh.global_node_list = list(set(nodes))
                val = submesh_neu.value 
                direction = submesh_neu.direction
                typeBC = submesh_neu.type
            
                new_submesh = Boundary(submesh_neu.submesh, val, direction, typeBC )
                submesh_item[i] = new_submesh


    def add_bonded_interface_constraint(self,submesh1,submesh2, tol = 1.E-6):
        ''' This function connects the nodes from submesh1 on 
        submesh2

        create a dictionary with constraint pares and
        add to the self.bonded_interface_constraint_list
        which contains all the constraints

        This constraint can be used for dual or primal assembly
        
        parameters:
            submesh1 : SubMesh obj
                submesh of the first subdomain 
            submesh2  : SubMesh obj
                submesh of the second subdomain 


        return 
            bound_interface_constraint_dict : dict
                dictionary with interface par, can be used to map global 
                to local node numbers

        '''

        domain_key_1 = submesh1.elem_dataframe['domain'].iloc[0]
        domain_key_2 = submesh2.elem_dataframe['domain'].iloc[0]

        interface_elem_dict_1 = self.find_partition_in_domain_by_elem(domain_key_1,submesh1.elements_list)

        if not interface_elem_dict_1:
            print('Not possible to apply Bonded Interfaces for the given SubMesh object')
            return None

        interface_elem_dict_2 = self.find_partition_in_domain_by_elem(domain_key_2,submesh2.elements_list)

        if not interface_elem_dict_2:
            print('Not possible to apply Bonded Interfaces for the given SubMesh object')
            return None

        int_dict = {domain_key_1:submesh1, domain_key_2:submesh2}
        self.bonded_interface_constraint_list.append(int_dict)

        # update dataframe
        for partition_key_2 in interface_elem_dict_1:
            elem_interface_1 = interface_elem_dict_1[partition_key_2]
            for partition_key_1 in interface_elem_dict_2:
                elem_interface_2 = interface_elem_dict_2[partition_key_1]
                self.update_partitions_neighbors_in_dataframe(elem_interface_2,partition_key_2)
                self.update_partitions_neighbors_in_dataframe(elem_interface_1,partition_key_1)

        # update nodes in dataframe
        node_at_interface = self.replace_nodes_in_dataframe(submesh1,submesh2,tol)
        self.update_submesh_with_global_mesh()

        return int_dict, node_at_interface 

    def replace_nodes_in_dataframe(self,submesh1,submesh2, tol = 1.0E-6):
        ''' This function replace the nodes in submesh 2 in the dataframe by 
        nodes in SubMesh 1.
        This will result that subdomain will share the same indexes 
        of nodes whith is import for the split method, which generetes partitions

        This method also update the self.global_to_local_node_dict which map
        old global nodes to new node index


        paramenters:
            submesh1 : SubMesh obj
                SubmMesh which the nodes will be preserved
            submesh2 : SubMesh obj
                SubmMesh which the nodes will be replaced
            tol : float 
                 tolerance for node distance
        return:
            nodes_at_interface : int
                number of nodes in the interface

        '''

        nodes_in_1 =  submesh1.global_node_list
        nodes_in_2 =  submesh2.global_node_list
        nodes_at_interface = 0
        global_to_local_node_dict = {}
        local_to_global_node_dict = {}
        small_tolerance = 1E8

        for j,node_id in enumerate(nodes_in_2): 
            node_coord_1 = self.nodes[node_id]
            # try to find global index based on node coord
            for k,global_node_id in enumerate(nodes_in_1):
                node_coord_2 = self.nodes[global_node_id]
                node_dist = np.linalg.norm(node_coord_1 - node_coord_2)
                if node_dist<small_tolerance:
                    small_tolerance = node_dist
                if node_dist<tol:
                    global_to_local_node_dict[global_node_id] = node_id
                    local_to_global_node_dict[node_id] = global_node_id
                    nodes_at_interface += 1


        if len(global_to_local_node_dict.keys())==0:
            print('It was not possible to find node pairs given the tolerance parameter.\n' + \
                  'Please make sure that the select Edges or Faces in the tolerance range.\n' + \
                  'The smallest gap found was %5.5E' %small_tolerance)
        elif len(global_to_local_node_dict.keys())<len(nodes_in_1):
            print('WARNING! Not possible to match all nodes in the given set.\n' + \
                  'The smallest gap found was %5.5E' %small_tolerance)

        # update global dict
        self.global_to_local_node_dict.update(global_to_local_node_dict)
        self.local_to_global_node_dict.update(local_to_global_node_dict)

        # update the remaining columns
        column_list = list(np.arange(self.node_idx, self.last_column_id_in_dataframe))
        self.replace_dataframe_columns(column_list,self.local_to_global_node_dict)

        return nodes_at_interface

    def change_domain_physical_tag(self, domain_key, current_phys_key, new_phys_key):
        ''' change in pandas dataframe "self.el_df" the physical_id "phys_group" 
        in domain equal "domain_key" for a new physical key equal new_phys_key

        parameters
            domain_key : int or list
                domain key to modify

            current_phys_key : int
                current physical key to be modified
            
            new_phys_key : int
                new physical key

        return 
            self.el_df : Pandas DataFrame
                pandas dataframe with mesh information
        '''
        
        if isinstance(domain_key,int):
            domain_list = [domain_key]
        else:
            domain_list = domain_key
        
        for domain_id in domain_list:
            phys_col =  self.el_df.columns.get_loc("phys_group")
            dict_map = {current_phys_key:new_phys_key}
            domain_rows =  self.el_df[self.el_df["domain"] == domain_id].index
            self.replace_dataframe_columns(phys_col,dict_map, domain_rows)

        return self.el_df

    def replace_dataframe_columns(self,list_of_column_index,dict_map, rows = None):
        ''' This function replace values in the self.el_df based on column_index
        and old value.

        parameters:
            column_index : list
                lsit with number of column index
            dict_map : dict
                dict [old_value] = new_value
                old_value : int or float
                    old value to be replaced
                new_value : int or float
                    new value to replace old value

        return:
            el_df : pandas dataframe
                pandas dataframe with new value in column number
                equal column_index

        '''
        if rows is None:
            new_df = self.el_df.iloc[:,list_of_column_index].replace(dict_map)
            self.el_df.iloc[:,list_of_column_index] = new_df
        else:
            new_df = self.el_df.iloc[rows,list_of_column_index].replace(dict_map)
            self.el_df.iloc[rows,list_of_column_index] = new_df

        return self.el_df

    def update_partitions_neighbors_in_dataframe(self,elem_list,partitions_neighbors):
        ''' update partitions neighbor in dataframe "self.el_df"
        '''
        for elem in elem_list:
            if self.el_df.loc[elem,'partitions_neighbors'] is None:
                self.el_df.loc[elem,'partitions_neighbors'] = [-partitions_neighbors]
            else:
                if -partitions_neighbors not in self.el_df.loc[elem,'partitions_neighbors']:
                    self.el_df.loc[elem,'partitions_neighbors'].append( -partitions_neighbors)


    def find_partition_in_domain_by_elem(self,domain_key,elem_list):
        ''' This function finds the partition_id of a elem "elem_num"
        which has nodes that are in a domain which has key "domain_key"

        parameters:

        domain_key : int
            domain key to look for a partition
        elem_num : int
            element number based on self.el_df

        return 
             interface_elem_dict
                where the partition_id is the key and the value 
                is a list of elements

        '''
        
        domain = self.get_domain(domain_key)
        p_id = None
        slice_dataframe = self.el_df.iloc[:,self.node_idx:].to_dict('index')
        interface_elem_dict = {}
        for elem_num in elem_list:
            nodes_in_elem_target = list(slice_dataframe[elem_num].values())
            for elem in domain.elements_list:
                nodes_in_domain =  list(np.array(list(slice_dataframe[elem].values())))
                intersection_set = set(nodes_in_elem_target).intersection(nodes_in_domain)

                if len(intersection_set)>1:
                    bool = True
                elif len(intersection_set)==1 and np.nan not in intersection_set:
                    bool = True
                else:
                    bool = False
                
                if bool:
                    p_id = int(self.el_df.loc[elem,['partition_id']])
                    self.el_df.loc[elem_num,['partition_id']] = p_id
                    try:
                        interface_elem_dict[p_id].append(elem)
                    except:
                        interface_elem_dict[p_id] = []
                        interface_elem_dict[p_id].append(elem)


        return interface_elem_dict


    def add_linear_equatity_node_const(self, const_obj1, const_obj2, value=0.0, dof_tag = 'xyz'):
        ''' This function add constraint objects defined in two domain and
        in the mechanical assembly:

        decouple system
        K1 u1 = f1
        K2 uf = f2

        couple system with node constraint

        K1 u1 = f1 + B1*lambda
        K2 u2 = f2 + B2*lambda
        B1 + B2 = value

        arguments:
        
            const_obj1 : as Constraint class
                constaint object with a list of nodes in a domain

            const_obj2 : as Constraint class
                constaint object with a list of nodes in a domain

            value : float or list
                float or list with values of the constraint

            dof_tag: str or list
                str or list with the DOFs to be considered for the constraint
        '''

        # changing string dof_tag to local_dof_list
        if isinstance(dof_tag,str):
            local_dof_list = [0,0,0]
            tag_list = ['x','y','z']
            for i,tag in enumerate(tag_list):
                if tag in dof_tag:
                    local_dof_list[i]
        else:
            local_dof_list = dof_tag

        return None
        


class CraigBamptonComponent(MechanicalSystem):
    
    def __init__(self):
        super().__init__()
    
        self.M_local = None
        self.K_local = None
        self.T_local = None
        self.T = None
        self.P = None
        self.red2globaldof = None
        self.global2reddof = None
        self.P_cyclic = None # Permutation of cyclic symmetry condition
        self.num_cyclic_dofs = 0
        
    def compute(self,M, K, master_dofs, slave_dofs, no_of_modes=5):
        ''' compute the craig bampton reduction
        Computes the Craig-Bampton basis for the System M and K with the input
        Matrix b.

        Parameters
        ----------
        M : ndarray
            Mass matrix of the system.
        K : ndarray
            Stiffness matrix of the system.
            master_dofs : ndarray
            input with dofs of master nodes
        slave_dofs : ndarray
            input with dofs of slave nodes
        no_of_modes : int, optional
            Number of internal vibration modes for the reduction of the system.
            Default is 5.
        one_basis : bool, optional
            Flag for setting, if one Craig-Bampton basis should be returned or if
            the static and the dynamic basis is chosen separately

        Returns
        -------
        if `one_basis=True` is chosen:

        V : array
            Basis consisting of static displacement modes and internal vibration
            modes

        if `one_basis=False` is chosen:

        V_static : ndarray
            Static displacement modes corresponding to the input vectors b with
            V_static[:,i] being the corresponding static displacement vector to
            b[:,i].
        V_dynamic : ndarray
            Internal vibration modes with the boundaries fixed.
        omega : ndarray
            eigenfrequencies of the internal vibration modes.

        Examples
        --------
        TODO

        Notes
        -----
        There is a filter-out command to remove the interface eigenvalues of the
        system.

        References
        ----------
        TODO

        '''
        # boundaries
        ndof = M.shape[0]       
        K_tmp = K.copy()
        
        K_bb = K_tmp[np.ix_(master_dofs, master_dofs)]
        K_ii = K_tmp[np.ix_(slave_dofs,  slave_dofs)]
        K_ib = K_tmp[np.ix_(slave_dofs,  master_dofs)]
        Phi = splinalg.spsolve(K_ii,K_ib)
        
        K_local = self.build_local(K_bb,K_ii,K_ib)

        # inner modes
        M_tmp = M.copy()
        # Attention: introducing eigenvalues of magnitude 1 into the system
        M_bb = M_tmp[np.ix_(master_dofs, master_dofs)]
        M_ii = M_tmp[np.ix_(slave_dofs,  slave_dofs)]
        M_ib = M_tmp[np.ix_(slave_dofs,  master_dofs)]
        
        M_local = self.build_local(M_bb,M_ii,M_ib)

        num_of_masters = len(master_dofs)
        num_of_slaves = ndof - num_of_masters
        
        if no_of_modes>num_of_slaves:
            no_of_modes = num_of_slaves-1
            print('Replacing number of modes to %i' %no_of_modes)
            
        omega, V_dynamic = splinalg.eigsh(K_ii, no_of_modes, M_ii)


        I = np.identity(num_of_masters)
        Zeros = np.zeros( (num_of_masters, no_of_modes))

        T_local_row_1 = np.hstack((I,Zeros))
        T_local_row_2 = np.hstack((Phi.todense(),V_dynamic))
        T_local = np.vstack((T_local_row_1,T_local_row_2))
        
        local_indexes = []
        local_indexes.extend(master_dofs)
        local_indexes.extend(slave_dofs)

        P = sparse.csc_matrix((ndof, ndof), dtype=np.int8)
        P[local_indexes, np.arange(ndof)] = 1


        T = P.dot(T_local)
        

        #omega = np.sqrt(omega)
        self.M_local = M_local
        self.K_local = K_local
        self.T_local = T_local
        self.T = T
        self.P = P
        
        return T, T_local, P, K_local, M_local

    def build_local(self,M_bb,M_ii,M_ib):
        return sparse.vstack((sparse.hstack((M_bb,M_ib.T)), sparse.hstack((M_ib,M_ii)))).tocsc()

    def local2global(self,M,P):

        return P.dot(M).dot(P)

    def check_symmetry(self,M):
        dif = abs(M - M.T)
        if dif.max() > 0.0:
            return False
        else:
            return True
    
    def create_mapping_dict(self,global_id, local_id):
        ''' This function creates 2 dicts to mapping
        local dofs to global dofs and vice and versa
        
        parameters:
            global_id : list
                list with global dofs
            
            local_id : list
                list with global dofs
        return
            gloal2local: dict
            local2global: dict
            
        '''
        gloal2local = {}
        local2global = {}
        
        for local_i,global_i in zip(local_id,global_id):
            gloal2local[global_i] = local_i
            local2global[local_i] = global_i
        
        return gloal2local, local2global
        
        
    def get_reduced_ciclic_symm_system(self, K, M, f, num_of_cyclic_dofs):
        ''' This function gets the reduced system of the global cyclic system operators
        
            
        Global assembly cyclic matrices
        
        
        | I               -R                                0         |
        | 0     (Khh + Kii +R*Khl + R^-1*Klh )      (Khi + R^-1*Kli)  |
        | 0             (Kih + R*Kil)                       Kii       |  
        
        Reduced assembly cyclic matrices
        
        | (Khh + Kii +R*Khl + R^-1*Klh )      (Khi + R^-1*Kli)  |
        | (Kih + R*Kil)                       Kii       |
        
        
        parameters
        
        K : sparse_matrix
        
        M : sparse_matrix, 
        
        f : np.array
        
        num_of_cyclic_dofs : int
        
        returns
        
        K_red : sparse_matrix
        
        M_red : sparse_matrix, 
        
        f_red : np.array
        
        
        '''
        pass
        
    def insert_dirichlet_boundary_cond(self, K=None, M = None, f=None, dir_dof = [], value = 0.0, reduced=False):
        '''
         This function inserts dirichlet B.C.
         
         parameters:
         
         
        
        '''
            
        dirichlet_stiffness = 1.0E10
        #dirichlet_stiffness = 1.0
        dir_force = np.zeros(len(f))    
        for dof in dir_dof:
            # dir_force = -value*K[:,dof]
            K[dof,:] = 0.0
            #K[:,dof] *= 0.0
            K[dof,dof] = dirichlet_stiffness
            dir_force[dof] = value*dirichlet_stiffness
            
            if M is not None:
                M[dof,:] *= 0.0
                M[:,dof] *= 0.0
                M[dof,dof] *= 0.0
                #M[dof,dof] = 1.0
            
        if abs(value)>0.0:
            print('Dirichlet boundary with generate a non-symetric Stiffness Matrix')
        else:
            # generate a symmetric matrix for zero dirichlet B.C = 0.0
            for dof in dir_dof:
                K[:,dof] *= 0.0
                K[dof,dof] = dirichlet_stiffness
                
            
        f += dir_force
        
        if M is None:
            return K,f    
        else:
            return K,M,f    

    def create_permutation_matrix(self,local_indexes):
        ''' create a Permutation matrix based on local id
        
        '''
        ndof = len(local_indexes)
        P = sparse.csc_matrix((ndof, ndof), dtype=np.int8)
        P[local_indexes, np.arange(ndof)] = 1
        return P.T
    
    def create_selection_operator(self, i_indexes, K, f=None, remove = False):
        ''' This function creates a selection operators for columns and rows
        and also creates its inverse operator,
        
        given a set of row position, a selection operator B will be created 
        and also its inverse operator.
        
        Lets assume that out matrix K after a Permutation P has the format below
        
        K = [Kii  Kib
             Kbi Kbb]
        
        Kii is the selected portion of K
        Deleted_operator = Kbb
        DiagonalOperator = Kbi = Kib
        
        
        after create B 
        Kii = S(K) = B * K * B.T = (B * (B * K).T).T
        K = Sinv(Kii) = Binv *Kii * BinvT
        
        also can be seen as
        Kii = S(K) * B.T
        K = Binv *Kii * BinvT
        
        We also assume that ub can be calculated by ui uisng:
        ub = (Kbb)^-1(Kbi)(ui)
        
        The we also provide the T operator
         ui = |I        0     | (ui)
         ub   |0 (Kbb)^-1(Kbi)|

         Which is the right operator only with the selector operation is applied on system with the folloing form
         K u = f
         
         K =   [Kii   0 
               Kbi Kbb]
         
        u = [ ui
              ub]
        
        f = [fi
             0]
        
        This implies that ui = Kii^(-1)(fi)
        ub = (Kbb)^-1(Kbi)(ui)
         
        
        
        parametes
        
            i_indexes : list
                list with values to be keep or removed depending on the variable
            
            K : scipy.sparse matrix
                matrix to be selected
            
            f : np.array
                right hand side of the system Ku = f
            
            remove : Boolean
                default value = False
                if false Kii will be select, if true Kbb will be select
                
        return 
            Kii : sparse.matrix
            S : lambda function
                Boolen selecton
            S_inv : lambda function
                expansiton operator
            T : lambda function
                linear operator that maps ui into ub
        usage
        >>>  Kii, S, S_inv = obj.create_selection_operator(i_indexes, K)
        >>>  (Kii - S(K)).max() == 0
        >>>  (K - S_inv(Kii)).max() == 0
        >>>   u = T(ui) 
        
        '''
        
        ndof , g = K.shape
        n_i = len(i_indexes)
        n_b = ndof - n_i
        all_id = list(range(ndof))
        b_indexes = list(set(all_id).difference(i_indexes))
        local_indexes = []
        local_indexes.extend(i_indexes)
        local_indexes.extend(b_indexes)
        
        if remove:
            b_indexes, i_indexes = i_indexes, b_indexes
            n_i, n_b = n_b, n_i

        P = self.create_permutation_matrix(local_indexes)
        Kp = (P.dot(K)).dot(P.T).todense()
        
        Kii = K[np.ix_(i_indexes, i_indexes)]
        Kib = K[np.ix_(i_indexes, b_indexes)]
        Kbb = K[np.ix_(b_indexes, b_indexes)]
        Kbi = K[np.ix_(b_indexes, i_indexes)]
        
        B = sparse.csc_matrix((n_i, ndof), dtype=np.int8)
        B[np.arange(n_i), i_indexes] = 1

        S = lambda K : (B.dot(K)).dot(B.T)
        
        S_inv = lambda Kii : P.T.dot(sparse.vstack((sparse.hstack(( Kii , Kib )) , sparse.hstack((Kbi , Kbb))))).dot(P)
        
        
        if abs(Kib).max() != 0.0:
            print('Transformation matrix T(ui) is not right, please do not use it!!!')

        #ub = lambda ui : splinalg.spsolve(Kbb,Kbi.dot(ui))
        #T = lambda ui : P.T.dot( np.concatenate((ui, ub(ui)), axis=-1) )
        
        
        map_obj = MapInt2Global(Kbb, Kbi, P.T)
        T = map_obj.build_map_function()
        
        if f is None:
            return Kii, S, S_inv, T
        else:
            print('returning reduced right side, please umpack 5 elementes')
            return Kii, S, S_inv, T, f[i_indexes]
            
    def insert_cyclic_symm_boundary_cond(self, K=None, M = None, f=None, low_dofs = [], high_dofs = [], theta = 0.0):            
        ''' This function modify the system operators in order to solve cyclic symmetry problems
        
        ''' 
        if theta > 0.0:
            eitheta = np.exp(np.complex(0,theta))
            inv_eitheta = np.exp(-np.complex(0,theta))
        else:
            eitheta = 1.0
            inv_eitheta = 1.0
        
        if self.check_symmetry(K):
            print('Stiffness matrix is not symmetric, the symetric cyclic may not work propertily')
        
        if len(low_dofs) != len(high_dofs):
            raise('Number of low nad high dofs must be the same')
        
        ndof,a = K.shape
        all_dofs_list = list(range(ndof))
        
        local_id = []
        local_id.extend(low_dofs)
        local_id.extend(high_dofs)
        interior_dofs = list(set(all_dofs_list).difference(local_id))
        local_id.extend(interior_dofs)
        
        if len(local_id) != ndof:
            raise('Inconsistency in cyclic symmetry dofs.')
        
        P = self.create_permutation_matrix(local_id)
        
        n_symmetry = len(low_dofs)
        n_interior = len(interior_dofs)
        
        
        K_mod = self.assembly_cyclic_matrix(K, low_dofs, high_dofs, interior_dofs , eitheta, inv_eitheta)
        M_mod = self.assembly_cyclic_matrix(M, low_dofs, high_dofs, interior_dofs , eitheta, inv_eitheta)
        f_mod = P.dot(f)
        
        return K_mod, M_mod, f_mod, P
    
    def assembly_cyclic_matrix(self,K, low_dofs, high_dofs, interior_dofs , eitheta, inv_eitheta):
        ''' assembly cyclic matrices
        
        
        | I               -R                                0         |
        | 0     (Khh + Kii +R*Khl + R^-1*Klh )      (Khi + R^-1*Kli)  |
        | 0             (Kih + R*Kil)                       Kii       |  
        
        '''
        n_symmetry = len(low_dofs)
        n_interior = len(interior_dofs)
        
        Kii = K[np.ix_(interior_dofs, interior_dofs)]
        Khh = K[np.ix_(high_dofs, high_dofs)]
        Kll = K[np.ix_(low_dofs, low_dofs)]
        Khi = K[np.ix_(high_dofs, interior_dofs)]
        Kli = K[np.ix_(low_dofs, interior_dofs)]
        Khl = K[np.ix_(high_dofs, low_dofs)]
        
        I = sparse.csc_matrix((n_symmetry, n_symmetry))
        I[np.arange(n_symmetry),np.arange(n_symmetry)] = 1.0 
        
        R = eitheta*I
        R_inv = inv_eitheta*I
        
        
        Zeros1 = sparse.csc_matrix((n_symmetry, n_symmetry))
        Zeros2 = sparse.csc_matrix((n_symmetry,n_interior))
        
        K_mod_row_1 = sparse.hstack((I,-R,Zeros2))
        K_mod_row_2 = sparse.hstack((Zeros1,Khh + Kll + R.dot(Khl) + R_inv.dot(Khl.T), Khi + R_inv.dot(Kli)))
        K_mod_row_3 = sparse.hstack((Zeros2.T,Khi.T + R.dot(Kli).T,Kii))
        
        K_mod = sparse.vstack((K_mod_row_1,K_mod_row_2,K_mod_row_3))
        
        return K_mod
        
  
class MapInt2Global():
    '''
        We also assume that ub can be calculated by ui uisng:
        ub = (Kbb)^-1(Kbi)(ui)
        
        The we also provide the T operator
         ui = |I        0     | (ui)
         ub   |0 (Kbb)^-1(Kbi)|

         Which is the right operator only with the selector operation is applied on system with the folloing form
         K u = f
         
         K =   [Kii   0 
               Kbi Kbb]
         
        u = [ ui
              ub]
        
        f = [fi
             0]
        
        This implies that ui = Kii^(-1)(fi)
        ub = (Kbb)^-1(Kbi)(ui)
    '''
    def __init__(self,Kbb, Kbi, P):
        self.P = P
        self.Kbi = Kbi
        self.Kbb = Kbb
    
    def build_map_function(self):
        
        P = self.P
        Kbi = self.Kbi
        Kbb = self.Kbb 
        
        ub = lambda ui : splinalg.spsolve(Kbb,Kbi.dot(ui))        
        T = lambda ui : P.dot(np.concatenate((ub(ui), ui), axis=-1))  
        return T
    
    
def get_dirichlet_dofs(submesh_obj, direction ='xyz', id_matrix=None):
    ''' get dirichlet dofs given a submesh and a global id_matrix
    
    # parameters:
        # submesh_obj : amfe.SubMesh
            # submesh object with nodes and element of dirichlet
        # direction : str
            # direction to consirer 'xyz'
        # id_matrix : dict
            # dict maps nodes to DOFs
            
    # return 
        # dir_dofs : list
            # list with Dirichlet dofs
    '''

    x_dir = 0
    y_dir = 1
    z_dir = 2
    
    dofs_to_keep = []
    if 'x' in direction:
        dofs_to_keep.append(x_dir)

    if 'y' in direction:
        dofs_to_keep.append(y_dir)
    
    if 'z' in direction:
        dofs_to_keep.append(z_dir)
    
    dir_nodes = submesh_obj.global_node_list
    
    dir_dofs = []
    for node, dofs in id_matrix.items():
        if node in dir_nodes:
            local_dofs = []
            for i in dofs_to_keep:
                try:
                    local_dofs.append(dofs[i])
                except:
                    print('It is not possible to issert dof %i as dirichlet dof' %i)
            dir_dofs.extend(local_dofs)
    
    return dir_dofs