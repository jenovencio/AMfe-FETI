# -*- coding: utf-8 -*-
"""
Created on Fri May  8 16:58:03 2015

@author: johannesr
"""

import numpy as np
import scipy as sp

from amfe.mesh import *
from amfe.element import *
from amfe.assembly import *
from amfe.boundary import *


# Default values:
kwargs = {'E_modul' : 210E9, 'poisson_ratio' : 0.3, 'element_thickness' : 1, 'density' : 1E4}
element_class_dict = {'Tri3' : Tri3(**kwargs), 'Tri6' : Tri6(**kwargs)}



class MechanicalSystem():
    '''
    Mase class for mechanical systems with the goal to black-box the routines
    of assembly and element selection.

    Parameters
    ----------
    None

    Internal variables
    ------------------
    element_class_dict : dict
        dictionary containing instances of the elements with keywords from the amfe.
        If the element properties should be changed, the whole element_class_dict has to be updated.

    Notes
    -----
    The element_class_dict is the key for using multiple elements in one mesh.
    '''

    def __init__(self):
        self.T_output = []
        self.u_output = []
        self.element_class_dict = element_class_dict
        pass

    def load_mesh_from_gmsh(self, msh_file):
        '''
        Load the mesh from a msh-file generated by gmsh.

        Parameters
        ----------
        msh_file : str
            file name to an existing .msh file

        '''
        self.mesh_class = Mesh()
        self.mesh_class.import_msh(msh_file)
        self.node_list = np.array(self.mesh_class.nodes)
        self.element_list = np.array(self.mesh_class.elements)
        self.ndof_global = self.mesh_class.no_of_dofs
        self.node_dof = self.mesh_class.node_dof
        self.assembly_class = Assembly(self.mesh_class, self.element_class_dict)

    def load_mesh_from_csv(self, node_list_csv, element_list_csv, node_dof=2, explicit_node_numbering=False):
        '''
        Loads the mesh from two csv-files containing the node and the element list.

        Parameters
        ----------
        node_list_csv: str
            filename of the csv-file containing the coordinates of the nodes (x, y, z)
        element_list_csv: str
            filename of the csv-file containing the nodes which belong to one element
        node_dof: int, optional
            degree of freedom per node as saved in the csv-file
        explicit_node_numbering : bool, optional
            flag stating, if the node numbers are explcitly numbered in the csv file, i.e. if the first column gives the numbers of the nodes.

        Returns
        -------
        None

        Examples
        --------
        todo

        '''
        self.node_dof = node_dof
        self.mesh_class = Mesh()
        self.mesh_class.read_nodes_from_csv(node_list_csv, node_dof=node_dof, explicit_node_numbering=explicit_node_numbering)
        self.mesh_class.read_elements_from_csv(element_list_csv, explicit_node_numbering=explicit_node_numbering)
        self.node_list = self.mesh_class.nodes.copy()
        self.element_list = self.mesh_class.elements.copy()
        self.ndof_global = self.node_list.size
        self.assembly_class = Assembly(self.mesh_class, self.element_class_dict)
        



    def apply_dirichlet_boundaries(self, dirichlet_boundary_list):
        '''
        Applies dirichlet-boundaries to the system.

        Parameters
        ----------
        dirichlet_boundary_list : list
            list containing the dirichlet-boundary triples (DBT)

            >>> [DBT_1, DBT_2, DBT_3, ]

        Returns
        -------
        None

        Notes
        -----
        each dirchilet_boundary_triple is itself a list containing

        >>> DBT = [master_dof=None, [list_of_slave_dofs], B_matrix=None]

        master_dof : int / None
            the dof onto which the slave dofs are projected. The master_dof
            will be overwritten at the end, i.e. if the master dof should
            participate at the end, it has to be a member in teh list of
            slave_dofs. If the master_dof is set to None, the slave_dofs will
            be fixed
        list_of_slave_dofs : list containing ints
            The list of the dofs which will be projected onto the master dof;
            the weights of the projection are stored in the B_matrix
        B_matrix : ndarras / None
            The weighting-matrix which gives enables to apply complicated
            boundary conditions showing up in symmetry-conditions or rotational
            dofs. The default-value for B_matrix is None, which weighs all
            members of the slave_dof_list equally with 1.

        Examples
        --------

        The dofs 0, 2 and 4 are fixed:

        >>> mysystem = MechanicalSystem()
        >>> DBT = [None, [0, 2, 4], None]
        >>> mysystem.apply_dirichlet_boundaries([DBT, ])

        The dofs 0, 1, 2, 3, 4, 5, 6 are fixed and the dofs 100, 101, 102, 103 have all the same displacements:

        >>> mysystem = MechanicalSystem()
        >>> DBT_fix = [None, np.arange(7), None]
        >>> DBT_disp = [100, [100, 101, 102, 103], None]
        >>> mysystem.apply_dirichlet_boundaries([DBT_fix, DBT_disp])

        Symmetry: The displacement of dof 21 is negativ equal to the displacement of dof 20, i.e. u_20 + u_21 = 0

        >>> mysystem = MechanicalSystem()
        >>> DBT_symm = [20, [20, 21], np.array([1, -1])]
        >>> mysystem.apply_dirichlet_boundaries([DBT_symm, ])

        '''
        self.dirichlet_bc_class = DirichletBoundary(self.ndof_global, dirichlet_boundary_list)
        self.b_constraints = self.dirichlet_bc_class.b_matrix()
        self.ndof_global_constrained = self.b_constraints.shape[-1]


    def apply_neumann_boundaries(self, neumann_boundary_list):
        '''Applies neumann-boundaries to the system.

        Parameters
        ----------
        neumann_boundary_list : list
            list containing the neumann boundary NB lists:

            >>> NB = [dofs_list, type, properties, B_matrix=None]

        Notes
        -----
        the neumann_boundary_list is a list containing the neumann_boundaries:

        >>> [dofs_list, load_type, properties, B_matrix=None]

        dofs_list : list
            list containig the dofs which are loaded
        load_type : str out of {'stepload', 'dirac', 'harmonic', 'ramp', 'const'}
            string specifying the load type
        properties : dict
            dict with the properties for the given load_type (see table below)
        B_matrix : ndarray / None
            Vector giving the load weights for the given dofs in dofs_list. If None is chosen, the weight will be 1 for every dof by default.

        the load_type-Keywords and the corresponding properties are:


        ===========  =====================
        load_type    properties
        ===========  =====================
        'stepload'   (amplitude, time)
        'dirac'      (amplitude, time)
        'harmonic'   (amplitude, frequency)
        'ramp'       (slope, time)
        'const'      (amplitude)
        ===========  =====================

        Examples
        --------

        Stepload on dof 1, 2 and 3 starting at 0.1 s with amplitude 1KN:

        >>> mysystem = MechanicalSystem()
        >>> NB = [[1, 2, 3], 'stepload', (1E3, 0.1), None]
        >>> mysystem.apply_neumann_boundaries([NB, ])

        Harmonic loading on dof 4, 6 and 8 with frequency 8 Hz = 2*2*pi rad and amplitude 100N:

        >>> mysystem = MechanicalSystem()
        >>> NB = [[1, 2, 3], 'harmonic', (100, 8), None]
        >>> mysystem.apply_neumann_boundaries([NB, ])


        '''
        self.neumann_bc_class = NeumannBoundary(self.ndof_global, neumann_boundary_list)
        self._f_ext_without_bc = self.neumann_bc_class.f_ext()


    def export_paraview(self, filename):
        '''Export the system with the given information to paraview
        '''
        if len(self.T_output) is 0:
            self.T_output.append(0)
            self.u_output.append(np.zeros(self.ndof_global))
        print('Start exporting mesh for paraview to', filename)
        self.mesh_class.set_displacement_with_time(self.u_output, self.T_output)
        self.mesh_class.save_mesh_for_paraview(filename)
        print('Mesh for paraview successfully exported')
        pass

    def M_global(self):
        '''
        Return the global stiffness matrix with dirichlet boundary conditions imposed.
        Computes the Mass-Matrix every time again
        '''
        _M = self.assembly_class.assemble_m()
        self._M_bc = self.b_constraints.T.dot(_M.dot(self.b_constraints))
        return self._M_bc

    def K_global(self, u=None):
        '''Return the global tangential stiffness matrix with dirichlet boundary conditions imposed'''
        if u is None:
            u = np.zeros(self.b_constraints.shape[-1])
        _K = self.assembly_class.assemble_k(self.b_constraints.dot(u))
        self._K_bc = self.b_constraints.T.dot(_K.dot(self.b_constraints))
        return self._K_bc

    def f_int_global(self, u):
        '''Return the global elastic restoring force of the system '''
        _f = self.assembly_class.assemble_f(self.b_constraints.dot(u))
        self._f_bc = self.b_constraints.T.dot(_f)
        return self._f_bc

    def f_ext_global(self, u, du, t):
        '''return the global nonlinear external force of the right hand side of the equation, i.e. the excitation'''
        return self.b_constraints.T.dot(self._f_ext_without_bc(t))

    def K_and_f_global(self, u):
        '''
        return the global tangential stiffness matrix and nonlinear force vector in one assembly run.
        '''
        _K, _f = self.assembly_class.assemble_k_and_f(self.b_constraints.dot(u))
        self._K_bc = self.b_constraints.T.dot(_K.dot(self.b_constraints))
        self._f_bc = self.b_constraints.T.dot(_f)
        return self._K_bc, self._f_bc

    def write_timestep(self, t, u):
        '''
        write the timestep to the mechanical_system class
        '''
        self.T_output.append(t)
        self.u_output.append(self.b_constraints.dot(u))



class ReducedSystem(MechanicalSystem):
    '''
    Class for reduced systems.
    It is directyl inherited from MechanicalSystem.
    Provides the interface for an integration scheme and so on where a basis vector is to be chosen...


    Parameters
    ----------
    V_basis : ndarray, optional
        Basis onto which the problem will be projected with an Galerkin-Projection.

    Notes
    -----
    The Basis V is a Matrix with x = V*q mapping the reduced set of coordinates q onto the physical coordinates x. The coordinates x are constrained, i.e. the x denotes the full system in the sense of the problem set and not of the pure finite element set.

    The system runs without providing a V_basis when constructing the method only for the unreduced routines.

    Examples
    --------
    TODO

    '''

    def __init__(self, V_basis=None, **kwargs):
        MechanicalSystem.__init__(self, **kwargs)
        self.V = V_basis

    def K_and_f_global(self, u):
        u_full = self.V.dot(u)
        self._K_unreduced, self._f_unreduced = MechanicalSystem.K_and_f_global(self, u_full)
        self._K_reduced = self.V.T.dot(self._K_unreduced.dot(self.V))
        self._f_int_reduced = self.V.T.dot(self._f_unreduced)
        return self._K_reduced, self._f_int_reduced


    def K_global(self, u):
        return self.V.T.dot(MechanicalSystem.K_global(self, self.V.dot(u)).dot(self.V))

    def f_ext_global(self, u, du, t):
        return self.V.T.dot(MechanicalSystem.f_ext_global(self, self.V.dot(u), du, t))

    def f_int_global(self, u):
        return self.V.T.dot(MechanicalSystem.f_int_global(self, self.V.dot(u)))

    def M_global(self):
        return self.V.T.dot(MechanicalSystem.M_global(self).dot(self.V))

    def write_timestep(self, t, u):
        MechanicalSystem.write_timestep(self, t, self.V.dot(u))

    def K_unreduced(self, u=None):
        '''
        unreduced Stiffness Matrix
        '''
        return MechanicalSystem.K_global(self, u)

    def f_int_unreduced(self, u):
        return MechanicalSystem.f_int_global(self, u)

    def M_unreduced(self):
        return MechanicalSystem.M_global(self)
