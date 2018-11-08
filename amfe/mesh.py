# Copyright (c) 2017, Lehrstuhl fuer Angewandte Mechanik, Technische
# Universitaet Muenchen.
#
# Distributed under BSD-3-Clause License. See LICENSE-File for more information
#
"""
Mesh module of amfe. It handles the mesh from import, defining the dofs for the
boundary conditions and the export.
"""

__all__ = ['Mesh',
           'create_xdmf_from_hdf5',
          ]

import os
import copy
from .wrappers.read_abaqus_mesh import read_inp, create_amfe_elem_data_frame, create_amfe_node_array

# XML stuff
from xml.etree.ElementTree import Element, SubElement
from xml.etree import ElementTree
from xml.dom import minidom



import pandas as pd
import h5py
import numpy as np
import logging


from .element import Tet4, \
    Tet10, \
    Tri3, \
    Tri6, \
    Quad4, \
    Quad8, \
    Bar2Dlumped, \
    LineLinearBoundary, \
    LineQuadraticBoundary, \
    Tri3Boundary, \
    Tri6Boundary, \
    Hexa8, \
    Hexa20, \
    Quad4Boundary, \
    Quad8Boundary, \
    Prism6



from .mesh_tying import master_slave_constraint

# Element mapping is described here. If a new element is implemented, the
# features for import and export should work when the followig list will be updated.
element_mapping_list = [
    # internal Name, XMF Key,   gmsh-Key, vtk/ParaView-Key, no_of_nodes, description
    ['Tet4',          'Tetrahedron',   4, 10,  4,
     'Linear Tetraeder / nodes on every corner'],
    ['Tet10',         'Tetrahedron_10',  11, 24, 10,
     'Quadratic Tetraeder / 4 nodes at the corners, 6 nodes at the faces'],
    ['Hexa8',         'Hexahedron', 5, 12, 8,
     'Linear brick element'],
    ['Hexa20',         'Hex_20', 17, 25, 20,
     'Quadratic brick element'],
    ['Tri6',          'Triangle_6',   9, 22,  6,
     'Quadratic triangle / 6 node second order triangle'],
    ['Tri3',          'Triangle',   2,  5,  3,
     'Straight triangle / 3 node first order triangle'],
    ['Tri10',         '',  21, 35, 10,
     'Cubic triangle / 10 node third order triangle'],
    ['Quad4',         'Quadrilateral',   3,  9,  4,
     'Bilinear rectangle / 4 node first order rectangle'],
    ['Quad8',         'Quadrilateral_8',  16, 23,  8,
     'Biquadratic rectangle / 8 node second order rectangle'],
    ['Prism6',         'Wedge', 6, 23,  6,
     'Trilinear 6 node prism'],
    ['straight_line', 'Edge',   1,  3,  2,
     'Straight line composed of 2 nodes'],
    ['quadratic_line', 'Edge_3',  8, 21,  3,
     'Quadratic edge/line composed of 3 nodes'],
    ['point',       '', 15, np.NAN,  1, 'Single Point'],
    ['Quad4Boundary',         'Quadrilateral',   3001,  9,  4,
     'Bilinear rectangle / 4 node first order rectangle'],
    # Bars are missing, which are used for simple benfield truss
]

#
# Building the conversion dicts from the element_mapping_list
#
gmsh2amfe        = dict([])
amfe2gmsh        = dict([])
amfe2vtk         = dict([])
amfe2xmf         = dict([])
amfe2no_of_nodes = dict([])

for element in element_mapping_list:
    gmsh2amfe.update({element[2] : element[0]})
    amfe2gmsh.update({element[0] : element[2]})
    amfe2vtk.update( {element[0] : element[3]})
    amfe2xmf.update({element[0] : element[1]})
    amfe2no_of_nodes.update({element[0] : element[4]})

# Some conversion stuff fron NASTRAN to AMFE
nas2amfe = {'CTETRA' : 'Tet10',
            'CHEXA' : 'Hexa8'}

# Same for Abaqus
abaq2amfe = {'C3D10' : 'Tet10',
             'C3D8' : 'Hexa8',
             'C3D20' : 'Hexa20',
             'C3D4' : 'Tet4',
             'C3D6' : 'Prism6', # 6 node prism
             'C3D8I' : 'Hexa8', # acutally the better version
             'B31' : None,
             'CONN3D2' : None,
            }

# Abaqus faces for identifying surfaces
abaq_faces = {
    'Hexa8': {'S1' : ('Quad4', np.array([0, 1, 2, 3])),
              'S2' : ('Quad4', np.array([4, 7, 6, 5])),
              'S3' : ('Quad4', np.array([0, 4, 5, 1])),
              'S4' : ('Quad4', np.array([1, 5, 6, 2])),
              'S5' : ('Quad4', np.array([2, 6, 7, 3])),
              'S6' : ('Quad4', np.array([3, 7, 4, 0])),
             },

    'Hexa20' : {'S1': ('Quad8', np.array([ 0,  1,  2,  3,  8,  9, 10, 11])),
                'S2': ('Quad8', np.array([ 4,  7,  6,  5, 15, 14, 13, 12])),
                'S3': ('Quad8', np.array([ 0,  4,  5,  1, 16, 12, 17,  8])),
                'S4': ('Quad8', np.array([ 1,  5,  6,  2, 17, 13, 18,  9])),
                'S5': ('Quad8', np.array([ 2,  6,  7,  3, 18, 14, 19, 10])),
                'S6': ('Quad8', np.array([ 3,  7,  4,  0, 19, 15, 16, 11]))},

    'Tet4': {'S1' : ('Tri3', np.array([0, 1, 2])),
             'S2' : ('Tri3', np.array([0, 3, 1])),
             'S3' : ('Tri3', np.array([1, 3, 2])),
             'S4' : ('Tri3', np.array([2, 3, 0])),
            },

    'Tet10': {'S1' : ('Tri6', np.array([0, 1, 2, 4, 5, 6])),
              'S2' : ('Tri6', np.array([0, 3, 1, 7, 8, 4])),
              'S3' : ('Tri6', np.array([1, 3, 2, 8, 9, 5])),
              'S4' : ('Tri6', np.array([2, 3, 0, 9, 7, 6])),
             },

    'Prism6' : {'S1': ('Tri3', np.array([0, 1, 2])),
                'S2': ('Tri3', np.array([3, 5, 4])),
                'S3': ('Quad4', np.array([0, 3, 4, 1])),
                'S4': ('Quad4', np.array([1, 4, 5, 2])),
                'S5': ('Quad4', np.array([2, 5, 3, 0])),
                },
}


def check_dir(*filenames):
    '''
    Check if paths exists; if not, the given paths will be created.

    Parameters
    ----------
    *filenames : string or list of strings
        string containing a path.

    Returns
    -------
    None
    '''
    for filename in filenames:  # loop on files
        dir_name = os.path.dirname(filename)
        # check if directory does not exist; then create directory
        if not os.path.exists(dir_name) or dir_name == '':
            os.makedirs(os.path.dirname(filename))          # then create directory
            logging.info("Created directory: " + os.path.dirname(filename))


def prettify_xml(elem):
    '''
    Return a pretty string from an XML Element-Tree

    Parameters
    ----------
    elem : Instance of xml.etree.ElementTree.Element
        XML element tree

    Returns
    -------
    str : string
        well formatted xml file string
    '''
    rough_string = ElementTree.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml()


def shape2str(tupel):
    '''
    Convert a tupel to a string containing the numbers of the tupel for xml
    export.

    Parameters
    ----------
    tupel : tupel
        tupel containing numbers (usually the shape of an array)

    Returns
    -------
    str : string
        string containing the numbers of the tupel
    '''
    return ' '.join([str(i) for i in tupel])


def h5_set_attributes(h5_object, attribute_dict):
    '''
    Add the attributes from attribute_dict to the h5_object.

    Parameters
    ----------
    h5_object : instance of h5py File, DataSet or Group
        hdf5 object openend with h5py
    attribute_dict : dict
        dictionary with keys and attributes to be added to the h5_object

    Returns
    -------
    None
    '''
    for key in attribute_dict:
        h5_object.attrs[key] = attribute_dict[key]
    return


def create_xdmf_from_hdf5(filename):
    '''
    Create an accompanying xdmf file for a given hdmf file.

    Parameters
    ----------
    filename : str
        filename of the hdf5-file. Produces an XDMF-file of same name with
        .xdmf ending.

    Returns
    -------
    None

    '''
    filename_no_dir = os.path.split(filename)[-1]
    # filename_no_ext = os.path.splitext(filename)[0]

    with h5py.File(filename, 'r') as f:
        h5_topology = f['mesh/topology']
        h5_nodes = f['mesh/nodes']
        h5_time_vals = f['time_vals']

        xml_root = Element('Xdmf', {'Version':'2.2'})
        domain = SubElement(xml_root, 'Domain')
        time_grid = SubElement(domain, 'Grid', {'GridType':'Collection',
                                                'CollectionType':'Temporal'})
        # time loop
        for i, T in enumerate(f['time']):
            spatial_grid= SubElement(time_grid, 'Grid',
                                     {'Type':'Spatial',
                                      'GridType':'Collection'})

            time = SubElement(spatial_grid, 'Time', {'TimeType':'Single',
                                                     'Value':str(T)})
            # loop over all mesh topologies
            for key in h5_topology.keys():
                grid = SubElement(spatial_grid, 'Grid', {'Type':'Uniform'})
                topology = SubElement(grid, 'Topology',
                                      {'TopologyType':h5_topology[key].attrs['TopologyType'],
                                       'NumberOfElements':str(h5_topology[key].shape[0])})
                topology_data = SubElement(topology, 'DataItem',
                                       {'NumberType':'Int',
                                        'Format':'HDF',
                                        'Dimensions':shape2str(h5_topology[key].shape)})
                topology_data.text = filename_no_dir + ':/mesh/topology/' + key

                # Check, if mesh is 2D or 3D
                xdmf_node_type = 'XYZ'
                if h5_nodes.shape[-1] == 2:
                    xdmf_node_type = 'XY'

                geometry = SubElement(grid, 'Geometry',
                                      {'Type':'Uniform',
                                       'GeometryType':xdmf_node_type})
                geometry_data_item = SubElement(geometry, 'DataItem',
                                                {'NumberType':'Float',
                                                 'Format':'HDF',
                                                 'Dimensions':shape2str(h5_nodes.shape)})
                geometry_data_item.text = filename_no_dir + ':/mesh/nodes'

                # Attribute loop for export of displacements, stresses etc.
                for key_t in h5_time_vals.keys():
                    field = h5_time_vals[key_t]
                    if field.attrs['ParaView'] == np.True_:
                        field_attr = SubElement(grid, 'Attribute',
                                                {'Name':field.attrs['Name'],
                                                 'AttributeType':
                                                    field.attrs['AttributeType'],
                                                 'Center':field.attrs['Center']})
                        no_of_components = field.attrs['NoOfComponents']
                        field_dim = (field.shape[0] // no_of_components,
                                     no_of_components)
                        field_data = SubElement(field_attr, 'DataItem',
                                                {'ItemType':'HyperSlab',
                                                 'Dimensions':shape2str(field_dim)})

                        field_hyperslab = SubElement(field_data, 'DataItem',
                                                     {'Dimensions':'3 2',
                                                      'Format':'XML'})

                        # pick the i-th column via hyperslab; If no temporal values
                        # are pumped out, use the first column
                        if i <= field.shape[-1]: # field has time instance
                            col = str(i)
                        else: # field has no time instance, use first col
                            col = '0'
                        field_hyperslab.text = '0 ' + col + ' 1 1 ' + \
                                                str(field.shape[0]) + ' 1'
                        field_hdf = SubElement(field_data, 'DataItem',
                                               {'Format':'HDF',
                                                'NumberType':'Float',
                                                'Dimensions':shape2str(field.shape)})
                        field_hdf.text = filename_no_dir + ':/time_vals/' + key_t

                # Attribute loop for cell values like weights
                if 'time_vals_cell' in f.keys():
                    h5_time_vals_cell = f['time_vals_cell']
                    for key_2 in h5_time_vals_cell.keys():
                        field = h5_time_vals_cell[key_2][key]
                        if field.attrs['ParaView'] == np.True_:
                            field_attr = SubElement(grid, 'Attribute',
                                                    {'Name':field.attrs['Name'],
                                                     'AttributeType':
                                                        field.attrs['AttributeType'],
                                                     'Center':field.attrs['Center']})
                            no_of_components = field.attrs['NoOfComponents']
                            field_dim = (field.shape[0] // no_of_components,
                                         no_of_components)
                            field_data = SubElement(field_attr, 'DataItem',
                                                    {'ItemType':'HyperSlab',
                                                     'Dimensions':shape2str(field_dim)})

                            field_hyperslab = SubElement(field_data, 'DataItem',
                                                         {'Dimensions':'3 2',
                                                          'Format':'XML'})

                            # pick the i-th column via hyperslab; If no temporal values
                            # are pumped out, use the first column
                            if i <= field.shape[-1]: # field has time instance
                                col = str(i)
                            else: # field has no time instance, use first col
                                col = '0'
                            field_hyperslab.text = '0 ' + col + ' 1 1 ' + \
                                                    str(field.shape[0]) + ' 1'
                            field_hdf = SubElement(field_data, 'DataItem',
                                                   {'Format':'HDF',
                                                    'NumberType':'Float',
                                                    'Dimensions':shape2str(field.shape)})
                            field_hdf.text = filename_no_dir + ':/time_vals_cell/' \
                                             + key_2 + '/' + key

    # write xdmf-file
    xdmf_str = prettify_xml(xml_root)
    filename_no_ext, ext = os.path.splitext(filename)
    with open(filename_no_ext + '.xdmf', 'w') as f:
        f.write(xdmf_str)


class Mesh:
    '''
    Class for handling the mesh operations.

    Attributes
    ----------
    nodes : ndarray
        Array of x-y-z coordinates of the nodes. Dimension is
        (no_of_nodes, no_of_dofs_per_node).
        If no_of_dofs_per_node: z-direction is dropped!
    connectivity : list
        List of nodes indices belonging to one element.
    constraint_list: ndarray
        Experimental: contains constraints imported from nastran-files via
        import_bdf()
    el_df : pandas.DataFrame
        Pandas Dataframe containing Element-Definitions of the Original file
        (e.g. *.msh or *.bdf-File)
    ele_obj : list
        List of element objects. The list contains actually only the pointers
        pointing to the element object. For each combination of element-type
        and material only one Element object is instanciated.
        ele_obj contains for each element a pointer to one of these Element
        objects.
    neumann_connectivity : list
        list of nodes indices belonging to one element for neumann BCs.
    neumann_obj : list
        List of element objects for the neumann boundary conditions.
    nodes_dirichlet : ndarray
        Array containing the nodes involved in Dirichlet Boundary Conditions.
    dofs_dirichlet : ndarray
        Array containing the dofs which are to be blocked by Dirichlet Boundary
        Conditions.
    no_of_dofs_per_node : int
        Number of dofs per node. Is 3 for 3D-problems, 2 for 2D-problems. If
        rotations are considered, this nubmer can be >3.
    no_of_elements : int
        Number of elements in the whole mesh associated with an element object.
    no_of_nodes : int
        Number of nodes of the whole system.
    no_of_dofs : int
        Number of dofs of the whole system (including constrained dofs).
    element_class_dict : dict
        Dictionary containing objects of elements.
    element_boundary_class_dict : dict
        Dictionary containing objects of skin elements.
    node_idx : int
        index describing, at which position in the Pandas Dataframe `el_df`
        the nodes of the element start.
    domain_dict : dict
        dict of submeshs which represents the domain of the problem 
    group_dict : dict
        dict of submeshs which are generate by split_in_group method
    '''

    def __init__(self):
        '''
        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        self.nodes = np.array([])
        self.connectivity = []
        self.ele_obj = []
        self.domain_dict = {} # dict of submeshs 
        self.groups = {} # dict of submeshs after split_in_group method
        self.neumann_connectivity = []
        self.neumann_obj = []
        self.nodes_dirichlet = np.array([], dtype=int)
        self.dofs_dirichlet = np.array([], dtype=int)
        self.constraint_list = []  # experimental; Introduced for nastran meshes
        # the displacements; They are stored as a list of numpy-arrays with
        # shape (ndof, no_of_dofs_per_node):
        self.no_of_dofs_per_node = 0
        self.no_of_dofs = 0
        self.no_of_nodes = 0
        self.no_of_elements = 0
        self.el_df = pd.DataFrame()
        self.node_idx = 0
        
        # elements and node sets
        self.elem_sets_dict = {}
        self.node_sets_dict = {}
        
        # Element Class dictionary with all available elements
        # This dictionary is only needed for load_group_to_mesh()-method
        kwargs = { }
        self.element_class_dict = {'Tet4'  : Tet4(**kwargs),
                                   'Tet10' : Tet10(**kwargs),
                                   'Hexa8' : Hexa8(**kwargs),
                                   'Hexa20': Hexa20(**kwargs),
                                   'Prism6' : Prism6(**kwargs),
                                   'Tri3'  : Tri3(**kwargs),
                                   'Tri6'  : Tri6(**kwargs),
                                   'Quad4' : Quad4(**kwargs),
                                   'Quad8' : Quad8(**kwargs),
                                   'Bar2Dlumped' : Bar2Dlumped(**kwargs),
                                  }

        kwargs = {'val' : 1., 'direct' : 'normal'}

        self.element_boundary_class_dict = {
            'straight_line' : LineLinearBoundary(**kwargs),
            'quadratic_line': LineQuadraticBoundary(**kwargs),
            'Tri3'          : Tri3Boundary(**kwargs),
            'Tri6'          : Tri6Boundary(**kwargs),
            'Quad4'         : Quad4Boundary(**kwargs),
            'Quad8'         : Quad8Boundary(**kwargs),}

        # actual set of implemented elements
        self.element_2d_set = {'Tri6', 'Tri3', 'Quad4', 'Quad8', }
        self.element_3d_set = {'Tet4', 'Tet10', 'Hexa8', 'Hexa20', 'Prism6'}

        self.boundary_2d_set = {'straight_line', 'quadratic_line'}
        self.boundary_3d_set = {'straight_line', 'quadratic_line',
                                'Tri6', 'Tri3', 'Tri10', 'Quad4', 'Quad8'}

    def _update_mesh_props(self):
        '''
        Update the number properties of nodes and elements when the mesh has
        been updated
        
        It updates the following properties of the mesh-class-object:
            - no_of_nodes
            - no_of_dofs
            - no_of_elements
            
        '''
        
        if len(self.connectivity)>0:
            self.no_of_nodes = len(set(list(np.concatenate( self.connectivity, axis=0 ))))
        else:
            self.no_of_nodes = len(self.nodes)  # old method 

        self.no_of_dofs = self.no_of_nodes*self.no_of_dofs_per_node
        self.no_of_elements = len(self.connectivity)

    def import_csv(self, filename_nodes, filename_elements,
                   explicit_node_numbering=False, ele_type=False):
        '''
        Imports the nodes list and elements list from 2 different csv files.

        Parameters
        -----------
        filename_nodes : str
            name of the file containing the nodes in csv-format
        filename_elements : str
            name of the file containing the elements in csv-format
        explicit_node_numbering : bool, optional
            Flag stating, if the nodes are explicitly numbered in the csv-file.
            When set to true, the first column is assumed to have the node numbers
            and is thus ignored.
        ele_type: str
            Spezifiy elements type of the mesh (e.g. for a Tri-Mesh different
            elements types as Tri3, Tri4, Tri6 can be used)
            If not spezified value is set to 'False'

        Returns
        --------
        None

        Examples
        ---------
        TODO

        '''
        print('This function is deprecated! It does not work properly!')
        #######################################################################
        # NODES
        #######################################################################
        try:
            self.nodes = np.genfromtxt(filename_nodes, delimiter = ',', skip_header = 1)
        except:
            ImportError('Error while reading file ' + filename_nodes)
        # when line numbers are erased if they are content of the csv
        if explicit_node_numbering:
            self.nodes = self.nodes[:,1:]

        #######################################################################
        # ELEMENTS
        #######################################################################
        # Dictionary um an Hand der Anzahl der Knoten des Elements auf den Typ
        # des Elements zu schließen
        mesh_type_dict = {3: "Tri3",
                          4: "Quad4",
                          2: "Bar2D"} # Bislang nur 2D-Element aus csv auslesbar

        print('Reading elements from csv...  ', end="")
        self.connectivity = np.genfromtxt(filename_elements,
                                          delimiter = ',',
                                          dtype = int,
                                          skip_header = 1)
        if self.connectivity.ndim == 1: # Wenn nur genau ein Element vorliegt
            self.connectivity = np.array([self.connectivity])
            # Falls erste Spalte die Elementnummer angibt, wird diese hier
        # abgeschnitten, um nur die Knoten des Elements zu erhalten
        if explicit_node_numbering:
            self.connectivity = self.connectivity[:,1:]

        if ele_type:  # If element type is spezified, use this spezified type
            mesh_type = ele_type
        # If element type is not spzezified, try to determine element type
        # depending on the number of nodes per element (see default values for
        # different number of nodes per element in 'mesh_type_dict')
        else:
            try:  # Versuche Elementtyp an Hand von Anzahl der Knoten pro Element auszulesen
                (no_of_ele, no_of_nodes_per_ele) = self.connectivity.shape
                mesh_type = mesh_type_dict[no_of_nodes_per_ele] # Weise Elementtyp zu
            except:
                print('FEHLER beim Einlesen der Elemente. Typ nicht vorhanden.')
                raise

        print('Element type is {0}...  '.format(mesh_type), end="")
        self._update_mesh_props()
        print('Reading elements successful.')
        return

    def import_inp(self, filename, scale_factor=1.):
        '''
        Import Abaqus input file.

        Parameters
        ----------
        filename : string
            filename of the .msh-file
        scale_factor : float, optional
            scale factor for the mesh to adjust the units. The default value is
            1, i.e. no scaling is done.


        Returns
        -------
        None

        Notes
        -----
        This function is heavily experimental. It is just working for a subset
        of Abaqus input files and the goal is to capture the mesh of the model.

        The internal representation of the elements is done via a Pandas
        Dataframe object.

        '''

        print('*************************************************************')
        print('\nLoading Abaqus-mesh from', filename)
    
        nodes_dict, elem_list, nset_list, elset_list  = read_inp(filename)
        el_df, node_idx = create_amfe_elem_data_frame(elem_list)
        nodes = create_amfe_node_array(nodes_dict)
        
        
        for rows in nset_list:
            self.node_sets_dict[rows['node_set']] = rows['node_list']


        
        for rows in elset_list:
            self.self.elem_sets_dict[rows['elem_set']] = rows['elem_dict']


        self.el_df = el_df
        self.node_idx = node_idx
        self.nodes = nodes*scale_factor
        
        
        
        
        elem_set = self.get_elem_types()
        
        self.no_of_dofs_per_node = 2
        for i in elem_set:
            if i in self.element_3d_set:
                self.no_of_dofs_per_node = 3
        
        if self.no_of_dofs_per_node ==2:
            print('WARNING! 2D case were selected')
        
        if self.no_of_dofs_per_node ==3:
            print('WARNING! 3D case were selected')
            
        self._update_mesh_props()
        # printing some information regarding the physical groups
        print('Mesh', filename, 'successfully imported.',
              '\nAssign a material to a physical group.')
        print('*************************************************************')
        return

    def import_bdf(self, filename, scale_factor=1.):
        '''
        Import a NASTRAN mesh.

        Parameters
        ----------
        filename : string
            filename of the .bdf-file
        scale_factor : float, optional
            scale factor for the mesh to adjust the units. The default value is
            1, i.e. no scaling is done.

        Returns
        -------
        None

        Notes
        -----
        This function is heavily experimental. It is just working for a subset
        of NASTRAN input files and the goal is to capture the mesh and the
        constraints of the model. The constraints are captured in the
        constraint_list-object of the class.

        The internal representation of the elements is done via a Pandas
        Dataframe object.

        '''
        comment_tag = '$'
        long_format_tag = '*'
        print('*************************************************************')
        print('\nLoading NASTRAN-mesh from', filename)

        nodes_list = []
        elements_list = []
        constraint_list = []
        # Flag indicating, that element was read in previous line
        element_active = False

        with open(filename, 'r') as infile:
            file_data = infile.read().splitlines()

        # Loop over all lines in the file
        for line in file_data:
            # Filter out comments
            if comment_tag in line:
                element_active = False
                continue

            if long_format_tag in line:  # Long format
                s = [line[:8], ]
                s.extend([line[i*16:(i+1)*16] for i in range(len(line)//16)])
                # Note: here some more logics is necessary to handle line
                # continuation
            elif ',' in line:  # Free field format
                s = line.split(',')
            else:  # The regular short format
                s = [line[i*8:(i+1)*8] for i in range(len(line)//8)]

            if len(s) < 1:  # Empty line
                element_active = False
                continue

            # Get the nodes
            if 'GRID' in s[0]:
                nodes_list.append([int(s[1]),
                                   float(s[3]), float(s[4]), float(s[5])])

            elif s[0].strip() in nas2amfe:
                elements_list.append(s)
                element_active = 'Element'
            elif 'RBE' in s[0]:
                constraint_list.append(s)
                element_active = 'Constraint'
            elif s[0] != '        ':  # There is an unknown element
                element_active = False

            # Catch the free lines where elements are continued
            elif s[0] == '        ' and element_active:
                if element_active == 'Element':
                    elements_list[-1].extend(s[1:])
                if element_active == 'Constraint':
                    constraint_list[-1].extend(s[1:])

        self.no_of_dofs_per_node = 3 # this is just hard coded right now...
        self.nodes = np.array(nodes_list, dtype=float)[:,1:]
        self.nodes *= scale_factor # scaling of nodes

        nodes_dict = pd.Series(index=np.array(nodes_list, dtype=int)[:,0],
                               data=np.arange(len(nodes_list)))

        for idx, ptr in enumerate(elements_list):
            tmp = [nas2amfe[ptr.pop(0).strip()],
                   int(ptr.pop(0)), int(ptr.pop(0))]
            tmp.extend([nodes_dict[int(i)] for i in ptr])
            elements_list[idx] = tmp

        for idx, ptr in enumerate(constraint_list):
            tmp = [ptr.pop(0).strip(), int(ptr.pop(0)), int(ptr.pop(1))]
            tmp.append([nodes_dict[int(i)] for i in ptr[4:]])
            constraint_list[idx] = tmp

        self.constraint_list = constraint_list
        self.el_df = df = pd.DataFrame(elements_list, dtype=int)
        df.rename(copy=False, inplace=True,
                  columns={0 : 'el_type',
                           1 : 'idx_nastran',
                           2 : 'phys_group',
                          })
        self.node_idx = 3
        self._update_mesh_props()
        # printing some information regarding the physical groups
        print('Mesh', filename, 'successfully imported.',
              '\nAssign a material to a physical group.')
        print('*************************************************************')
        return

    def import_msh(self, filename, scale_factor=1.):
        '''
        Import a gmsh-mesh.
        
        This method sets the following properties:
            - el_df: Element Definitions as pandas Dataframe
                    (Attention! This property is not the property that defines
                    the elements for later calculation. This is located in
                    the connectivity property)
            - node_idx: First Column in el_df pandas dataframe where the first
                    node-id of each element is stored
            - no_of_dofs_per_node: important to recognize 2D vs. 3D problem
            - nodes: Node Definitions (Locations)

        Parameters
        ----------
        filename : string
            filename of the .msh-file
        scale_factor : float, optional
            scale factor for the mesh to adjust the units. The default value is
            1, i.e. no scaling is done.

        Returns
        -------
        None

        Notes
        -----
        The internal representation of the elements is done via a Pandas Dataframe
        object. This gives the possibility to dynamically choose a part of the mesh
        for boundary conditons etc.

        '''
        tag_format_start   = "$MeshFormat"
        tag_format_end     = "$EndMeshFormat"
        tag_nodes_start    = "$Nodes"
        tag_nodes_end      = "$EndNodes"
        tag_elements_start = "$Elements"
        tag_elements_end   = "$EndElements"

        logging.info('\n*************************************************************')
        logging.info('Loading gmsh-mesh from ' + filename)

        with open(filename, 'r') as infile:
            data_geometry = infile.read().splitlines()

        for s in data_geometry:
            if s == tag_format_start: # Start Formatliste
                i_format_start   = data_geometry.index(s) + 1
            elif s == tag_format_end: # Ende Formatliste
                i_format_end     = data_geometry.index(s)
            elif s == tag_nodes_start: # Start Knotenliste
                i_nodes_start    = data_geometry.index(s) + 2
                n_nodes          = int(data_geometry[i_nodes_start-1])
            elif s == tag_nodes_end: # Ende Knotenliste
                i_nodes_end      = data_geometry.index(s)
            elif s == tag_elements_start: # Start Elementliste
                i_elements_start = data_geometry.index(s) + 2
                n_elements       = int(data_geometry[i_elements_start-1])
            elif s == tag_elements_end: # Ende Elementliste
                i_elements_end   = data_geometry.index(s)

        # Check inconsistent dimensions
        if (i_nodes_end-i_nodes_start)!=n_nodes \
            or (i_elements_end-i_elements_start)!= n_elements:
            raise ValueError('Error while processing the file!',
                             'Dimensions are not consistent.')

        # extract data from file to lists
        list_imported_mesh_format = data_geometry[i_format_start : i_format_end]
        list_imported_nodes = data_geometry[i_nodes_start : i_nodes_end]
        list_imported_elements = data_geometry[i_elements_start : i_elements_end]

        # conversion of the read strings to integer and floats
        for j in range(len(list_imported_mesh_format)):
            list_imported_mesh_format[j] = [float(x) for x in
                                            list_imported_mesh_format[j].split()]
        for j in range(len(list_imported_nodes)):
            list_imported_nodes[j] = [float(x) for x in
                                      list_imported_nodes[j].split()]
        for j in range(len(list_imported_elements)):
            list_imported_elements[j] = [int(x) for x in
                                         list_imported_elements[j].split()]

        # Construct Pandas Dataframe for the elements (self.el_df and df for shorter code)
        #subset_list_imported_elements = [row[0:4] for row in list_imported_elements] 
        
        #---------------------------------------------------------------------
        # Handling partitioned mesh
        
        # spliting elem list in nodes_list and partition list
        subset_list_imported_elements = []
        partitions_num_list = []
        partitions_index_list = []
        partitions_neighbors_list = []
        for elem_id,col in enumerate(list_imported_elements):
            total_columns = len(list_imported_elements[elem_id])
            elem_tag = col[2]
            start_node_id = elem_tag + 3
            col_slice = col[0:5]
            col_slice.extend(col[start_node_id:])
            subset_list_imported_elements.append(col_slice) 
            
            # creating lists with number of partitions and partition index list 
            # and neighbors list per elem
            if elem_tag==4:
                partitions_num_list.append(col[5])
                partitions_index_list.append(col[6])
                partitions_neighbors_list.append(None)
            elif elem_tag>4:
                partitions_num_list.append(col[5])
                partitions_index_list.append(col[6])
                partitions_neighbors_list.append(col[7:start_node_id])                
            else:
                partitions_num_list.append(0)
                partitions_index_list.append(None)
                partitions_neighbors_list.append(None)
        
        
        
        self.el_df = df = pd.DataFrame(subset_list_imported_elements)
        df.rename(copy=False, inplace=True,
                  columns={0 : 'idx_gmsh',
                           1 : 'el_type',
                           2 : 'no_of_tags',
                           3 : 'phys_group',
                           4 : 'geom_entity'})


        # determine the index, where the nodes of the element start in the dataframe
        if len(df[df['no_of_tags'] != 2]) == 0:
            self.node_idx = node_idx = 5
            
        elif len(df[df['no_of_tags'] != 2]) > 0:
            # geeting columns names in order to reorder
            df_col = df.columns.tolist()
            df_col_start = df_col[0:5]
            df_col_end = df_col[5:]
            df_col_middle = ['no_of_mesh_partitions','partition_id','partitions_neighbors']
            
            # appending partitions to pandas dataframe
            df[df_col_middle[0]] = pd.Series(partitions_num_list)
            df[df_col_middle[1]] = pd.Series(partitions_index_list)
            df[df_col_middle[2]] = pd.Series(partitions_neighbors_list)
            
            # reorder dataframe columns
            new_col_order = df_col_start + df_col_middle + df_col_end
            self.el_df = df = df[new_col_order]
            
            self.node_idx = node_idx = 8
            
        else:
            raise('''The type of mesh is not supported yet.
        Either you have a corrupted .msh-file or you have a too
        complicated mesh partition structure.''')

        # correct the issue wiht gmsh index starting at 1 and amfe index starting with 0
        df.iloc[:, node_idx:] -= 1

        # change the el_type to the amfe convention
        df['el_type'] = df.el_type.map(gmsh2amfe)

        element_types = pd.unique(df['el_type'])
        # Check, if the problem is 2d or 3d and adjust the dimension of the nodes
        # Check, if there is one 3D-Element in the mesh!
        self.no_of_dofs_per_node = 2
        for i in element_types:
            if i in self.element_3d_set:
                print('3D case was choosen')
                self.no_of_dofs_per_node = 3

        # fill the nodes to the array
        self.nodes = np.array(list_imported_nodes)[:,1:1+self.no_of_dofs_per_node]
        self.nodes *= scale_factor # scaling of nodes

        # Change the indices of Tet10-elements, as they are numbered differently
        # from the numbers used in AMFE and ParaView (last two indices permuted)
        if 'Tet10' in element_types:
            row_loc = df['el_type'] == 'Tet10'
            i = self.node_idx
            df.ix[row_loc, i + 9], df.ix[row_loc, i + 8] = \
            df.ix[row_loc, i + 8], df.ix[row_loc, i + 9]
        # Same node nubmering issue with Hexa20
        if 'Hexa20' in element_types:
            row_loc = df['el_type'] == 'Hexa20'
            hexa8_gmsh_swap = np.array([0,1,2,3,4,5,6,7,8,11,13,9,16,18,19,
                                        17,10,12,14,15])
            i = self.node_idx
            df.ix[row_loc, i:] = df.ix[row_loc, i + hexa8_gmsh_swap].values

        self._update_mesh_props()
        # printing some information regarding the physical groups
        logging.info('Mesh ' + filename + ' successfully imported.' + \
              '\nAssign a material to a physical group.')
        logging.info('*************************************************************')
        return

    def load_group_to_mesh(self, key, material, mesh_prop='phys_group'):
        '''
        Add a physical group to the main mesh with given material.
        
        It generates the connectivity list (mesh-class-property connectivity)
        which contains the element configuration as array
        and provides a map with pointers to Element-Objects (Tet, Hex etc.)
        which already contain information about material that is passed.
        Each element gets a pointer to such an element object.

        Parameters
        ----------
        key : int
            Key for mesh property which is to be chosen. Matches the group given
            in the gmsh file. For help, the function mesh_information or
            boundary_information gives the groups
        material : Material class
            Material class which will be assigned to the elements
        mesh_prop : {'phys_group', 'geom_entity', 'el_type', 'partition_id'}, optional
            label of which the element should be chosen from. Standard is
            physical group.

        Returns
        -------
        None

        '''
        # asking for a group to be chosen, when no valid group is given
        df = self.el_df
        if mesh_prop not in df.columns:
            logging.info('The given mesh property "' + str(mesh_prop) + '" is not valid!',
                  'Please enter a valid mesh prop from the following list:\n')
            for i in df.columns:
                logging.info(i)
            return
        while key not in pd.unique(df[mesh_prop]):
            self.mesh_information(mesh_prop)
            logging.info('\nNo valid ' + mesh_prop, ' is given.\n(Given ' + mesh_prop + \
                  ' is ' + key + ')')
            key = int(input('Please choose a ' + mesh_prop + ' to be used as mesh: '))

        # make a pandas dataframe just for the desired elements
        elements_df = df[df[mesh_prop] == key]

        
        
        connectivity = self.compute_connectivity_and_add_material(elements_df,material)
        
        # log info some output stuff
        logging.info('*************************************************************')
        logging.info('\n '+ mesh_prop + str(key) + ' with ' + str(len(connectivity)) + \
              ' elements successfully added.')
        logging.info('Total number of elements in mesh:' + str(len(self.ele_obj)))
        logging.info('*************************************************************')
        
        return None

    def compute_connectivity_and_add_material(self,elements_df,material):
        ''' This method compute the elements connectivity
        and assign a material for each element
        
        argument:
            elements_df : Pandas DataFrame
                dataframe with elements information
            
            material: Material obj
                amfe material object
        
        return:
            connectivity : list 
                list of elements connectivity
        '''
        
        # add the nodes of the chosen group
        connectivity = [np.nan for i in range(len(elements_df))]
        for i, ele in enumerate(elements_df.values):
            no_of_nodes = amfe2no_of_nodes[elements_df.el_type.iloc[i]]
            connectivity[i] = np.array(ele[self.node_idx :
                                           self.node_idx + no_of_nodes],
                                       dtype=int)

        self.connectivity.extend(connectivity)

        # make a deep copy of the element class dict and apply the material
        # then add the element objects to the ele_obj list
        ele_class_dict = copy.deepcopy(self.element_class_dict)
        for i in ele_class_dict:
            ele_class_dict[i].material = material
        object_series = elements_df.el_type.map(ele_class_dict)
        self.ele_obj.extend(object_series.values.tolist())
        self._update_mesh_props()
        

        
        return connectivity

    def load_subset(self,subset_dict,material):
        '''
        Load a specifity mesh intersection to the main mesh with given material.
        
        It generates the connectivity list (mesh-class-property connectivity)
        which contains the element configuration as array
        and provides a map with pointers to Element-Objects (Tet, Hex etc.)
        which already contain information about material that is passed.
        Each element gets a pointer to such an element object.

        Parameters
        ----------
        subset_dict : dict
            dict[Key] = mesh_prop : {'phys_group', 'geom_entity', 'el_type', 'partition_id'}
            optional label of which the element should be chosen from. Standard is
            physical group.
            
            dict value = for mesh property which is to be chosen. Matches the group given
            in the gmsh file. For help, the function mesh_information or
            boundary_information gives the groups
            
        material : Material class
            Material class which will be assigned to the elements
        

        Returns
        -------
        None
        '''
                # asking for a group to be chosen, when no valid group is given
        df = self.el_df
        
        for sub_key in subset_dict:
            if sub_key not in df.columns:
                print('The given mesh property "' + str(sub_key) + '" is not valid!',
                      'Please enter a valid mesh prop from the following list:\n')
                for i in df.columns:
                    print(i)
                return
        
        submesh_key = ''
        for sub_key in subset_dict:
            key = subset_dict[sub_key]
            submesh_key = submesh_key + sub_key[0:2] + '_' + str(subset_dict[sub_key]) +  '_'
            while key not in pd.unique(df[sub_key]):
                self.mesh_information(sub_key)
                print('\nNo valid', sub_key, 'is given.\n(Given', sub_key,
                      'is', key, ')')
                subset_dict[sub_key] = int(input('Please choose a ' + sub_key + ' to be used as mesh: '))
                

        # make a pandas dataframe just for the desired elements
        sub_df = df.copy()
        try:
            for sub_key in subset_dict:
                sub_df = sub_df[sub_df[sub_key] == subset_dict[sub_key]]
        except:
            print('Make sure there is a intersection among groups')

        elements_df = sub_df
        # add the nodes of the chosen group
        connectivity = [np.nan for i in range(len(elements_df))]
        for i, ele in enumerate(elements_df.values):
            no_of_nodes = amfe2no_of_nodes[elements_df.el_type.iloc[i]]
            connectivity[i] = np.array(ele[self.node_idx :
                                           self.node_idx + no_of_nodes],
                                       dtype=int)

        self.connectivity.extend(connectivity)

        # make a deep copy of the element class dict and apply the material
        # then add the element objects to the ele_obj list
        ele_class_dict = copy.deepcopy(self.element_class_dict)
        for i in ele_class_dict:
            ele_class_dict[i].material = material
        object_series = elements_df.el_type.map(ele_class_dict)
        self.ele_obj.extend(object_series.values.tolist())
        self._update_mesh_props()

        # print some output stuff
        print('\nIntersection of the following groups: ')
        for sub_key in subset_dict:
            print( sub_key, subset_dict[sub_key])
        
        print('with', len(connectivity), ' elements successfully added.')
        print('Total number of elements in mesh:', len(self.ele_obj))
        print('*************************************************************')
        
        #elem_list  = elements_df['idx_gmsh']
        submesh = SubMesh(submesh_key,len(connectivity),elements_df,parent_mesh=self)
        return submesh

    def tie_mesh(self, master_key, slave_key, master_prop='phys_group',
                 slave_prop='phys_group', tying_type='fixed', robustness=4,
                 verbose=False, conform_slave_mesh=True, fix_mesh_dist=1E-3):
        '''
        Tie nonconforming meshes for a given master and slave side.


        Parameters
        ----------
        master_key : int or string
            mesh key of the master face mesh. The master face mesh has to be at
            least the size of the slave mesh. It is better, when the master
            mesh is larger than the slave mesh.
        slave_key : int or string
            mesh key of the slave face mesh or point cloud
        master_prop : string, optional
            mesh property for which master_key is specified.
            Default value: 'phys_group'
        slave_prop : string, optional
            mesh property for which slave_key is specified.
            Default value: 'phys_group'
        tying_type : string {'fixed', 'slide'}
            Mesh tying type. 'fixed' glues the meshes together while 'slide'
            allows for a sliding motion between the meshes.
        robustness : int, optional
            Integer value indicating, how many master elements should be
            considered for one slave node.

        Returns
        -------
        slave_dofs : ndarray, type: int
            slave dofs of the tied mesh
        row : ndarray, type: int
            row indices of the triplet description of the master slave
            conversion
        col : ndarray, type: int
            col indices of the triplet description of the master slave
            conversion
        val : ndarray, type: float
            values of the triplet description of the master slave conversion

        Notes
        -----
        The master mesh has to embrace the full slave mesh. If this is not the
        case, the routine will fail, a slave point outside the master mesh
        cannot be addressed to a specific element.

        '''
        df = self.el_df
        master_elements = df[df[master_prop]  == master_key]
        slave_elements = df[df[slave_prop]  == slave_key]

        master_nodes = master_elements.iloc[:, self.node_idx:].values
        master_obj =  master_elements.el_type.values
        slave_nodes = np.unique(slave_elements.iloc[:,self.node_idx:].values)
        slave_nodes = np.array(slave_nodes[np.isfinite(slave_nodes)], dtype=int)
        slave_dofs, row, col, val = master_slave_constraint(master_nodes,
            master_obj, slave_nodes, nodes=self.nodes, tying_type=tying_type,
            robustness=robustness, verbose=verbose,
            conform_slave_mesh=conform_slave_mesh, fix_mesh_dist=fix_mesh_dist)

        print('*'*80)
        print(('Tied mesh part {0} as master mesh to part {1} as slave mesh. \n'
              + 'In total {2} slave dofs were tied using the tying type {3}.'
              + '').format(master_key, slave_key, len(slave_dofs), tying_type)
             )
        print('*'*80)
        return (slave_dofs, row, col, val)

    def mesh_information(self, mesh_prop='phys_group'):
        '''
        print some information about the mesh that is being imported
        Attention: This information is not about the mesh that is already
        loaded for further calculation. Instead it is about the mesh that is
        found in an import-file!

        Parameters
        ----------
        mesh_prop : str, optional
            mesh property of the loaded mesh. This mesh property is the basis
            for selection of parts of the mesh for materials and boundary
            conditions. The default value is 'phys_group' which is the physical
            group, if the mesh comes from gmsh.

        Returns
        -------
        None

        '''
        df = self.el_df
        if mesh_prop not in df.columns:
            print('The given mesh property "' + str(mesh_prop) + '" is not valid!',
                  'Please enter a valid mesh prop from the following list:\n')
            for i in df.columns:
                print(i)
            return

        phys_groups = pd.unique(df[mesh_prop])
        print('The loaded mesh contains', len(phys_groups),
              'physical groups:')
        for i in phys_groups:
            print('\nPhysical group', i, ':')
            # print('Number of Nodes:', len(self.phys_group_dict [i]))
            print('Number of Elements:', len(df[df[mesh_prop] == i]))
            print('Element types appearing in this group:',
                  pd.unique(df[df[mesh_prop] == i].el_type))

        return

    def set_neumann_bc(self, key, val, direct, time_func=None,
                       shadow_area=False,
                       mesh_prop='phys_group'):
        '''
        Add group of mesh to neumann boundary conditions.

        Parameters
        ----------
        key : int
            Key of the physical domain to be chosen for the neumann bc
        val : float
            value for the pressure/traction onto the element
        direct : str 'normal' or ndarray
            array giving the direction, in which the traction force should act.
            alternatively, the keyword 'normal' may be given. Default value:
            'normal'.
        time_func : function object
            Function object returning a value between -1 and 1 given the
            input t:

            >>> val = time_func(t)

        shadow_area : bool, optional
            Flat setting, if force should be proportional to the shadow area,
            i.e. the area of the surface projected on the direction. Default
            value: 'False'.
        mesh_prop : str {'phys_group', 'geom_entity', 'el_type'}, optional
            label of which the element should be chosen from. Default is
            phys_group.

        Returns
        -------
        None
        '''
        
        #-----------------------------------------------------------------
        # creating submesh_boundary object to easily handle boundary contitions 
        # in subdomain. Only works if a domain has send set or load
        # ex. MechanicalSystem.set_domain({'phys_group':11},my_material) 
        #     Mesh.load_subset({'phys_group':11},my_material)
        
        if self.domain_dict:
            submesh_obj = self.get_submesh(mesh_prop,key)
            neu_boundary = Submesh_Boundary(submesh_obj,
                                             val, 
                                             direct, 
                                             time_func, 
                                             typeBC='neumann')
            
            nm_connectivity = neu_boundary.connectivity
            calc_newton_connectivity = False
            for sub_key, sub_obj in self.domain_dict.items():
                sub_obj.append_bondary_condition(neu_boundary)
        else:
            calc_newton_connectivity = True

        #-----------------------------------------------------------------
        
        df = self.el_df
        while key not in pd.unique(df[mesh_prop]):
            self.mesh_information(mesh_prop)
            logging.info('\nNo valid' + mesh_prop + 'is given.\n(Given' + \
                  mesh_prop +  'is' + str(key) +  ')')
            key = int(input('Please choose a ' + mesh_prop +
                            ' to be used for the Neumann Boundary conditions: '))

        # make a pandas dataframe just for the desired elements
        elements_df = df[df[mesh_prop] == key]
        ele_type = elements_df['el_type'].values
        
        # old implementation
        #--------------------------------------------------------------------
        # add the nodes of the chosen group
        if calc_newton_connectivity:
            nm_connectivity = [np.nan for i in range(len(elements_df))]

            for i, ele in enumerate(elements_df.values):
                nm_connectivity[i] = np.array(ele[self.node_idx : self.node_idx
                                              + amfe2no_of_nodes[ele_type[i]]],
                                              dtype=int)

        #--------------------------------------------------------------------
        
        self.neumann_connectivity.extend(nm_connectivity)


        # make a deep copy of the element class dict and apply the material
        # then add the element objects to the ele_obj list
        ele_class_dict = copy.deepcopy(self.element_boundary_class_dict)
        for i in ele_class_dict:
            ele_class_dict[i].__init__(val=val, direct=direct,
                                       time_func=time_func,
                                       shadow_area=shadow_area)

        object_series = elements_df['el_type'].map(ele_class_dict)
        self.neumann_obj.extend(object_series.values.tolist())
        # self._update_mesh_props() <- old implementation: not necessary!

        # print some output stuff
        logging.info('\n' + mesh_prop + str(key) + ' with ' + str(len(nm_connectivity)) + \
              ' elements successfully added to Neumann Boundary.')
        logging.info('Total number of neumann elements in mesh:' + str(len(self.neumann_obj)))
        logging.info('Total number of elements in mesh:' + str(len(self.ele_obj)))
        logging.info('*************************************************************')

    def set_dirichlet_bc(self, key, coord, mesh_prop='phys_group',
                         output='internal', id_matrix = None):
        '''
        Add a group of the mesh to the dirichlet nodes to be fixed. It sets the
        mesh-properties 'nodes_dirichlet' and 'dofs_dirichlet'

        Parameters
        ----------
        key : int
            Key for mesh property which is to be chosen. Matches the group given
            in the gmsh file. For help, the function mesh_information or
            boundary_information gives the groups
        coord : str {'x', 'y', 'z', 'xy', 'xz', 'yz', 'xyz'}
            coordinates which should be fixed
        mesh_prop : str {'phys_group', 'geom_entity', 'el_type'}, optional
            label of which the element should be chosen from. Default is
            phys_group.
        output : str {'internal', 'external'}
            key stating, boundary information is stored internally or externally
        id_matrix : dict
            dict that maps nodes to system Dofs



        Returns
        -------
        nodes : ndarray, if output == 'external'
            Array of nodes belonging to the selected group
        dofs : ndarray, if output == 'external'
            Array of dofs respecting the coordinates belonging to the selected
            groups

        '''
        
        
        #-----------------------------------------------------------------
        # creating submesh_boundary object to easily handle boundary contitions 
        # in subdomain. Only works if a domain has send set or load
        # ex. MechanicalSystem.set_domain({'phys_group':11},my_material) 
        #     Mesh.load_subset({'phys_group':11},my_material)
        
        if self.domain_dict:
            submesh_obj = self.get_submesh(mesh_prop,key)
            diri_boundary = Submesh_Boundary(submesh_obj,
                                             val= 0.0, 
                                             direction = coord, 
                                             time_func=None, 
                                             typeBC='dirichlet')
            
            for sub_key, sub_obj in self.domain_dict.items():
                sub_obj.append_bondary_condition(diri_boundary)
        
        #-----------------------------------------------------------------
        
        # asking for a group to be chosen, when no valid group is given
        df = self.el_df
        while key not in pd.unique(df[mesh_prop]):
            self.mesh_information(mesh_prop)
            print('\nNo valid', mesh_prop, 'is given.\n(Given', mesh_prop,
                  'is', key, ')')
            key = int(input('Please choose a ' + mesh_prop +
                            ' to be chosen for Dirichlet BCs: '))

        # make a pandas dataframe just for the desired elements
        elements_df = df[df[mesh_prop] == key]
        # pick the nodes, make them unique and remove NaNs
        all_nodes = elements_df.iloc[:, self.node_idx:]
        unique_nodes = np.unique(all_nodes.values.reshape(-1))
        unique_nodes = unique_nodes[np.isfinite(unique_nodes)]

        # build the dofs_dirichlet, a list containing the dirichlet dofs:
        dofs_dirichlet = []
        
        if id_matrix is None:
            # Old implementation which not uses a ID_matrix
            if 'x' in coord:
                dofs_dirichlet.extend(unique_nodes * self.no_of_dofs_per_node)
            if 'y' in coord:
                dofs_dirichlet.extend(unique_nodes * self.no_of_dofs_per_node + 1)
            if 'z' in coord and self.no_of_dofs_per_node > 2:
                # TODO: Separate second if and throw error or warning
                dofs_dirichlet.extend(unique_nodes * self.no_of_dofs_per_node + 2)
        else:
            # new implementation which uses a ID_matrix
            id_dof = []
            if 'x' in coord:
                id_dof.extend([0])
            if 'y' in coord:
                id_dof.extend([1])
            if 'z' in coord and self.no_of_dofs_per_node > 2:
                # TODO: Separate second if and throw error or warning
                id_dof.extend([2])

            for node in unique_nodes:
                for dof in id_dof:
                    dofs_dirichlet.extend([id_matrix[int(node)][dof]])
        
        dofs_dirichlet = np.array(dofs_dirichlet, dtype=int)
        # TODO: Folgende Zeilen sind etwas umstaendlich, erst conversion to list, dann extend und dann zurueckconversion
        nodes_dirichlet = unique_nodes
        # nodes_dirichlet = self.nodes_dirichlet.tolist()
        # nodes_dirichlet.extend(unique_nodes)
        # nodes_dirichlet = np.array(nodes_dirichlet, dtype=int)

        if output is 'internal':
            dofs_dirichlet = np.append(dofs_dirichlet, self.dofs_dirichlet)
            self.dofs_dirichlet = np.unique(dofs_dirichlet)

            nodes_dirichlet = np.append(nodes_dirichlet, self.nodes_dirichlet)
            self.nodes_dirichlet = np.unique(nodes_dirichlet)

        # print some output stuff
        logging.info('\n' + mesh_prop + str(key) + 'with' + str(len(unique_nodes)) + \
              ' nodes successfully added to Dirichlet Boundaries.')
        logging.info('Total number of nodes with Dirichlet BCs:' +  str(len(self.nodes_dirichlet)))
        logging.info('Total number of constrained dofs:' +  str(len(self.dofs_dirichlet)))
        logging.info('*************************************************************')
        if output is 'external':
            return nodes_dirichlet, dofs_dirichlet

    def deflate_mesh(self):
        '''
        Deflate the mesh, i.e. delete nodes which are not connected to an
        element.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''

        nodes_vec = np.concatenate(self.connectivity)
        elements_on_node = np.bincount(np.array(nodes_vec, dtype=int))
        mask = np.zeros(self.nodes.shape[0], dtype=bool)
        # all nodes which show up at least once
        mask[:len(elements_on_node)] = elements_on_node != 0
        idx_transform = np.zeros(len(self.nodes), dtype=int)
        idx_transform[mask] = np.arange(len(idx_transform[mask]))
        self.nodes = self.nodes[mask]
        # deflate the connectivities
        for i, nodes in enumerate(self.connectivity):
            self.connectivity[i] = idx_transform[nodes]
        for i, nodes in enumerate(self.neumann_connectivity):
            self.neumann_connectivity[i] = idx_transform[nodes]

        # deflate the element_dataframe
        df = self.el_df
        for col in df.iloc[:,self.node_idx:]:
            nan_mask = np.isfinite(df[col].values)
            indices = np.array(df[col].values[nan_mask], dtype=int)
            df[col].values[nan_mask] = idx_transform[indices]

        self._update_mesh_props()
        print('**************************************************************')
        print('Mesh successfully deflated. ',
              '\nNumber of nodes in old mesh:', len(mask),
              '\nNumber of nodes in deflated mesh:', np.count_nonzero(mask),
              '\nNumer of deflated nodes:', len(mask) - np.count_nonzero(mask))
        print('**************************************************************')

    def save_mesh_xdmf(self, filename, field_list=None, bmat=None, u=None, timesteps=None):
        '''
        Save the mesh in hdf5 and xdmf file format.

        Parameters
        ----------
        filename : str
            String consisting the path and the filename
        field_list : list
            list containing the fields to be exported. The list is a list of
            tupels containing the array with the values in the columns and a
            dictionary with the attribute information:

                >>> # example field list with reduced displacement not to export
                >>> # ParaView and strain epsilon to be exported to ParaView
                >>> field_list = [(q_red, {'ParaView':False, 'Name':'q_red'}),
                                  (eps, {'ParaView':True,
                                         'Name':'epsilon',
                                         'AttributeType':'Tensor6',
                                         'Center':'Node',
                                         'NoOfComponents':6})]
        bmat : csrMatrix
            CSR-Matrix describing the way, how the Dirichlet-BCs are applied:
            u_unconstr = bmat @ u_constr

        u : nparray
            matrix with displacement vectors as columns for different timesteps
            
        timesteps : nparray
            vector with timesteps a displacement vector is stored in u


        Returns
        -------
        None

        Notes
        -----
        Only one homogeneous mesh is exported. Thus only the mesh made of the
        elements which occur most often is exported. The other meshes are
        discarded.

        '''
        # generate a zero displacement if no displacements are passed.
        if not u or not timesteps:
            u = [np.zeros((self.no_of_nodes * self.no_of_dofs_per_node,)),]
            timesteps = np.array([0])


        # determine the part of the mesh which has most elements
        # only this part will be exported!
        ele_types = np.array([obj.name for obj in self.ele_obj], dtype=object)
        el_type_export = np.unique(ele_types)
        connectivties_dict = dict()
        for el_type in el_type_export:
            # Boolean matrix giving the indices for the elements to export
            el_type_ix = (ele_types == el_type)

            # select the nodes to export an make an array of them
            # As the list might be ragged, it has to be put to list and then to
            # array again.
            connectivity_export = np.array(self.connectivity)[el_type_ix]
            connectivity_export = np.array(connectivity_export.tolist())
            connectivties_dict[el_type] = connectivity_export

        # make displacement 3D vector, as paraview only accepts 3D vectors
        q_array = np.array(u, dtype=float).T
        if self.no_of_dofs_per_node == 2:
            tmp_2d = q_array.reshape((self.no_of_nodes,2,-1))
            x, y, z = tmp_2d.shape
            tmp_3d = np.zeros((x,3,z))
            tmp_3d[:,:2,:] = tmp_2d
            q_array = tmp_3d.reshape((-1,z))

        h5_q_dict = {'ParaView':True,
                     'AttributeType':'Vector',
                     'Center':'Node',
                     'Name':'Displacement',
                     'NoOfComponents':3}

        h5_time_dict = {'ParaView':True,
                        'Name':'Time'}

        if field_list is None:
            new_field_list = []
        else:
            new_field_list = field_list.copy()

        new_field_list.append((q_array, h5_q_dict))

        check_dir(filename)

        # write the hdf5 file with the necessary attributes
        with h5py.File(filename + '.hdf5', 'w') as f:
            # export mesh with nodes and topology
            h5_nodes = f.create_dataset('mesh/nodes', data=self.nodes)
            h5_nodes.attrs['ParaView'] = True
            for el_type in connectivties_dict:
                h5_topology = f.create_dataset('mesh/topology/' + el_type,
                                               data=connectivties_dict[el_type],
                                               dtype=np.int)
                h5_topology.attrs['ParaView'] = True
                h5_topology.attrs['TopologyType'] = amfe2xmf[el_type]

            # export timesteps
            h5_time = f.create_dataset('time', data=np.array(timesteps))
            h5_set_attributes(h5_time, h5_time_dict)

            # export bmat if given
            if bmat is not None:
                h5_bmat = f.create_group('mesh/bmat')
                h5_bmat.attrs['ParaView'] = False
                for par in ('data', 'indices', 'indptr', 'shape'):
                    array = np.array(getattr(bmat, par))
                    h5_bmat.create_dataset(par, data=array, dtype=array.dtype)

            # export fields in new_field_list
            for data_array, data_dict in new_field_list:

                # consider heterogeneous meshes:
                # export Cell variables differently than other time variables
                export_cell = False
                if 'Center' in data_dict:
                    if data_dict['Center'] == 'Cell':
                        export_cell = True

                if export_cell:
                    for el_type in connectivties_dict:
                        # Boolean matrix giving the indices for the elements to export
                        el_type_ix = (ele_types == el_type)
                        location = 'time_vals_cell/{}/{}'.format(
                                data_dict['Name'], el_type)
                        h5_dataset = f.create_dataset(location,
                                                      data=data_array[el_type_ix])
                        h5_set_attributes(h5_dataset, data_dict)
                else:
                    h5_dataset = f.create_dataset('time_vals/' + data_dict['Name'],
                                                  data=data_array)
                    h5_set_attributes(h5_dataset, data_dict)

        # Create the xdmf from the hdf5 file
        create_xdmf_from_hdf5(filename + '.hdf5')
        return

    def split_in_groups(self, group_type = 'phys_group', parent=None, elem_dataframe =None):        
        '''creating submeshs based on different types of elements grouping 
        This routines update the create groups variables in the mesh class
           
        inputs
            group_type = {'phys_group', 'geom_entity', 'el_type', 'partition_id'}
            parent = Parent instance which contains all the information of the group.
            Ex. parent = mesh instance
           
        output
            List of SubMeshs
           
        '''
        # creating element list with tag to separate elements in groups
        
        
        if elem_dataframe is None:
            elem_dataframe = self.el_df
        
        if group_type in list(elem_dataframe.columns):      
            elem_group_series = elem_dataframe[group_type]
        
        else:
            print('List of avaliables elem_group series:')
            print(list(elem_dataframe.columns))
            raise ValueError('WARNING. Please provide a elem_group series for splitting mesh in groups.')
            return None
    
        # creating local dictionary for grouping
        groups_dict = {}        
        for elem, group_id in elem_group_series.iteritems():
            if group_id is not None:
                try:
                    group_id = int(group_id)
                except:
                    logging.debug('Warning! Group id is type string. Some function may not work properly!')
                    group_id = group_id

                try:
                    groups_dict[group_id].append(elem)
                    
                except:    
                    groups_dict[group_id] = []
                    groups_dict[group_id].append(elem)
                
            else:
                print('Warning! Could not assign Element %i to group! Element was move to the None group' %elem)
                group_id = None
                #continue      
        
        self.groups_dict = groups_dict
        
        self.groups = {}
        for key in self.groups_dict:
            sub_num_of_elem = len(elem_group_series)
            if parent is None:
               parent = self
               
            #
            # passsing to submesh only a part of data frame which correspond to the submesh
            elem_dataframe_i = elem_dataframe.loc[groups_dict[key],:]

            # selecting if parent mesh will be a copy or a pointer
            if group_type=='partition_id':
                copy_parent = True
            else:
                copy_parent = False

            submesh = SubMesh(key, sub_num_of_elem, elem_dataframe_i, parent, groups_dict[key], copy_parent)
            self.groups[key] = submesh
            self.__last_tag__ = group_type

    def get_elem_types(self,elem_id_list=None):
        
        if elem_id_list is not None:
            database = self.el_df.iloc[elem_id_list]
        else:
            database = self.el_df
        
        return set(database['el_type'])
    
    def get_phys_group_types(self,elem_id_list=None):
        
        if elem_id_list is not None:
            database = self.el_df.iloc[elem_id_list]
        else:
            database = self.el_df
        
        return set(database['phys_group'])
        
    def get_submesh(self,tag,value):
        ''' This function returns a submeh obj with a subset of the mesh based on tag 
        and value. Also it is possible to specify a direction, which is great for boundary conditions
        '''
        
        if not(self.groups): 
            self.split_in_groups(tag)
            self.__last_tag__ = tag
        
        elif self.__last_tag__ != tag:
            self.split_in_groups(tag)
            
        try:
            domain = self.groups[value]            
            return domain
        
        except:
            print("Please select a valid key value")

    def set_domain(self,tag,value):
        ''' This function returns a submeh obj with a subset of the mesh based on tag 
        and value. Also it is possible to specify a direction, which is great for boundary conditions
        '''
        
        if not(self.groups): 
            self.split_in_groups(tag)
            self.__last_tag__ = tag
        
        elif self.__last_tag__ != tag:
            self.split_in_groups(tag)
            
        try:
            domain = self.groups[value]
            self.domain_dict[value] = self.groups[value]
            
            # creating elements dictionaty of the domain
            elem_start_index = domain.parent_mesh.node_idx
            elem_last_index = len(domain.elem_dataframe.columns)
            elem_connec = domain.elem_dataframe.iloc[:,elem_start_index:elem_last_index]
            elem_connec = elem_connec.dropna(1) # removing columns with NaN
            elem_connec = elem_connec.astype(int) # convert all to int
            self.elements_dict = elem_connec
            domain.parent_mesh = copy.deepcopy(self) # update parent mesh
            return domain
        
        except:
            print("Please select a valid key value")

    def translation(self, offset):
        ''' This function translate a set of nodes based on the ref_point_vector 
        in the reference coordinate system
        
        arguments:
        offset : np.array
            np.array to apply the translation
    
        return :
            new_mesh : Mesh Obj
                new_mesh with the new nodes coordinates
        '''
        
        coord = self.nodes
        num_nodes, dim = coord.shape
        ref_point_vector = offset[:dim]
        center_vector = np.tile(ref_point_vector , [num_nodes,1])
    
        trans_coord = coord + center_vector 

        # creating a new mesh
        new_mesh = copy.deepcopy(self)
        new_mesh.nodes = trans_coord
        
        return new_mesh

    def rot_z(self ,alpha, unit='deg', ref_point_vector = np.array([0.0,0.0,0.0])):
    
        ''' This function rotate a set of nodes based on the rotation angule alpha
        and a reference coordinate system
    
        arguments:
            
            alpha: float
                angule to apply rotation in z axis
            
            unit: str
                unit type of alpha, internally alpha must be in rad
            
            ref_point_vector: np.array
                np.array with the vectors of a coordinate system [e1,e2,e3]
        
        '''
    
        coord = self.nodes
        num_nodes, dim = coord.shape
        if unit=='deg':
            # transforme to rad
            alpha_rad = np.deg2rad(alpha)
        elif unit=='rad':
            alpha_rad = alpha
        else:
            raise('The selected unit is %s. This unit is not supported.' %unit)

        
        cos_a = np.cos(alpha_rad)
        sin_a = np.sin(alpha_rad)
        R = np.matrix([[cos_a, sin_a,0],
                       [-sin_a, cos_a,0],
                       [0.0, 0.0, 1.0]])
    
        R = R[:dim,:dim]
        #checking reference point to rotate coordinates
        bool_coord = False
        if np.any(ref_point_vector != 0.0):
            bool_coord = True

        if bool_coord:
            ref_point_vector = ref_point_vector[:dim]
            center_vector = np.tile(ref_point_vector , [num_nodes,1])
            trans_coord = coord - center_vector 
        else:
            trans_coord = coord 

        global_nodes = trans_coord.dot(R)
    

        # back to original coord
        if bool_coord:
            global_nodes = global_nodes + center_vector



        # creating a new mesh
        new_mesh = copy.deepcopy(self)
        new_mesh.nodes = np.array(global_nodes)
        
        return new_mesh



class SubMesh():
    ''' The SubMesh is a class which provides a easy data structure for dealing 
    with subsets of the global mesh
    '''
    def __init__(self,t,num_of_elem, elem_dataframe=None, parent_mesh=None, elem_list = None, copy_parent=True):
    
        if copy_parent:
            self.parent_mesh = copy.deepcopy(parent_mesh)
        else:
            self.parent_mesh = parent_mesh

        self.num_of_elem =num_of_elem
        self.key = t
        self.elements_list = elem_list
        self.elem_dataframe = elem_dataframe
        self.partitions = {}
        self.__material__ = None
        self.elements_dict = {}
        self.elem_connect_to_node_dict = {}
        #self.subset_list()
        
        self.interface_elements_dict = {}
        self.has_partitions = False
        self.is_partition = False
        self.neighbor_partitions = []
        self.interface_nodes_dict = {}
        self.local_connectivity = []
        self.local_node_list = []
        self.global_to_local_dict = {}
        self.direction = None
        self.neumann_submesh = []
        self.dirichlet_submesh = []
        self.global_node_list = []
        self.create_node_list()
        self.problem_type = 2  # self.problem_type = 2 -> 2D problem / self.problem_type = 3 -> 3D problem; 
    
    def create_elem_dict(self):
        ''' create a list with element number as key
        and node list as values
        '''
        node_idx = self.parent_mesh.node_idx
        for index, row in self.elem_dataframe.iterrows():
            self.elements_dict[index] = list(row.iloc[node_idx:].dropna().astype(int))
        
        return self.elements_dict
    
    def get_element(self,elem_id):
        ''' Get the nodes assossiate with elem_id

        parameters
            elem_id : int
                element id
        return
            list of nodes assossiated with element
        '''
        if not self.elements_dict:
            self.create_elem_dict()
        
        if elem_id in self.elements_dict:
            return self.elements_dict[elem_id]
        else:
            print('Element number does not belong to the Submesh, please select another element number')
            return None

    def get_node_coord(self, node_id):
        ''' get the coordinates given a node_id
        
        paramenters
            node_id: int
                node identifier
        return 
            node_coord : np.array
        '''
        node_coord = self.parent_mesh.nodes[node_id]
        return node_coord
    
    def get_elem_coord(self, elem_id):
        ''' get the coordinates given a elem_id
        
        paramenters
            elem_id: int
                element identifier
        return 
            elem_coord : np.array
        '''
        node_list = self.get_element(elem_id)
        elem_coord = []
        for node_id in node_list:
            elem_coord.append(list(self.get_node_coord(node_id)))
        
        return np.array(elem_coord)
    
    def get_normal_to_element(self, elem_id, orientation = 1.0):
        ''' get the normal vector given a elem_id
        
        paramenters
            elem_id: int
                element identifier
            orientation : float
                change the orientation of the normal vector either 1.0 or -1.0
            
        return 
            normal_vector : np.array
        '''
        coord = self.get_elem_coord(elem_id)
        vector1 = coord[1] - coord[0]
        try:
            # for 3D cases
            vector2 = coord[2] - coord[0]
            normal_vec  = np.cross(vector1, vector2)
            unit_normal_vec = normal_vec/np.linalg.norm(normal_vec)    
        except:
            # for 2D cases
            normal_vec = np.array([-vector1[1], vector1[0],0.0])
            unit_normal_vec = normal_vec/np.linalg.norm(normal_vec)    
            
        return orientation*unit_normal_vec
    
    def create_elem_connect_to_node(self):
        ''' This function creates a dict
        with nodes as keys which points to elements connected 
        to the node
        
        return self.elem_connect_to_node_dict
        '''
        
        self.elem_connect_to_node_dict = {}
        
        if not self.elements_dict:
            self.create_elem_dict()
        
        for elem_id, node_list in self.elements_dict.items():
            for node_id in node_list:
                if node_id in self.elem_connect_to_node_dict:
                    self.elem_connect_to_node_dict[node_id].append(elem_id)
                else:
                    self.elem_connect_to_node_dict[node_id] = [elem_id]
        
        return self.elem_connect_to_node_dict
    
    def get_normal_to_node(self, node_id, method = 'average' , orientation = 1.0):
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
        if not self.elem_connect_to_node_dict:
            self.create_elem_connect_to_node()
        
        elem_list = self.elem_connect_to_node_dict[node_id]
        
        
        if method == 'average':
            pass
        
        elif method == 'first':
            elem_list = [elem_list[0]]
        
        else:
            print('Methods is nor implemented. Select either "first" or "average"')
            
        normal_vec = np.zeros(3)
        div = 0.0
        for elem_id in elem_list:
            normal_vec += self.get_normal_to_element(elem_id,orientation)
            div += 1.0
        
        normal_vec = normal_vec/div
        unit_normal_vec = normal_vec/np.linalg.norm(normal_vec)
        return unit_normal_vec
    
    def set_material(self,material):
        self.__material__ = material
        
    def add_local_mesh(self,local_connectivity, local_node_list,global_to_local_dict = None, local_to_global_dict = None ):
        
        self.local_connectivity = local_connectivity
        self.local_node_list = local_node_list
        self.global_to_local_dict = global_to_local_dict
        self.local_to_global_dict = local_to_global_dict
        
    def subset_list(self):
        
        ''' This methods return a new dataframe with
        only the rows at the self.elements_list
        '''
        self.elem_dataframe = self.parent_mesh.el_df.iloc[self.elements_list,:]
    
    def get_submesh(self,tag,value):
        ''' Get a submesh from parent_mesh based on tag and key value

        paramenters:
            tag : str
                string with group label
            value : int
                value of the group label to be select

        return
            SubMesh : obj
                parent SubMesh object based on tag and value

        '''
        return self.parent_mesh.get_submesh(tag,value)

    def create_node_list(self):
        node_list = []
        elem_start_index = self.parent_mesh.node_idx
        elem_last_index = len(self.parent_mesh.el_df.columns)
        
        for node_id in range(elem_start_index, elem_last_index):
            nodes = self.elem_dataframe.iloc[:,node_id].tolist()
            node_list.extend(nodes)
        
        # removing duplicate node
        node_set = set(node_list)
        node_set = {int(x) for x in node_set if x==x}
        self.global_node_list = list(node_set)        
        
        return self.global_node_list
                
    def __inherit_neumann_nodes__(self,parent_neumann_submesh):
        
        for parent_obj in parent_neumann_submesh:
            for node in parent_obj.submesh.global_node_list:
                if node in self.global_node_list:
                    self.neumann_submesh.append(parent_obj)
                    break

    def __inherit_dirichlet_nodes__(self,parent_dirichlet_submesh):
        
        for parent_obj in parent_dirichlet_submesh:
            for node in  parent_obj.submesh.global_node_list:
                if node in self.global_node_list:
                    self.dirichlet_submesh.append(parent_obj)
                    break        
    
    def split_in_partitions(self, group_type = 'partition_id'):
        
        Mesh.split_in_groups(self,group_type, self.parent_mesh, self.elem_dataframe)
        

        # for all kinds of groups
        for key in self.groups_dict:
            self.groups[key].__inherit_neumann_nodes__(self.neumann_submesh)
            self.groups[key].__inherit_dirichlet_nodes__(self.dirichlet_submesh)
            self.groups[key].set_material(self.__material__)


        if group_type == 'partition_id':
            self.has_partitions = True  
            
            for key in self.groups_dict:
                self.groups[key].is_partition = True
                self.groups[key].__find_interior_and_interface_element__()
                #self.groups[key].__inherit_neumann_nodes__(self.neumann_submesh)
                #self.groups[key].__inherit_dirichlet_nodes__(self.dirichlet_submesh)
                #self.groups[key].set_material(self.__material__)

            # find interface nodes
            for sub1_key in self.groups_dict:
                sub1 = self.groups[sub1_key]
                for sub2_key in sub1.neighbor_partitions:
                    self.__find_interface_nodes__(sub1_key,sub2_key)   
            
        return self.groups

    def __find_interface_nodes__(self,sub1_key,sub2_key):
        
        sub1 = self.groups[sub1_key]
        sub2 = self.groups[sub2_key]
        sub1_nodes = []
        sub2_nodes = []
        
        elem_start_index = self.parent_mesh.node_idx
        elem_last_index = len(self.parent_mesh.el_df.columns)
        #elements_dict = self.parent_mesh.el_df.iloc[:,elem_start_index:elem_last_index]
        #elements_dict = elements_dict.dropna(0).astype(int)  # remove rows with NaN   
        
        elements_dict = self.elem_dataframe.iloc[:,elem_start_index:elem_last_index]
        elements_dict = elements_dict.dropna(0).astype(int)  # remove rows with NaN   

        if sub1.neighbor_partitions and not(sub1_key in sub2.interface_nodes_dict.keys()):    
            print('Extract interface node from sub_%i and sub_%i' %(sub1_key,sub2_key))
            
            for elem in sub1.interface_elements_dict[sub2_key]:
                nodes1 = list(elements_dict.loc[elem])
                sub1_nodes.extend(nodes1)
            for elem in sub2.interface_elements_dict[sub1_key]:
                nodes2 = list(elements_dict.loc[elem])
                sub2_nodes.extend(nodes2)
            
            # exclud duplicated nodes
            #sub1_nodes = list(set(sub1_nodes))
            #sub2_nodes = list(set(sub2_nodes))
            
            nodes = list(set(sub1_nodes).intersection(sub2_nodes))
            sub1.interface_nodes_dict[sub2_key] = []
            sub2.interface_nodes_dict[sub1_key] = []
            sub1.interface_nodes_dict[sub2_key].extend(nodes)
            sub2.interface_nodes_dict[sub1_key].extend(nodes)
           
            #for node in sub1_nodes:
            #    if node in sub2_nodes:
            #        try:
            #            sub1.interface_nodes_dict[sub2_key].append(node)
            #            sub2.interface_nodes_dict[sub1_key].append(node)
            #        except:    
            #            sub1.interface_nodes_dict[sub2_key] = []
            #            sub2.interface_nodes_dict[sub1_key] = []
            #            sub1.interface_nodes_dict[sub2_key].append(node)
            #            sub2.interface_nodes_dict[sub1_key].append(node)
                        
        elif sub1_key in sub2.interface_nodes_dict.keys():
            print('Interface nodes from sub_%i and sub_%i already extracted' %(sub1_key,sub2_key))
        
        else:
            print('WARNING! This mesh group has no partition.')
    
    def __find_interior_and_interface_element__(self):
        
        
        elements_partitions_neighbors_series = self.elem_dataframe['partitions_neighbors']
        for key, neighbor_list in  elements_partitions_neighbors_series.iteritems():
            
            if neighbor_list is not None:
                for neighbor in neighbor_list:
                    self.neighbor_partitions.append(-neighbor)
                    try:
                        self.interface_elements_dict[-neighbor].append(key)
                    except:
                        self.interface_elements_dict[-neighbor] = []
                        self.interface_elements_dict[-neighbor].append(key)
        
        # removing duplicate elements        
        self.neighbor_partitions = list(set(self.neighbor_partitions))
        for n in self.neighbor_partitions:
            self.interface_elements_dict[n] = list(set(self.interface_elements_dict[n]))
    
    def append_bondary_condition(self,submesh):
        
        if submesh.type == 'neumann':
            self.append_neumann_bc(submesh)
        
        elif submesh.type == 'dirichlet':
            self.append_dirichlet_bc(submesh)
        
        else:
            print('Boundary type %s not supported!')
    
    def append_neumann_bc(self,neu_submesh):
        self.neumann_submesh.append(neu_submesh)
    
    def append_dirichlet_bc(self,dir_submesh):
        self.dirichlet_submesh.append(dir_submesh)
    
    def get_submesh_connectivity(self):
        ''' this method get then connectivity matrix of a subdomain
        this methods works for different types of element

        The connectivity is a list where the index represent the number of the element
        in a local reference

        return 
            connectivity : list
                dict with elements connectivity, the id are the number
                local references, and can not be associated with global mesh
                object

        '''

        connectivity = []
        node_idx = self.parent_mesh.node_idx
        elements_df = self.elem_dataframe
        for index, ele in elements_df.iterrows():
            connectivity.append(list(ele[node_idx:].dropna().astype(int)))

        return connectivity

    def get_element_type_list(self):
        ''' This function return a list of element type in submesh

        return 
            elem_type_list : list
                list with element types in SubMesh
        '''
        return list(set(self.elem_dataframe.el_type))

    def rot_z(self ,alpha, unit='deg', ref_point_vector = np.array([0.0,0.0,0.0])):
        ''' This function rotate a set of nodes based on the rotation angule alpha
        and a reference coordinate system. In the SubMesh Class this rotation is done 
        in the parent_mesh object
    
        arguments:
            
            alpha: float
                angule to apply rotation in z axis
            
            unit: str
                unit type of alpha, internally alpha must be in rad
            
            ref_point_vector: np.array
                np.array with the vectors of a coordinate system [e1,e2,e3]
        '''
       
        new_mesh = self.parent_mesh.rot_z(alpha, unit, ref_point_vector)
        self.parent_mesh = new_mesh
        return  None
        
class Submesh_Boundary():
    
    def __init__(self,submesh_obj,val= 0.0, direction='normal', time_func=None, typeBC='neumann'):
        ''' This class create and data structure to easily reuse boundary conditions 
        in submesh class        
        '''
        
        amfe_mesh = Mesh()
        self.submesh = submesh_obj 
        self.elements_list = submesh_obj.elem_dataframe.index.tolist()
        
        self.value = val
        self.direction = direction
        self.type = typeBC
        
        # make a deep copy of the element class dict and apply the material
        # then add the element objects to the ele_obj list
        
        self.connectivity = []
        object_series = []
        
        if typeBC == 'neumann':
            self.neumann_obj = []
            elem_start_index = self.submesh.parent_mesh.node_idx
            elem_last_index = len(self.submesh.elem_dataframe.columns)
            elem_connec = self.submesh.elem_dataframe.iloc[:,elem_start_index:elem_last_index]
            elem_connec = elem_connec.dropna(1) # removing columns with NaN
            for elem_key in self.elements_list: 
                
                self.connectivity.append(np.array(elem_connec.loc[elem_key]))
                #elem_gmsh_key = self.submesh.parent_mesh.elements_type_dict[elem_key]
                elem_type = self.submesh.elem_dataframe['el_type'].loc[elem_key]
                
                elem_neumann_class_dict = copy.deepcopy(amfe_mesh.element_boundary_class_dict[elem_type])
                elem_neumann_class_dict.__init__(val, direction)
                
                object_series.append(elem_neumann_class_dict)
            #object_series = elements_df['el_type'].map(ele_class_dict)
            self.neumann_obj.extend(object_series)                  
        