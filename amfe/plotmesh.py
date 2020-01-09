# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 14:59:24 2017

@author: ge72tih
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.tri as tri
import mpl_toolkits.mplot3d as a3

from matplotlib import collections  as mc
from matplotlib import colors, transforms
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import numpy as np
import itertools
import copy
import matplotlib.patches as mpatches

Two_D_elem_list = ['Tri3','Tri6','Quad4','Quad8','Bar2Dlumped','Quad4Boundary']
Tri_D_elem_list = ['Tet4','Tet10','Hexa8','Hexa20','Prism6']

Boundary_elem_list = ['LineLinearBoundary',
                      'LineQuadraticBoundary',
                      'Tri3Boundary',
                      'Tri6Boundary', 
                      'Quad4Boundary', 
                      'Quad8Boundary']
    

colors =['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
colors =[(0.5,0.3,0.3),
         (0,0,0.8),
         (0,0.5,0),
         (0.8,0.8,0),
         (.3,.3,0),
         (.6,.6,.8),
         (0,.5,.5),
         (1,0,0),
         (0.5,0.25,0)]

colors =  colors*10

max_number_of_color = len(colors)

def plot_submesh(submesh_obj,ax=None, color_id = None,plot_1d = False):
    ''' This function plots 2D meshes, suport elements are 
    Triagules = Tri3, Tri6
    lines = straight_line, straight_line
    
    Arguments
    submesh_obj: submesh instance
    ax: matplotlib axes
    
    Return
    ax: matplotlib axes
    '''
           
    connectivity_tri = []
    lines = []
    patches = []
    x_tri = []
    y_tri = []
    lines_counter = 0
    points = []

    try:
        submesh_obj.parent_mesh.elements_type_dict = submesh_obj.parent_mesh.el_df['el_type'].to_dict()  
        elem_start_index = submesh_obj.parent_mesh.node_idx
        elem_last_index = len(submesh_obj.parent_mesh.el_df.columns)
        elements_dict = submesh_obj.parent_mesh.el_df.iloc[:,elem_start_index:elem_last_index]
        submesh_obj.parent_mesh.elements_dict = elements_dict.to_dict('index')
        d = submesh_obj.parent_mesh.elements_dict
        submesh_obj.elements_list = submesh_obj.elem_dataframe.index.tolist()
        
        
        for key, value in d.items():
            submesh_obj.parent_mesh.elements_dict[key] = [ int(j) for i, j in value.items() if not(np.isnan(j))]
        
        submesh_obj.parent_mesh.nodes_list = submesh_obj.parent_mesh.nodes.tolist()
        submesh_obj.parent_mesh.nodes_dict = {key:value for key,value in enumerate(submesh_obj.parent_mesh.nodes_list)}
        
    except:
        return None
    
    for elem in submesh_obj.elements_list:
        
        elem_connec = submesh_obj.parent_mesh.elements_dict[elem]
        if submesh_obj.parent_mesh.elements_type_dict[elem] == 2 or submesh_obj.parent_mesh.elements_type_dict[elem] == 'Tri3':
            connectivity_tri.append(elem_connec)
            
        elif submesh_obj.parent_mesh.elements_type_dict[elem] == 9 or submesh_obj.parent_mesh.elements_type_dict[elem] == 'Tri6':
            connectivity_tri.append(elem_connec[0:3])
            
        elif submesh_obj.parent_mesh.elements_type_dict[elem] == 3 or submesh_obj.parent_mesh.elements_type_dict[elem] == 'Quad4':
            #connectivity_quad.append(elem_connec) 
            elem_nodes = []
            for node in elem_connec:
                elem_nodes.append(submesh_obj.parent_mesh.nodes_dict[node][0:2])
            
            polygon = Polygon(elem_nodes, True)
            patches.append(polygon)
        
        elif submesh_obj.parent_mesh.elements_type_dict[elem] == 1 or submesh_obj.parent_mesh.elements_type_dict[elem] == 'straight_line':
           node1 = submesh_obj.parent_mesh.elements_dict[elem][0]
           node2 = submesh_obj.parent_mesh.elements_dict[elem][1]
           x = [submesh_obj.parent_mesh.nodes_dict[node1][0],submesh_obj.parent_mesh.nodes_dict[node1][1]]            
           y = [submesh_obj.parent_mesh.nodes_dict[node2][0],submesh_obj.parent_mesh.nodes_dict[node2][1]]
           lines.append([x,y])
           
        elif submesh_obj.parent_mesh.elements_type_dict[elem] == 8 or submesh_obj.parent_mesh.elements_type_dict[elem] == 'quadratic_line':
           node1 = submesh_obj.parent_mesh.elements_dict[elem][0]
           node2 = submesh_obj.parent_mesh.elements_dict[elem][1]
           x = [submesh_obj.parent_mesh.nodes_dict[node1][0],submesh_obj.parent_mesh.nodes_dict[node1][1]]            
           y = [submesh_obj.parent_mesh.nodes_dict[node2][0],submesh_obj.parent_mesh.nodes_dict[node2][1]]
           lines.append([x,y])

        elif submesh_obj.parent_mesh.elements_type_dict[elem] == 0 or submesh_obj.parent_mesh.elements_type_dict[elem] == 'point':
           node1 = submesh_obj.parent_mesh.elements_dict[elem][0]
           x = submesh_obj.parent_mesh.nodes_dict[node1][0]          
           y = submesh_obj.parent_mesh.nodes_dict[node1][1]
           points.append([x,y])
                
   
    for node in submesh_obj.parent_mesh.nodes_dict:
        x_tri.append(submesh_obj.parent_mesh.nodes_dict[node][0])
        y_tri.append(submesh_obj.parent_mesh.nodes_dict[node][1])
                
    if ax == None:
        fig = plt.figure()
        ax = plt.axes()    
    
    if connectivity_tri:
        ax.triplot(x_tri, y_tri, connectivity_tri)

    if patches:
        p = PatchCollection(patches)
        p.set_edgecolor('k')
        #p.set_facecolor(colors[lines_counter])
        if color_id is None:
            p.set_facecolor(colors[submesh_obj.key])
        else:    
            p.set_facecolor(colors[color_id])
        ax.add_collection(p)
        ax.autoscale()
        patches.clear()

    if points:
        ax.scatter(np.array(points).T[0], np.array(points).T[1], marker='o',label=str(submesh_obj.key))

    if plot_1d:
        lc = mc.LineCollection(lines, linewidths=2, color=colors[np.random.randint(0,9)])
        lc = mc.LineCollection(lines, linewidths=2, color=colors[lines_counter])
        ax.add_collection(lc)
        
        
        
    
    if submesh_obj.interface_nodes_dict:
        plot_nodes_in_the_interface(submesh_obj,ax)
                         
    return ax

def plot_submesh_obj(submesh_obj,ax=None, color_id=0):
    ''' This function plots 2D meshes, suport elements are 
    Triagules = Tri3, Tri6
    lines = straight_line, straight_line
    
    Arguments
    submesh_obj: submesh instance
    ax: matplotlib axes
    
    Return
    ax: matplotlib axes
    '''
           
    connectivity_tri = []
    lines = []
    patches = []
    x_tri = []
    y_tri = []
    lines_counter = 0
    p = None
    
    try:
        submesh_obj.parent_mesh.elements_type_dict = submesh_obj.parent_mesh.el_df['el_type'].to_dict()  
        elem_start_index = submesh_obj.parent_mesh.node_idx
        elem_last_index = len(submesh_obj.parent_mesh.el_df.columns)
        elements_dict = submesh_obj.parent_mesh.el_df.iloc[:,elem_start_index:elem_last_index]
        submesh_obj.parent_mesh.elements_dict = elements_dict.to_dict('index')
        d = submesh_obj.parent_mesh.elements_dict
        submesh_obj.elements_list = submesh_obj.elem_dataframe.index.tolist()
        
        
        for key, value in d.items():
            submesh_obj.parent_mesh.elements_dict[key] = [ int(j) for i, j in value.items() if not(np.isnan(j))]
        
        submesh_obj.parent_mesh.nodes_list = submesh_obj.parent_mesh.nodes.tolist()
        submesh_obj.parent_mesh.nodes_dict = {key:value for key,value in enumerate(submesh_obj.parent_mesh.nodes_list)}
        
    except:
        return None
    
    for elem in submesh_obj.elements_list:
        
        elem_connec = submesh_obj.parent_mesh.elements_dict[elem]
        if submesh_obj.parent_mesh.elements_type_dict[elem] == 2 or submesh_obj.parent_mesh.elements_type_dict[elem] == 'Tri3':
            connectivity_tri.append(elem_connec)
            
        elif submesh_obj.parent_mesh.elements_type_dict[elem] == 9 or submesh_obj.parent_mesh.elements_type_dict[elem] == 'Tri6':
            connectivity_tri.append(elem_connec[0:3])
            
        elif submesh_obj.parent_mesh.elements_type_dict[elem] == 3 or submesh_obj.parent_mesh.elements_type_dict[elem] == 'Quad4':
            #connectivity_quad.append(elem_connec) 
            elem_nodes = []
            for node in elem_connec:
                elem_nodes.append(submesh_obj.parent_mesh.nodes_dict[node][0:2])
            
            polygon = Polygon(elem_nodes, True)
            patches.append(polygon)
        
        elif submesh_obj.parent_mesh.elements_type_dict[elem] == 1 or submesh_obj.parent_mesh.elements_type_dict[elem] == 'straight_line':
           node1 = submesh_obj.parent_mesh.elements_dict[elem][0]
           node2 = submesh_obj.parent_mesh.elements_dict[elem][1]
           x = [submesh_obj.parent_mesh.nodes_dict[node1][0],submesh_obj.parent_mesh.nodes_dict[node1][1]]            
           y = [submesh_obj.parent_mesh.nodes_dict[node2][0],submesh_obj.parent_mesh.nodes_dict[node2][1]]
           lines.append([x,y])
           
        elif submesh_obj.parent_mesh.elements_type_dict[elem] == 8 or submesh_obj.parent_mesh.elements_type_dict[elem] == 'quadratic_line':
           node1 = submesh_obj.parent_mesh.elements_dict[elem][0]
           node2 = submesh_obj.parent_mesh.elements_dict[elem][1]
           x = [submesh_obj.parent_mesh.nodes_dict[node1][0],submesh_obj.parent_mesh.nodes_dict[node1][1]]            
           y = [submesh_obj.parent_mesh.nodes_dict[node2][0],submesh_obj.parent_mesh.nodes_dict[node2][1]]
           lines.append([x,y])
                
   
    for node in submesh_obj.parent_mesh.nodes_dict:
        x_tri.append(submesh_obj.parent_mesh.nodes_dict[node][0])
        y_tri.append(submesh_obj.parent_mesh.nodes_dict[node][1])
                
    if ax == None:
        fig = plt.figure()
        ax = plt.axes()    
    
    if connectivity_tri:
        ax.triplot(x_tri, y_tri, connectivity_tri)

    if patches:
        p = PatchCollection(patches)
        p.set_edgecolor('k')
        p.set_facecolor(colors[color_id])
        ax.add_collection(p)
        ax.autoscale()
        patches.clear()
        if lines_counter==max_number_of_color:
           lines_counter = 0
        else:
            lines_counter += 1
    
    # if lines:
        # #lc = mc.LineCollection(lines, linewidths=2, color=colors[np.random.randint(0,9)])
        # lc = mc.LineCollection(lines, linewidths=2, color=colors[lines_counter])
        # ax.add_collection(lc)
        # ax.autoscale()
        # ax.margins(0.1)
        # if lines_counter==max_number_of_color:
           # lines_counter = 0
        # else:
            # lines_counter += 1
    
    if submesh_obj.interface_nodes_dict:
        plot_nodes_in_the_interface(submesh_obj,ax)
         
    ax.legend([p, 'Domain'])         
    return ax, p
    
def plot_nodes_in_the_interface(submesh_obj,ax=None):
    ''' This function plots nodes at the interface 
    
    Arguments
    submesh_obj: submesh instance
    ax: matplotlib axes
    
    Return
    ax: matplotlib axes
    '''
        
    for partition_key in submesh_obj.interface_nodes_dict:
                
            nx = []
            ny = []
            
            interface_node = submesh_obj.interface_nodes_dict[partition_key]
            for node in interface_node :
                coord = submesh_obj.parent_mesh.nodes_dict[node]
                
                nx.append(coord[0])
                ny.append(coord[1])
            
            if ax == None:
                fig = plt.figure()
                ax = plt.axes()   
            
            
            plt.scatter(nx,ny)

def plot_deformed_subdomain(feti_obj,mult=1.0,ax=None):
    ''' This function plots deformed 2D meshes.
    Suport the same elements as plot_submesh method
    
    Arguments
        feti_obj: FETIsubdomain
        ax: matplotlib axes
        mult: float
            paramenter for scale the mesh plot
        
    Return
        ax: matplotlib axe 
    '''
    
    # create new node list
    new_coord_list = []
    for i,node_coord in enumerate(feti_obj.mesh.nodes):
        u = np.squeeze(np.asarray(feti_obj.displacement[feti_obj.id_matrix[i]]))
        new_coord = node_coord + mult*u
        new_coord_list.append(new_coord)
    
    
    sub_obj = copy.copy(feti_obj.submesh)
    

    nodes_dict = sub_obj.parent_mesh.nodes.copy()
    
    # create dict to local_to_global
    local_to_global_dict = {}
    for key in sub_obj.global_to_local_dict:
        local_to_global_dict[sub_obj.global_to_local_dict[key]] = key    

    for  i,new_coord in enumerate(new_coord_list):
        
        global_node_id = local_to_global_dict[i]
        sub_obj.parent_mesh.nodes[global_node_id] = new_coord
        
    ax = plot_submesh(sub_obj,ax)
    
    # back to original
    sub_obj.parent_mesh.nodes = nodes_dict
    
    return ax
        
def plot_domain(domain,ax=None):
    ''' This function plots subdomains 2D meshes.
    Suport the same elements as plot_submesh method
    
    Arguments
        domain: SubMesh intance
            domain must have groups dictionary with SubMesh instances of subdomain
            
        ax: matplotlib axes
        mult: float
            paramenter for scale the mesh plot
        
    Return
        ax: matplotlib axe 
    '''
    

    for i,j in enumerate(domain.groups):
        if ax == None:
            ax = plot_submesh(domain.groups[j], color_id = i )
        else:
            plot_submesh(domain.groups[j],ax, color_id = i)      
    
    return ax   

def plot_boundary_1d(mesh_obj,ax=None,linewidth=2):
    ''' This function plot the 1D boundary elements
    
    Argument:
        mesh_obj: amfe mesh instance
        ax: matplotlib Axes
    
    return 
        ax: matplotlib Axes

    '''           
        
    key_list = []
    mesh_obj.split_in_groups()
    lines_counter = 0
    color_list = []
    for sub_key,submesh_obj in mesh_obj.groups.items():
    
        connectivity_tri = []
        lines = []
        patches = []
        x_tri = []
        y_tri = []
        
        
        
        try:
            submesh_obj.parent_mesh.elements_type_dict = submesh_obj.parent_mesh.el_df['el_type'].to_dict()  
            elem_start_index = submesh_obj.parent_mesh.node_idx
            elem_last_index = len(submesh_obj.parent_mesh.el_df.columns)-1
            elements_dict = submesh_obj.parent_mesh.el_df.iloc[:,elem_start_index:elem_last_index]
            submesh_obj.parent_mesh.elements_dict = elements_dict.to_dict('index')
            d = submesh_obj.parent_mesh.elements_dict
            submesh_obj.elements_list = submesh_obj.elem_dataframe.index.tolist()
            
            
            for key, value in d.items():
                submesh_obj.parent_mesh.elements_dict[key] = [ int(j) for i, j in value.items() if not(np.isnan(j))]
            
            submesh_obj.parent_mesh.nodes_list = submesh_obj.parent_mesh.nodes.tolist()
            submesh_obj.parent_mesh.nodes_dict = {key:value for key,value in enumerate(submesh_obj.parent_mesh.nodes_list)}
            
        except:
            return None
        
        for elem in submesh_obj.elements_list:
                        
            if submesh_obj.parent_mesh.elements_type_dict[elem] == 3:
                #connectivity_quad.append(elem_connec) 
                elem_nodes = []
                for node in elem_connec:
                    elem_nodes.append(submesh_obj.parent_mesh.nodes_dict[node][0:2])
                
                polygon = Polygon(elem_nodes, True)
                patches.append(polygon)
            
            elif submesh_obj.parent_mesh.elements_type_dict[elem] == 1 or submesh_obj.parent_mesh.elements_type_dict[elem] == 'straight_line':
               node1 = submesh_obj.parent_mesh.elements_dict[elem][0]
               node2 = submesh_obj.parent_mesh.elements_dict[elem][1]
               x = [submesh_obj.parent_mesh.nodes_dict[node1][0],submesh_obj.parent_mesh.nodes_dict[node1][1]]            
               y = [submesh_obj.parent_mesh.nodes_dict[node2][0],submesh_obj.parent_mesh.nodes_dict[node2][1]]
               lines.append([x,y])
               
               
            elif submesh_obj.parent_mesh.elements_type_dict[elem] == 8 or submesh_obj.parent_mesh.elements_type_dict[elem] == 'quadratic_line':
               node1 = submesh_obj.parent_mesh.elements_dict[elem][0]
               node2 = submesh_obj.parent_mesh.elements_dict[elem][1]
               x = [submesh_obj.parent_mesh.nodes_dict[node1][0],submesh_obj.parent_mesh.nodes_dict[node1][1]]            
               y = [submesh_obj.parent_mesh.nodes_dict[node2][0],submesh_obj.parent_mesh.nodes_dict[node2][1]]
               lines.append([x,y])
                    
       
        for node in submesh_obj.parent_mesh.nodes_dict:
            x_tri.append(submesh_obj.parent_mesh.nodes_dict[node][0])
            y_tri.append(submesh_obj.parent_mesh.nodes_dict[node][1])
                    
        if ax == None:
            fig = plt.figure()
            ax = plt.axes()    
        
        if connectivity_tri:
            ax.triplot(x_tri, y_tri, connectivity_tri)
    
        if patches:
            p = PatchCollection(patches)
            p.set_edgecolor('k')
            p.set_facecolor(colors[np.random.randint(0,9)])
            ax.add_collection(p)
            ax.autoscale()
            patches.clear()
        
        if lines:
            #lc = mc.LineCollection(lines, linewidths=linewidth, color=colors[np.random.randint(0,9)],label=str(sub_key))
            lc = mc.LineCollection(lines, linewidths=linewidth, color=colors[lines_counter],label=str(sub_key))
            ax.add_collection(lc)
            ax.autoscale()
            ax.margins(0.1)
            if lines_counter==max_number_of_color:
                lines_counter = 0
            else:
                lines_counter += 1
            
        
        if submesh_obj.interface_nodes_dict:
            plot_nodes_in_the_interface(submesh_obj,ax)
        
                        
        ax.legend()
    return ax

def plot_subdomains(subdomains_dict,a=1.0,ax=None):
    ''' This function plots deformed 2D meshes.
    Suport the same elements as plot_submesh method
    
    Arguments
        subdomains_dict: dict
            dict with instances of FETIsubdomain
        ax: matplotlib axes
        a: float
            paramenter for scale the mesh plot
        
    Return
        ax: matplotlib axe 
    '''
    
    for i,j in enumerate(subdomains_dict):
        subi = subdomains_dict[j]
        strcommand = 'sub' + str(j) + ' =  subi'
        exec(strcommand)
        if i==0 and ax==None:
            ax = plot_deform_submesh(subi,mult=a)
        else:
            plot_deform_submesh(subi,ax,mult=a) 
    return ax
            
def plot_deformed_subdomains(subdomains_dict,a=1.0,ax=None):
    ''' This function plots deformed 2D meshes.
    Suport the same elements as plot_submesh method
    
    Arguments
        subdomains_dict: dict
            dict with instances of FETIsubdomain
        ax: matplotlib axes
        a: float
            paramenter for scale the mesh plot
        
    Return
        ax: matplotlib axe 
    '''
    
    for i,j in enumerate(subdomains_dict):
        subi = subdomains_dict[j]
        strcommand = 'sub' + str(j) + ' =  subi'
        exec(strcommand)
        if i==0 and ax==None:
            ax = plot_deform_submesh(subi,mult=a)
        else:
            plot_deform_submesh(subi,ax,mult=a) 
    return ax

def plotDeformMesh(connectivity, nodes, displacement, factor=1, ax = None):
    ''' This function plots Triagular 2D meshes.
    
    
    Arguments
        connectivity: list
            list of elements connectivity
        nodes: np.array
            array with nodes coordinates
        factor: float
            paramenter for scale the mesh plot
        ax: matplotlib axes
        
    Return
        ax: matplotlib axe 
    '''
    
    triangles = []
    x = []
    y = []
    
    for elem in connectivity:
        triangles.append([elem[0],elem[1],elem[2]])    
    
    jump = 2
    for i,node in enumerate(nodes):

        xc = node[0] + factor*displacement[0 + i*jump]
        yc = node[1] + factor*displacement[1 + i*jump]
        x.append(xc)
        y.append(yc)

    # Rather than create a Triangulation object, can simply pass x, y and triangles
    # arrays to triplot directly.  It would be better to use a Triangulation object
    # if the same triangulation was to be used more than once to save duplicated
    # calculations.
    
    if ax == None:
        fig = plt.figure()
        ax = plt.axes() 
        
    ax.autoscale()
    tri = ax.triplot(x, y, triangles)
    
    
    return tri, ax
            
def plotDeformTriMesh(connectivity, nodes, displacement, factor=1, ax = None):
    ''' This function plots Triagular 2D meshes.
    
    
    Arguments
        connectivity: list
            list of elements connectivity
        nodes: np.array
            array with nodes coordinates
        factor: float
            paramenter for scale the mesh plot
        ax: matplotlib axes
        
    Return
        ax: matplotlib axe 
    '''
    
    triangles = []
    x = []
    y = []
    
    for elem in connectivity:
        triangles.append([elem[0],elem[1],elem[2]])    
    
    jump = 2
    for i,node in enumerate(nodes):

        xc = node[0] + factor*displacement[0 + i*jump]
        yc = node[1] + factor*displacement[1 + i*jump]
        x.append(xc)
        y.append(yc)

    # Rather than create a Triangulation object, can simply pass x, y and triangles
    # arrays to triplot directly.  It would be better to use a Triangulation object
    # if the same triangulation was to be used more than once to save duplicated
    # calculations.
    
    if ax == None:
        fig = plt.figure()
        ax = plt.axes() 
        
    ax.autoscale()
    tri = ax.triplot(x, y, triangles)
    
    
    return tri, ax           
    
def plotDeformQuadMesh(connectivity, nodes, displacement, factor=1, ax = None, color_id=None):
    ''' This function plots Triagular 2D meshes.
    
    
    Arguments
        connectivity: list
            list of elements connectivity
        nodes: np.array
            array with nodes coordinates
        factor: float
            paramenter for scale the mesh plot
        ax: matplotlib axes
        
    Return
        p as PatchCollection
        ax: matplotlib axe 
    '''
    
    patches = []
    x = []
    y = []
    
    elem_dofs = 2
    elem_nodes = []
    for i,node in enumerate(nodes):
        xc = node[0] + factor*displacement[0 + i*elem_dofs]
        yc = node[1] + factor*displacement[1 + i*elem_dofs]
        x.append(xc)
        y.append(yc)

    for elem in connectivity:
        node_coord = []
        for node in elem:
            node_coord.append([x[node],y[node]])
        polygon = Polygon(node_coord, True)
        patches.append(polygon)

    if ax == None:
        fig = plt.figure()
        ax = plt.axes() 
        
    ax.autoscale()
    p = PatchCollection(patches)
    p.set_edgecolor('k')
    if color_id is None:
        p.set_facecolor(colors[np.random.randint(0,9)])
    else:
        p.set_facecolor(colors[color_id])
    ax.add_collection(p)
    ax.autoscale()
    patches.clear()
    
    return p, ax

def plot_deform_2D_cyclic_mesh(m, nsectors, u_dict, u_id, factor=1, ax = None, color_id=None):

    if ax == None:
        fig = plt.figure()
        ax = plt.axes() 

    connectivity = m.connectivity
    
    elem_nodes = connectivity[0].shape[0] 

    for i in range(nsectors):
        mi  = m.rot_z(i*(360/nsectors))    
        nodes = mi.nodes
        displacement = u_dict[i].T.real[u_id,:]
        if elem_nodes == 4:
            plotDeformQuadMesh(connectivity,nodes, displacement,factor=factor,ax=ax,color_id=color_id)
        elif elem_nodes == 3:
            plotDeformTriMesh( connectivity,nodes, displacement,factor=factor,ax=ax,color_id=color_id)
        else:
            raise('Connectivity node supported')
    
    return ax


def plot_deform_3Dcyclic_mesh(m, u_dict, u_id, nsectors, factor=1, ax = None, color_id=None):

    if ax==None:
            ax = a3.Axes3D(plt.figure()) 

    connectivity = m.connectivity
    
    elem_nodes = connectivity[0].shape[0] 

    for i in range(nsectors):
        mi  = m.rot_z(i*(360/nsectors))    
        nodes = mi.nodes
        displacement = u_dict[i].T.real[u_id,:]
        if elem_nodes == 4:
            plotDeformQuadMesh(connectivity,nodes, displacement,factor=factor,ax=ax,color_id=color_id)
        elif elem_nodes == 3:
            plotDeformTriMesh( connectivity,nodes, displacement,factor=factor,ax=ax,color_id=color_id)
        else:
            raise('Connectivity node supported')
    
    return ax

def plot2Dmesh(mesh_obj,ax=None, boundaries=True, counter=0):
    ''' This function plot mesh elements
    
    Argument:
        mesh_obj: amfe mesh instance
        ax: matplotlib Axes
    
    return 
        ax: matplotlib Axes

    '''       

    key_list = []
    mesh_obj.split_in_groups()
    
    for sub_key,submesh_obj in mesh_obj.groups.items():
        if ax == None:
            if isinstance(sub_key,int):
                ax = plot_submesh(submesh_obj,color_id = sub_key)
            else:
                ax = plot_submesh(submesh_obj,color_id=counter)
        else:
            if isinstance(sub_key,int):
                ax = plot_submesh(submesh_obj ,ax, color_id = sub_key)      
            else:
                ax = plot_submesh(submesh_obj ,ax, color_id=counter)      

        counter+=1
        #print(p)    
        #if p is not None:
        #    p.set_label(['domain %s' %str(sub_key)])
                
        
    if boundaries:
        plot_boundary_1d(mesh_obj,ax,linewidth=4)
    
    #leg = ax.get_legend()
    #leg.legendHandles.append(p)
    #print(leg.legendHandles)
    #leg.legendHandles[-1].set_facecolor('yellow')
    ax.legend()
    return ax  
    
def plot2Dcyclicmesh(mesh_obj, nsectors, ax=None, boundaries=True,**kwargs):
    
    if ax == None:
        fig = plt.figure()
        ax = plt.axes() 
    
    m = mesh_obj
    theta = 360/nsectors
    for i in range(nsectors):
        m1 = m.rot_z(i*theta) 
        ax = plot2Dmesh(m1,ax=ax)

    return ax

def plot3Dcyclicmesh(mesh_obj, nsectors, ax=None, **kwargs):
    
    if ax==None:
        ax = a3.Axes3D(plt.figure()) 

    m = mesh_obj
    theta = 360/nsectors
    for i in range(nsectors):
        m1 = m.rot_z(i*theta) 
        ax = plot3Dmesh(m1,ax=ax,**kwargs)

    return ax
    
def plot2Dnodes(mesh_obj,ax=None,plot_nodeid=False,fonte=12,color='red'):
    ''' This function plot the nodes of a mesh
    
    Argument:
        mesh_obj: amfe mesh instance
        ax: matplotlib Axes
    
    return 
        ax: matplotlib Axes

    '''     

    if ax == None:
        ax = plt.axes() 
    

    ax.plot(mesh_obj.nodes.T[0],mesh_obj.nodes.T[1],'o')
    
    if plot_nodeid:
        for i, coord in enumerate(mesh_obj.nodes):
            ax.text(coord[0],coord[1], str(i), color=color, fontsize=fonte)
            

    return ax  

def plot3Dnodes(mesh_obj,ax=None,plot_nodeid=False,fonte=12,color='red',marker='ko'):
    ''' This function plot the nodes of a mesh
    
    Argument:
        mesh_obj: amfe mesh instance
        ax: matplotlib Axes
    
    return 
        ax: matplotlib Axes

    '''     

    if ax == None:
        ax = a3.Axes3D(plt.figure()) 
    

    ax.plot(mesh_obj.nodes.T[0],mesh_obj.nodes.T[1],mesh_obj.nodes.T[2],marker)
    
    if plot_nodeid:
        for i, coord in enumerate(mesh_obj.nodes):
            ax.text(coord[0],coord[1],coord[2], str(i), color=color, fontsize=fonte)
            

    return ax  

def plot2Dnode_id(mesh_obj,node_id,ax=None,plot_nodeid=False,fonte=12,color='red'):
    ''' This function plot the nodes of a mesh
    
    Argument:
        mesh_obj: amfe mesh instance
        ax: matplotlib Axes
    
    return 
        ax: matplotlib Axes

    '''     

    if ax == None:
        ax = plt.axes() 
    
    coord = mesh_obj.nodes[node_id]

    ax.plot(coord[0],coord[1],'o')
    
    if plot_nodeid:
        ax.text(coord[0],coord[1], str(node_id), color=color, fontsize=fonte)
            

    return ax 

def plot3Dnode_id(mesh_obj,node_id,ax=None,plot_nodeid=False,fonte=12,color='red',marker='ko'):
    ''' This function plot the nodes of a mesh
    
    Argument:
        mesh_obj: amfe mesh instance
        ax: matplotlib Axes
    
    return 
        ax: matplotlib Axes

    '''     

    if ax == None:
        ax = a3.Axes3D(plt.figure()) 
    
    coord = mesh_obj.nodes[node_id]
    ax.scatter(coord[0],coord[1],coord[2],marker)
    
    if plot_nodeid:
        ax.text(coord[0],coord[1],coord[2], str(node_id), color=color, fontsize=fonte)
            

    return ax  

def plot_2D_system_solution(my_system, factor=1, ax = None, u_id = 1,facecolor=(1,1,1),highlight_nodes=[],linewidth=0.5):
    ''' This function plots Triagular 2D meshes.
    
    
    Arguments
        my_system : MechanicalSystem obj
            mechanical system obj with solution
        factor: float
            paramenter for scale the mesh plot
        ax: matplotlib axes

        u_id : int
            id of the displacment to be ploted
        
    Return
        p as PatchCollection
        ax: matplotlib axe 
    '''
    
    patches = []
    dof = []

    displacement = my_system.u_output[u_id].real
    node_list = my_system.mesh_class.nodes
    connectivity = my_system.mesh_class.connectivity
    #nodes = my_system.mesh_class.nodes[my_system.assembly_class.node_list]
    nodes = node_list
    highlight_nodes_coord = []
    elem_dofs = my_system.assembly_class.mesh.no_of_dofs_per_node
    elem_nodes = []
    for i,node in enumerate(nodes):
        xc = node[0] + factor*displacement[0 + i*elem_dofs]
        yc = node[1] + factor*displacement[1 + i*elem_dofs]
        dof.extend([xc,yc])
        
        if elem_dofs>2:
            zc = node[2] + factor*displacement[2 + i*elem_dofs]
            dof.extend([zc])
        
    dof = np.array(dof)    
    for elem in connectivity:
        node_coord = []
        for node in elem:
            dof_id = my_system.assembly_class.id_matrix[node]
            node_coord.append(dof[dof_id])
            if node in highlight_nodes:
                highlight_nodes_coord.append(dof[dof_id])
        polygon = Polygon(node_coord, True)
        patches.append(polygon)

    if ax == None:
        fig = plt.figure()
        ax = plt.axes() 
        
    if highlight_nodes:
        ax.plot(np.array(highlight_nodes_coord).T[0],np.array(highlight_nodes_coord).T[1],'yo')
    ax.autoscale()
    p = PatchCollection(patches)
    p.set_edgecolor('k')
    p.set_facecolor(facecolor)
    p.set_linewidth(linewidth)
    ax.add_collection(p)
    ax.autoscale()
    patches.clear()
    
    return p, ax
    
def plot_superdomain(superdomain_obj, factor = 1, ax = None):
    ''' plot superdomain results
    '''
    if ax == None:
        fig = plt.figure()
        ax = plt.axes() 
        
    connectivity = {}
    nodes = {}

    for sub_key in superdomain_obj.domains_key_list:
        sub = superdomain_obj.get_feti_subdomains(sub_key)
        connectivity[sub_key] = sub.mesh.connectivity
        nodes[sub_key] = sub.mesh.nodes
        quad, ax = plotDeformQuadMesh(connectivity[sub_key],nodes[sub_key],superdomain_obj._subdomain_displacement_dict[sub_key],factor,ax,color_id = sub_key ) 
          
    return ax

def plot3Dmesh(mesh_obj,ax=None, boundaries=True, alpha=0.2, color='grey', plot_nodes=True, scale = 1.0, Label = False,**kwargs):

    legend_handles = []
    if ax==None:
        ax = a3.Axes3D(plt.figure()) 
    
    mesh_obj.split_in_groups()
    nodes = mesh_obj.nodes*scale
    for key in mesh_obj.groups:
        submesh = mesh_obj.groups[key]
        elem_list_type = submesh.get_element_type_list()
        if len(elem_list_type)>1:
            submesh = copy.deepcopy(submesh)
        '''
        if len(elem_list_type)>1:
            print('SubMesh with more than one type of element. \n \
            This function do not support multiple type of elements. \
            moving to the next SubMesh')
            continue 
        
        elem_type = elem_list_type[0]
        '''
        for elem_type in elem_list_type:
            try:
                if len(elem_list_type)>1:
                    submesh_obj_single_elem = submesh.get_submesh('el_type',elem_type)
                    connect = submesh_obj_single_elem.get_submesh_connectivity()
                else:
                    connect = submesh.get_submesh_connectivity()
                    
                if elem_type in Tri_D_elem_list:
                    if elem_type=='Tet4':
                        connect = get_triangule_faces_from_tetrahedral(connect)
                        ax = plot_3D_polygon(nodes, connect, ax=ax, alpha=alpha, color=color, plot_nodes=plot_nodes,**kwargs)
                        legend_handles.append(mpatches.Patch(color=color, label=str(key)))

                    elif elem_type=='Tet10':
                        connect = get_triangule_faces_from_tetrahedral(np.array(connect).T[0:4].T)
                        ax = plot_3D_polygon(nodes, connect, ax=ax, alpha=alpha, color=color, plot_nodes=plot_nodes,**kwargs)
                        legend_handles.append(mpatches.Patch(color=color, label=str(key)))

                    elif elem_type=='Hexa20':
                        connect = get_quad_faces_from_hexa(np.array(connect).T[0:8].T)
                        ax = plot_3D_polygon(nodes, connect, ax=ax, alpha=alpha, color=color, plot_nodes=plot_nodes,**kwargs)
                        legend_handles.append(mpatches.Patch(color=color, label=str(key)))
                    
                    elif elem_type=='Hexa8':
                        connect = get_quad_faces_from_hexa(np.array(connect))
                        ax = plot_3D_polygon(nodes, connect, ax=ax, alpha=alpha, color=color, plot_nodes=plot_nodes,**kwargs)
                        legend_handles.append(mpatches.Patch(color=color, label=str(key)))
                    
                    else:
                        print('Warning! Type of element = %s not support by this method.' %elem_type)
            except:
                print('Element in mesh is not supported.')


        
    if  boundaries:
        for key in mesh_obj.groups:
            submesh = mesh_obj.groups[key]
            elem_list_type = submesh.get_element_type_list()
        
            if len(elem_list_type)>1:
                print('SubMesh with more than one type of element. \n \
                This function do not support multiple type of elements.')
            
            if not isinstance(elem_list_type[0],str):
                continue

            elem_type = elem_list_type[0]
            connect = mesh_obj.groups[key].get_submesh_connectivity()

            if elem_type in Two_D_elem_list:
                try:
                    color_bound = colors[submesh.key]
                except:
                    color_bound = colors[0]
                ax = plot_3D_polygon(nodes, connect, ax=ax, alpha=1, color=color_bound, plot_nodes=plot_nodes)
                legend_handles.append(mpatches.Patch(color=color_bound, label=str(key)))

    if not plot_nodes:
        x_max = max(nodes[:,0])
        x_min = min(nodes[:,0])
        y_max = max(nodes[:,1])
        y_min = min(nodes[:,1])
        z_max = max(nodes[:,2])
        z_min = min(nodes[:,2])

        i_max = max([x_max,y_max,z_max])
        i_min = min([x_min,y_min,z_min])
        max_coord = max([abs(i_max),abs(i_min)])
        #points = np.array([[x_min,y_min,z_min],
        #                    [x_max,y_max,z_max]])

        #points = np.array([[i_min,i_min,i_min],
        #                    [i_max,i_max,i_max]])
        #ax.plot(points[:,0], points[:,1], points[:,2], 'wo')
        ax.set_xlim((-max_coord,max_coord))
        ax.set_ylim((-max_coord,max_coord))
        ax.set_zlim((-max_coord,max_coord))

    if Label:
        ax.legend(handles= legend_handles,fontsize=30)
    return ax
        
def plot3D_submesh(submesh,ax=None, alpha=0.2, color='grey', plot_nodes=True, interface_nodes=True, scale = 1.0,linewidth=None):
    ''' This function add a plot Submesh to a ax

    argument 
        
    submesh : SubMesh obj
        SubMesh obj to be plotted
    
    ax : Axes3D
            matplotlib Axes3D object to plot the polygon
    
     
    alpha : float
        float whcih controls the Polygon transparence
        
    color : str or tuple
        color of the polygon, check matplotlib document 
        to see the supported color names
        
    plot_nodes : Boolen
        Boolen to plot node in the SubMesh

    '''
    elem_list_type = submesh.get_element_type_list()
    
    if ax==None:
        ax = a3.Axes3D(plt.figure()) 
        
    if len(elem_list_type)>1:
        raise('SubMesh with more than one type of element. \n \
        This function do not support multiple type of elements.')

    mesh_obj = submesh.parent_mesh    
    nodes = copy.deepcopy(mesh_obj.nodes)*scale
    elem_type = elem_list_type[0]
    connect = submesh.get_submesh_connectivity()
    if elem_type in Tri_D_elem_list:
        if elem_type=='Tet4':
            connect = get_triangule_faces_from_tetrahedral(connect)
            ax = plot_3D_polygon(nodes, connect, ax=ax, alpha=alpha, color=color, plot_nodes=plot_nodes,linewidth=linewidth)
        
        elif elem_type=='Hexa20':
            connect = get_quad_faces_from_hexa(np.array(connect).T[0:8].T)
            ax = plot_3D_polygon(nodes, connect, ax=ax, alpha=alpha, color=color, plot_nodes=plot_nodes, linewidth=linewidth)
        
        elif elem_type=='Hexa8':
                    connect = get_quad_faces_from_hexa(np.array(connect))
                    ax = plot_3D_polygon(nodes, connect, ax=ax, alpha=alpha, color=color, plot_nodes=plot_nodes, linewidth=linewidth)
        else:
            raise('Type of element = %s not support by this method.' %elem_type)
    elif elem_type in Two_D_elem_list:
        ax = plot_3D_polygon(nodes, connect, ax=ax, alpha=alpha, color=color, plot_nodes=plot_nodes, linewidth=linewidth)
    else:
        print('Type of element = %s not support by this method.' %elem_type)
        
        
    if interface_nodes:
        for nei_id in submesh.interface_nodes_dict:
            interface_nodes = submesh.interface_nodes_dict[nei_id]
            ax = plot_3D_interface_nodes(interface_nodes, nodes, color=colors[nei_id], ax=ax)
            
    if not plot_nodes:
        x_max = max(nodes[:,0])
        x_min = min(nodes[:,0])
        y_max = max(nodes[:,1])
        y_min = min(nodes[:,1])
        z_max = max(nodes[:,2])
        z_min = min(nodes[:,2])

        i_max = max([x_max,y_max,z_max])
        i_min = min([x_min,y_min,z_min])

        #points = np.array([[x_min,y_min,z_min],
        #                    [x_max,y_max,z_max]])

        points = np.array([[i_min,i_min,i_min],
                            [i_max,i_max,i_max]])
        ax.plot(points[:,0], points[:,1], points[:,2], 'wo')            
            
    return ax

def plot_3D_polygon(points_coord, vertice_matrix, ax=None, alpha=0.2, color='grey', plot_nodes=True,linewidth=None):
    ''' This function plots 3D polygonas based on points coordinates and
    matrix with the vertices of the polygons

    argument
        points_coord : np.array
            array with the point coordinates
        
        vertice_matrix : np.array
            matrix representing each polygon, where the number of pointer
            to the points_coord array with np.int
       
        ax : Axes3D
            matplotlib Axes3D object to plot the polygon
        
        alpha : float
            float whcih controls the Polygon transparence
        
        color : str or tuple
            color of the polygon, check matplotlib document 
            to see the supported color names

    return
        ax : Axes3D
            matplotlib Axes3D wich polygon object
    '''

    if ax==None:
       ax = a3.Axes3D(plt.figure()) 


    vts = points_coord[vertice_matrix, :]
    pol = a3.art3d.Poly3DCollection(vts)
    pol.set_alpha(alpha)
    pol.set_color(color)
    pol.set_edgecolor((0,0,0))
    pol.set_linewidth(linewidth)
    ax.add_collection3d(pol)
    if plot_nodes:
        nodes_in_elem = set(np.array(vertice_matrix).reshape(-1))
        points = points_coord[np.ix_(list(nodes_in_elem),[0,1,2])]
        ax.plot(points[:,0], points[:,1], points[:,2], 'ko', markersize=1)

    return ax

def get_triangule_faces_from_tetrahedral(tetrahedron_list,surface_only=True):
    ''' Create a list of 2D faces based on the index of 
    of a 3D tetrahedron

    arguments
        tetrahedron_list : list
            list with the index of a tetraedron [i, j, k, l]
        
        surface_only : Boolean
            only keep the Triangules in the Surface of the list of tetraedrons
    return 
        tri_list : list
            list with the index [i, j, k] of a the unique set triagules faces 
            in the tetrahefron list

    '''

    tri =[]
    for Tet in tetrahedron_list:
        for i,j,k in itertools.combinations([0,1,2,3], 3):
            #tri.append([Tet[i],Tet[j],Tet[k]])
            temp_list = [Tet[i],Tet[j],Tet[k]] 
            temp_list.sort()
            tri.append(temp_list)
    
    #print('All triangles %i' %len(tri))
    
    unique_tri_list = list(np.unique(tri,axis=0))
    #print('Only unique triangles %i' %len(unique_tri_list))
    
    if surface_only:
        surface_tri = []
        for unique_tri in unique_tri_list:
            num_of_occurances = tri.count(unique_tri.tolist())
            if num_of_occurances==1:
                surface_tri.append(unique_tri)
        tri_list = surface_tri
        #print('Surface triangles %i' %len(tri_list))

    else:
        tri_list = unique_tri_list

    return tri_list

def get_quad_faces_from_hexa(hexa_list,surface_only=True):
    ''' Create a list of 2D faces based on the index of 
    of a 3D Hexa

    arguments
        hexa_list : list
            list with the index of a tetraedron [i, j, k, l, h, g]
        
        surface_only : Boolean
            only keep the Triangules in the Surface of the list of tetraedrons
    return 
        quad_list : list
            list with the index [i, j, k, h] of a the unique set triagules faces 
            in the tetrahefron list

    '''

    quad =[]
    quad_ordered = []

    #All_square = itertools.combinations([0,1,2,3,4,5,6,7], 4)
    All_square = [[0,1,2,3],
                  [0,1,3,4],
                  [1,2,6,5],
                  [4,5,6,7],
                  [2,6,7,3],
                  [0,3,7,4]]

    for Hex in hexa_list:
        for i,j,k,h in All_square:
            quad_face = [Hex[i],Hex[j],Hex[k],Hex[h]]
            quad.append(quad_face)
            quad_ordered.append(list(np.sort(quad_face)))
    
    quad_list = []
    if surface_only:
        surface_quad = []
        for j,unique_quad in  enumerate(quad_ordered):
            if quad_ordered.count(unique_quad) == 1:
                quad_list.append(quad[j])
            else:
                continue
    else:
        quad_list = quad 

    return quad_list

def plot_3D_interface_nodes(nodes_list,node_coord, color=(0,0,0), ax=None):
    ''' This function plots the nodes
    given a list of nodes, the nodes coordinates and 
    the color to be plotted
    
    nodes_list : list
        list with nodes to be plotted
    node_coord : np.array
        array with node coordinates
    color : str or tuple
        color of the polygon, check matplotlib document 
        to see the supported color names
    ax : Axes3D
            matplotlib Axes3D object to plot the polygon
    '''
    
    if ax==None:
        ax = a3.Axes3D(plt.figure()) 
        
    points = node_coord[nodes_list]
    ax.plot(points[:,0], points[:,1], points[:,2], 'o', color=color)
    return ax
    
def plot_3D_displacement(system_obj,factor=1.0, scale=1.0,ax=None, displacement_id = 1, **kwargs):
    ''' This function plots the displacement of a system_obj
    
    arguments
        system_obj :  MechanicalSystem obj 
            MechanicalSystem obj with u_output 
        
        factor : float
            scale factor of the displacement
        scale : float
            scale for the mesh plot
        ax : Axes3D
            matplotlib Axes3D object to plot the polygon
        
        displacement_id : int
            id of the displacement to be plotted u_output[displacement_id]
    
    returns
        ax : Axes3D
            matplotlib Axes3D object with displacement
    
    '''
    
    if ax==None:
        ax = a3.Axes3D(plt.figure())
    
    # converting 1D displacement array to array compatible with id_matrix
    new_node_position = []
    nodes_coord = system_obj.mesh_class.nodes
    displacement = system_obj.u_output[displacement_id] 
    local_mesh_class = copy.deepcopy(system_obj.mesh_class)
    for node_id,dof_list in system_obj.assembly_class.id_matrix.items():
        old_node_coord = nodes_coord[node_id]
        delta_coord = factor*displacement[dof_list]
        #new_node_coord = old_node_coord + delta_coord
        #new_node_position.append(new_node_coord)
        local_mesh_class.nodes[node_id] = old_node_coord + delta_coord
    
    #local_mesh_class.nodes = np.array(new_node_position)

    ax = plot3Dmesh(local_mesh_class,ax=ax, boundaries=False, scale=scale, **kwargs)
    return ax

def plotmesh(mesh_obj,ax=None, boundaries=True, alpha=0.2, color='grey', plot_nodes=True, scale = 1.0, Label = False):
    
    dimension = mesh_obj.no_of_dofs_per_node
    if dimension==3:
        ax1 = plot3Dmesh(mesh_obj,ax=ax, boundaries=boundaries, alpha=alpha, color=color, plot_nodes=plot_nodes, scale = scale, Label = Label)
    elif dimension==2:
        ax1 = plot2Dmesh(mesh_obj,ax=ax, boundaries=boundaries)
    else:
        raise('Dimension is not supported')
    return ax1
   

def plot_system_solution(my_system, factor=1, ax = None, u_id = 1,facecolor=(1,1,1),boundaries=False, alpha=1, color='grey', plot_nodes=False, scale = 1.0, Label = False,collections = [],linewidth=0.5):   
    
    mesh_obj = my_system.mesh_class
    dimension = mesh_obj.no_of_dofs_per_node
    
    if dimension==3:
        displacement = my_system.u_output
        pltmesh = Plot3DMesh(mesh_obj,scale=scale, displacement_list = displacement, ax = ax, alpha=alpha)
        pltmesh.show(factor=factor,plot_nodes=False,displacement_id=u_id, collections=[])
        ax1 = pltmesh.ax
    elif dimension==2:
        ax1 = plot_2D_system_solution(my_system, factor=factor, ax =ax, u_id = u_id, facecolor=facecolor,linewidth=linewidth )
    else:
        raise('Dimension is not supported')
    return ax1
    

class Plot3DMesh():
    def __init__(self,mesh_obj,ax=None, boundaries=True, displacement_list = None,alpha=0.2, color='grey', 
                 plot_nodes=True, scale = 1.0, factor =1.0, Label = False, displacement_id=1, edgecolor=(0,0,0), linewidth=0.1, **kwargs):
        
        self.mesh_obj = copy.deepcopy(mesh_obj)
        self.updated_mesh_obj = copy.deepcopy(mesh_obj)
      
        if ax is None:
            self.ax = a3.Axes3D(plt.figure()) 
        else:
            self.ax = ax
        
        self.boundaries = boundaries
        self.alpha =alpha
        self.color = color
        self.plot_nodes=plot_nodes
        self.scale = scale
        self.Label = Label
        self.displacement_list = displacement_list
        self.displacement_id = displacement_id
        self.factor = factor
        self.last_scale = None
        self.edgecolor = edgecolor
        self.linewidth = linewidth
        self.pre_proc()
        
        
    def pre_proc(self):
        
        self.elem_type_connec = {}
        self.elem_type_polygons = {}
        mesh_obj = self.mesh_obj
        nodes = mesh_obj.nodes*self.scale
        mesh_obj.split_in_groups()
        for key in mesh_obj.groups:
            submesh = mesh_obj.groups[key]
            elem_list_type = submesh.get_element_type_list()
            
            if len(elem_list_type)>1:
                print('SubMesh with more than one type of element. \n \
                This function do not support multiple type of elements. \
                moving to the next SubMesh')
                continue 
            
            elem_type = elem_list_type[0]
            try:
                connect = copy.deepcopy(mesh_obj.groups[key].get_submesh_connectivity())
                if elem_type in Tri_D_elem_list:
                    if elem_type=='Tet4':
                        connect = get_triangule_faces_from_tetrahedral(connect)
                        
                    elif elem_type=='Tet10':
                        connect = get_triangule_faces_from_tetrahedral(np.array(connect).T[0:4].T)
                        
                    elif elem_type=='Hexa20':
                        connect = get_quad_faces_from_hexa(np.array(connect).T[0:8].T)
                        
                    elif elem_type=='Hexa8':
                        connect = get_quad_faces_from_hexa(np.array(connect))
                        
                    else:
                        print('Type of element = %s not support by this method.' %elem_type)
                        continue
                        
                    self.elem_type_polygons[elem_type] = self.create_3D_polygons(nodes, connect)
                    self.elem_type_connec[elem_type] = connect
            except:
                print('Element in mesh is not supported.')
        
    def show(self,factor=1.0,displacement_id=1,scale=None,plot_nodes=None, collections=[],ax=None):

        if ax is not None:
            self.ax = ax
        
        if scale is not None:
            self.scale = scale

        if plot_nodes is None:
            plot_nodes = self.plot_nodes
            
        if self.last_scale is not None:
            if self.scale!=self.last_scale: 
                self.mesh_obj.nodes = self.mesh_obj.nodes*self.scale
                self.last_scale = self.scale
        else:       
            self.mesh_obj.nodes = self.mesh_obj.nodes*self.scale
            self.last_scale = self.scale

        
        try:
            elem_type_polygons = self.update_nodes(factor=factor, displacement = self.displacement_list[displacement_id])
        except:
            elem_type_polygons = self.update_nodes()

        
        points_coord = self.mesh_obj.nodes
        
        
                
        #restart collection
        self.ax.collections = collections
        for elem_type, pols in  self.elem_type_polygons.items():
            self.ax.add_collection3d(pols)
            
            if plot_nodes:
                vertice_matrix = self.elem_type_connec[elem_type]
                nodes_in_elem = set(np.array(vertice_matrix).reshape(-1))
                points = points_coord[np.ix_(list(nodes_in_elem),[0,1,2])]
                self.ax.plot(points[:,0], points[:,1], points[:,2], 'ko', markersize=1)
        
        return self.ax
    
    def set_displacement(self,displacement_list):
        ''' This function sets the self.displacement variable
        in order to plot deformed mesh
        '''
        self.displacement_list = displacement_list
        
    def update_displacement(self,displacement,factor=1.0):

        nodes = np.copy(self.mesh_obj.nodes)
        ndof = len(displacement)
        displacemet_3_ny_m = displacement.reshape((int(ndof/3),3))
        nodes += factor*displacemet_3_ny_m.real
        return nodes

    def update_nodes(self,factor=1.0,displacement=None):

        for elem_type, connect in  self.elem_type_connec.items():
            if displacement is not None:
                nodes = self.update_displacement(displacement,factor)
            else:
                nodes = self.mesh_obj.nodes
            self.elem_type_polygons[elem_type] = self.create_3D_polygons(nodes, connect)
            self.updated_mesh_obj.nodes = nodes
        return self.elem_type_polygons[elem_type]
        
    def create_3D_polygons(self,points_coord, vertice_matrix, edgecolor = None):
        ''' This function plots 3D polygonas based on points coordinates and
        matrix with the vertices of the polygons

        argument
            points_coord : np.array
                array with the point coordinates
            
            vertice_matrix : np.array
                matrix representing each polygon, where the number of pointer
                to the points_coord array with np.int
           
            ax : Axes3D
                matplotlib Axes3D object to plot the polygon
            
            alpha : float
                float whcih controls the Polygon transparence
            
            color : str or tuple
                color of the polygon, check matplotlib document 
                to see the supported color names

        return
            ax : Axes3D
                matplotlib Axes3D wich polygon object
        '''

        if edgecolor is None:
            edgecolor = self.edgecolor
            
        alpha = self.alpha
        color = self.color
        vts = points_coord[vertice_matrix, :]
        pol = a3.art3d.Poly3DCollection(vts)
        pol.set_alpha(alpha)
        pol.set_color(color)
        pol.set_edgecolor(self.edgecolor)
        pol.set_linewidth(self.linewidth)
        
        self.polygons = pol
        return self.polygons
        
    def set_equal_axis_lim(self,limit_tuple):
        ''' set equal limits for all 3 axes
        '''
        self.ax.set_xlim(limit_tuple)
        self.ax.set_ylim(limit_tuple)
        self.ax.set_zlim(limit_tuple)
        

def plot_cyclic_mesh(m,nsectors,ax=None,bc=None,**kwargs):
    dim = m.no_of_dofs_per_node
    if dim ==2:
        if ax == None:
            fig = plt.figure()
            ax = plt.axes() 

        plot2Dcyclicmesh(m,nsectors,ax=ax)
        if bc is not None:
            ax.set_xlim(bc)
            ax.set_ylim(bc)
        
    elif dim ==3:
        if ax==None:
            ax = a3.Axes3D(plt.figure()) 
        
        plot3Dcyclicmesh(m,nsectors,ax=ax,boundaries=False,plot_nodes=False,alpha=1.0)
        if bc is not None:
            ax.set_xlim(bc)
            ax.set_ylim(bc)
            ax.set_zlim(bc)

       
    else:
        raise('mesh object is not supported.')

    return ax


def plot_deform_3D_mesh(mesh_obj, displacement, factor=1.0, ax=None, boundaries=False, alpha=0.2, color='grey', plot_nodes=True, scale = 1.0, Label = False, **kwargs):
    
    if ax is None:
        ax = a3.Axes3D(plt.figure()) 
    
    m = copy.deepcopy(mesh_obj)
    dim = m.no_of_dofs_per_node
    if displacement is not None:
        new_coord = m.nodes + factor*displacement.reshape(m.nodes.shape)
        m.nodes = new_coord

    plot3Dmesh(m, ax=ax, boundaries=boundaries, alpha=alpha, color=color, plot_nodes=plot_nodes, scale = scale, Label = Label, **kwargs)

    return ax




def plot_deform_3D_cyclic_mesh(m, nsectors, u_dict, u_id=0, factor=1, ax = None, **kwargs):

    if ax is None:
        ax = a3.Axes3D(plt.figure()) 
        

    for i in range(nsectors):
        displacement = u_dict[i].T.real[u_id]
        mi  = m.rot_z(i*(360/nsectors))    
        plot_deform_3D_mesh(mi,displacement, factor=factor, ax=ax,**kwargs)
        

    return ax


def plot_force(force,coord,ax=None,dim=3,factor=1.0,**kwargs):

    v = force.reshape(int(force.shape[0]/dim),dim)

    if dim==2:
        if ax is None:
            fig = plt.figure()
            ax = plt.axes() 
        ax = plot_2d_force(v,coord,ax,factor,**kwargs)

    elif dim==3:
        if ax is None:
            ax = a3.Axes3D(plt.figure()) 
        plot_3d_force(v,coord,ax,factor,**kwargs)
    else:
        raise ValueError('Dimension = %i is not supported' %i)

    return ax

def plot_2d_force(v,coord,ax,factor,**kwargs):

    #ax.plot(coord,'o')
    for ci, vi in zip(coord,v): 
        ax.arrow(ci[0],ci[1],factor*vi[0],factor*vi[1],**kwargs)

    return ax

def plot_3d_force(v,coord,ax,factor,**kwargs):

    #ax.plot(coord,'o')
    for ci, vi in zip(coord,v): 
        ax.quiver(ci[0],ci[1],ci[2],factor*vi[0],factor*vi[1],factor*vi[2],**kwargs)

    return ax


def plot_deform_cyclic_mesh(m, nsectors,u_dict, u_id, factor=1, ax = None, bc=None,color_id=None,**kwargs):

    dim = m.no_of_dofs_per_node
    if dim==2:
        ax = plot_deform_2D_cyclic_mesh(m, nsectors, u_dict, u_id, factor=factor, ax = ax, color_id=color_id)
        if bc is not None:
            ax.set_xlim(bc)
            ax.set_ylim(bc)
        return 

    elif dim ==3:
        ax = plot_deform_3D_cyclic_mesh(m, nsectors, u_dict, u_id=u_id, factor=factor, ax = ax, **kwargs)
        if bc is not None:
            ax.set_xlim(bc)
            ax.set_ylim(bc)
            ax.set_zlim(bc)

    return ax
#----------------------------------------------------------------------
# This Function will be deprecated in the Future
#----------------------------------------------------------------------

# aliasing functions
plot_mesh = plot2Dmesh
