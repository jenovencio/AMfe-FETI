# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 14:59:24 2017

@author: ge72tih
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.tri as tri

from matplotlib import collections  as mc
from matplotlib import colors, transforms
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import numpy as np

import copy

colors =['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']

def plot_submesh(submesh_obj,ax=None):
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
        p.set_facecolor(colors[np.random.randint(0,9)])
        ax.add_collection(p)
        ax.autoscale()
        patches.clear()
    
    if lines:
        lc = mc.LineCollection(lines, linewidths=2, color=colors[np.random.randint(0,9)])
        ax.add_collection(lc)
        ax.autoscale()
        ax.margins(0.1)
    
    if submesh_obj.interface_nodes_dict:
        plot_nodes_in_the_interface(submesh_obj,ax)
        
                        
    return ax

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


            
def plot_deformed_subdomain(feti_obj,ax=None,mult=1.0):
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
    

    for j in domain.groups:
        if ax == None:
            ax = plot_submesh(domain.groups[j])
        else:
            plot_submesh(domain.groups[j],ax)      
    
    return ax   



def plot_boundary_1d(mesh_obj,ax=None):
    ''' This function plot the 1D boundary elements
    
    Argument:
        mesh_obj: amfe mesh instance
        ax: matplotlib Axes
    
    return 
        ax: matplotlib Axes

    '''           
        
    key_list = []
    
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
            lc = mc.LineCollection(lines, linewidths=2, color=colors[np.random.randint(0,9)],label=str(sub_key))
            ax.add_collection(lc)
            ax.autoscale()
            ax.margins(0.1)
            
        
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
        mult: float
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
            
            
            