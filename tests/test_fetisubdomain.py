# importing lib and setting a list of meshes to be tested
import sys 
import amfe
import numpy as np
import scipy
import copy
import unittest



def eval_amfe_case(mesh_obj,material_obj,value = 1e8):
    ''' This function evaluates a AMfe standard case
    given a mesh_obj, a material_onj and a value for Neumann 
    B.C.

    arguments 
        mesh_obj: AMfe mesh object
        material_obj : AMfe material object
        value : float
            value for Neumann B.C.
    
    return 
        u_amfe : np.array
            displcament calculated by AMfe standard solver
    '''
    my_comp = amfe.MechanicalSystem()
    m1 = copy.deepcopy(mesh_obj)
    my_comp.set_mesh_obj(m1)
    my_comp.set_domain(6,material_obj)
    my_comp.apply_dirichlet_boundaries(1, 'xy')
    my_comp.apply_neumann_boundaries(3, value, 'normal')
    s = amfe.LinearStaticsSolver(my_comp)
    s.solve()

    u_amfe = my_comp.u_output[1]
    return u_amfe, my_comp



def eval_fetidomain_with_single_domain(mesh_obj,material_obj,splitting_tag ='phys_group', solver_type ='cholsps',value = 1e8):
    my_comp = amfe.MechanicalSystem()
    m2 = copy.deepcopy(mesh_obj)
    my_comp.set_mesh_obj(m2)
    my_comp.set_domain(6,material_obj)
    my_comp.apply_dirichlet_boundaries(1, 'xy')
    my_comp.apply_neumann_boundaries(3, value, 'normal')
    domain = my_comp.domain
    
    
    master, feti_domain_dict = set_master_and_fetisubdomain(domain, splitting_tag, solver_type)
    return master, feti_domain_dict


def set_master_and_fetisubdomain(domain, splitting_tag, solver_type):
    ''' This function set the master object to 
    connect subdomain and also creates FETIsubdomain
    from submeshes

    arguments
        domain : AMfe SubMesh obj
            SubMesh object with a variable 
        splitting_tag : str
            string with the tag for splitting domain
        solver_type : str
            string with the solver type for the computation of the pseudoinverse

    return:
        master : AMfe master object
            master object with link between subdomains

        feti_domain_dict : dict
            dictionary with fetisubdomains object
    '''
    domain.split_in_partitions(splitting_tag)
    problem_type = domain.problem_type
    num_partitions = len(domain.groups)
    partitions_list = np.sort(list(domain.groups.keys()),axis=-1)
    
    master = amfe.Master()
    master.subdomain_keys = partitions_list
    feti_domain_dict = {}
    for sub_id in partitions_list:
        sub_i =  amfe.FETIsubdomain(domain.groups[sub_id])
        sub_i.assemble_K_and_total_force()
        sub_i.set_solver_option(solver_type)
        #sub_i.set_pinv_tolerance = 1.0e-8
        feti_domain_dict[sub_id] = sub_i
        G_dict = sub_i.calc_G_dict()
        subdomain_interface_dofs_dict = sub_i.num_of_interface_dof_dict
        subdomain_null_space_size = sub_i.null_space_size
        master.append_G_dict(G_dict)
        master_dict = master.append_partition_dof_info_dicts(sub_id,subdomain_interface_dofs_dict,subdomain_null_space_size)
        sub_i.calc_dual_force_dict()
        master.append_null_space_force(sub_i.null_space_force,sub_id)

    return master, feti_domain_dict       



def eval_superdomain(mesh_obj,material_obj,splitting_tag ='phys_group', solver_type ='cholsps',value = 1e8):
    my_comp = amfe.MechanicalSystem()
    m2 = copy.deepcopy(mesh_obj)
    my_comp.set_mesh_obj(m2)
    my_comp.set_domain(6,material_obj)
    my_comp.apply_dirichlet_boundaries(1, 'xy')
    my_comp.apply_neumann_boundaries(3, value, 'normal')
    domain = my_comp.domain
    domain.split_in_partitions(splitting_tag)
    super_domain = amfe.SuperDomain(domain.groups,method=solver_type)
    global_lambda, global_alpha = super_domain.solve_dual_interface()

    super_domain.eval_subdomain_displacement(global_lambda,global_alpha)
    return super_domain.master, super_domain.feti_subdomains_dict


class test_fetisubdomain(unittest.TestCase):
    
    tolerance = 1.0e-8
    num_of_hashes = 50

    def compute_fetidomain_with_single_domain(self, solver_type ='cholsps'):
        ''' This test evaluate the displacement of a domain 
        with a single subdomain and compare to the displacement of
        AMfe case

        '''


        msh_dict = {}
        msh_dict[0] = amfe.amfe_dir('meshes/test_meshes/1_rectangle_8_elem.msh')
        msh_dict[1] = amfe.amfe_dir('meshes/test_meshes/retangule_5_by_2_quad_par_1.msh')
        msh_dict[2] = amfe.amfe_dir('meshes/test_meshes/retangule_5_by_2_ref_quad_par_1.msh')
        msh_dict[3] = amfe.amfe_dir('meshes/test_meshes/retangule_5_by_2_ref_tri_par_1.msh')
        msh_dict[4] = amfe.amfe_dir('meshes/test_meshes/retangule_5_by_2_tri_par_1.msh')
        msh_dict[5] = amfe.amfe_dir('meshes/test_meshes/retangule_square_hole_5_by_2_ref_tri_par_1.msh')
        msh_dict[6] = amfe.amfe_dir('meshes/test_meshes/retangule_square_hole_5_by_2_quad_par_1.msh')

        print_columns = test_fetisubdomain.num_of_hashes
        solver_tolerance = 1.0E-8
        hash_size = int((print_columns - len(solver_type))/2)
        print('#'*print_columns)
        print('#'*hash_size + solver_type.upper() + '#'*hash_size)
        print('#'*print_columns )

        for id in range(7):
            print('#'*hash_size + 'Case ' + str(id) + '#'*hash_size)
            mshfile = msh_dict[id]
            m1 = amfe.Mesh()
            m1.import_msh(mshfile)

            #creating material
            my_material = amfe.KirchhoffMaterial(E=210E9, nu=0.3, rho=7.86E3, plane_stress=True, thickness=0.1)


            u_amfe, my_comp = eval_amfe_case(m1, my_material)

            master, feti_domain_dict = eval_fetidomain_with_single_domain(m1, my_material, solver_type=solver_type)
            self.assertEqual(len(feti_domain_dict),1)
            sub_key = list(feti_domain_dict.keys())[0]

            feti_domain_dict[sub_key].solve_local_displacement()
            u_feti = amfe.FetiSolver.average_displacement_calc(my_comp,feti_domain_dict)

            ratio = np.linalg.norm(u_amfe - u_feti)/np.linalg.norm(u_amfe)
            print('\nRelative error between AMfe solver and local FETIsolver is : %e' %ratio)
            self.assertLessEqual(ratio, test_fetisubdomain.tolerance)
    
            print('#'*print_columns)

    def test_fetidomain_with_single_domain_cholsps(self):
        solver_type = 'cholsps'
        self.compute_fetidomain_with_single_domain(solver_type)
     
    def test_fetidomain_with_single_domain_splusps(self):
        solver_type = 'splusps'
        self.compute_fetidomain_with_single_domain(solver_type)
    
    def test_fetidomain_with_single_domain_svd(self):
        solver_type = 'svd'
        self.compute_fetidomain_with_single_domain(solver_type)
    
    def compute_fetidomain_with_two_domains(self, solver_type ='cholsps'):
        msh_dict = {}
        msh_dict[0] = amfe.amfe_dir('meshes/test_meshes/retangule_5_by_2_quad_par_2_irreg.msh')
        msh_dict[1] = amfe.amfe_dir('meshes/test_meshes/retangule_5_by_2_ref_quad_par_2.msh')
        msh_dict[2] = amfe.amfe_dir('meshes/test_meshes/retangule_5_by_2_ref_tri_par_2.msh')
        msh_dict[3] = amfe.amfe_dir('meshes/test_meshes/retangule_5_by_2_tri_par_2.msh')
        msh_dict[4] = amfe.amfe_dir('meshes/test_meshes/retangule_square_hole_5_by_2_quad_par_2.msh')
        msh_dict[5] = amfe.amfe_dir('meshes/test_meshes/retangule_square_hole_5_by_2_tri_par_2.msh')
        msh_dict[6] = amfe.amfe_dir('retangule_square_hole_5_by_2_tri_par_8.msh')

        id = 0

        print_columns = test_fetisubdomain.num_of_hashes
        hash_size = int((print_columns - len(solver_type))/2)
        print('#'*print_columns)
        print('#'*hash_size + solver_type.upper() + '#'*hash_size)
        print('#'*print_columns )

        mshfile = msh_dict[id]

        # instanciating mesh class
        m1 = amfe.Mesh()
        m1.import_msh(mshfile)

        #creating material
        my_material = amfe.KirchhoffMaterial(E=210E9, nu=0.3, rho=7.86E3, plane_stress=True, thickness=0.1)
        u_amfe, my_comp = eval_amfe_case(m1, my_material)

        master, feti_domain_dict = eval_superdomain(m1, my_material, splitting_tag ='partition_id', solver_type=solver_type)
        u_feti = amfe.FetiSolver.average_displacement_calc(my_comp,feti_domain_dict)

        ratio = np.linalg.norm(u_amfe - u_feti)/np.linalg.norm(u_amfe)
        print('\nRelative error between AMfe solver and local FETIsolver is : %e' %ratio)
        self.assertLessEqual(ratio, test_fetisubdomain.tolerance)

    def test_fetidomain_with_two_domains_cholsps(self):
        self.compute_fetidomain_with_two_domains(solver_type ='cholsps')
        
    def test_fetidomain_with_two_domains_svd(self):
        self.compute_fetidomain_with_two_domains(solver_type ='svd')

    def test_fetidomain_with_two_domains_splusps(self):
        self.compute_fetidomain_with_two_domains(solver_type ='splusps')

if __name__ == '__main__':
    unittest.main()
    #test = test_fetisubdomain()
    #test.compute_fetidomain_with_two_domains()
