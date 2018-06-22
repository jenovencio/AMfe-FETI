
import sys 
import amfe
import numpy as np
import scipy
import copy
import time as timeit
import unittest

#------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------
#                                TEST CASES
#------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------
msh_dict = {}
msh_dict[0] = amfe.amfe_dir('meshes/test_meshes/1_rectangle_8_elem.msh')
msh_dict[1] = amfe.amfe_dir('meshes/test_meshes/retangule_5_by_2_quad_par_1.msh')
msh_dict[2] = amfe.amfe_dir('meshes/test_meshes/retangule_5_by_2_ref_quad_par_1.msh')
msh_dict[3] = amfe.amfe_dir('meshes/test_meshes/retangule_5_by_2_ref_tri_par_1.msh')
msh_dict[4] = amfe.amfe_dir('meshes/test_meshes/retangule_5_by_2_tri_par_1.msh')
msh_dict[5] = amfe.amfe_dir('meshes/test_meshes/retangule_square_hole_5_by_2_ref_tri_par_1.msh')
msh_dict[6] = amfe.amfe_dir('meshes/test_meshes/retangule_square_hole_5_by_2_quad_par_1.msh')

domain_id = {}
domain_id[0] = 6
domain_id[1] = 6
domain_id[2] = 6
domain_id[3] = 6
domain_id[4] = 6
domain_id[5] = 6
domain_id[6] = 6

bc = {}
bc[0] = [3,1,2]
bc[1] = [3,1,10]
bc[2] = [3,1,10]
bc[3] = [3,1,10]
bc[4] = [3,1,10]
bc[5] = [3,1,10]
bc[6] = [3,1,10]
#------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------


def create_base_case(id):
    ''' This function create a base Amfe case for testing

    argument
        id : int
            case id with diferent meshes
    return 
        my_comp : MechanicalSystem obj
            
    '''


    # select mesh to be plotted
    mshfile = msh_dict[id]

    # instanciating mesh class
    m1 = amfe.Mesh()
    m1.import_msh(mshfile)

    my_material = amfe.KirchhoffMaterial(E=210E9, nu=0.3, rho=7.86E3, plane_stress=True, thickness=0.1)

    my_comp = amfe.MechanicalSystem()
    my_comp.set_mesh_obj(m1)
    my_comp.set_domain(domain_id[id],my_material)
    return my_comp



def solve_base_amfe_case(comp,id, value = 1e8):
    ''' this function receives a mechanical component
    and return the displacement

    argument
        comp : MechanicalSystem obj
        id : int
            id with the test case
        value: float
            amplification factor for the Neumann B.C.

    return 
        ue : np.array
            displacement of the component
        elapsed : float
            execution time of the standard solver
        
    '''
    # solving problem with dirichlet
    my_comp = copy.deepcopy(comp)
    my_comp.apply_neumann_boundaries(bc[id][0], value, 'normal')
    my_comp.apply_dirichlet_boundaries(bc[id][1],'x')
    my_comp.apply_dirichlet_boundaries(bc[id][2],'y')
    s = amfe.LinearStaticsSolver(my_comp)
    
    start = timeit.clock() 
    s.solve()
    elapsed = timeit.clock()
    elapsed = elapsed - start
    ue = my_comp.u_output[1]
    return ue, elapsed

def test_singular_solver(comp, id, value = 1e8, solver_type='slusps'):
    ''' This fucntion solve a singular problem 
    using amna singular solvers. Avaliable solver are
    cholsps
    slusps
    svd

    argument
        comp : MechanicalSystem obj
        id : int
            case id
        value : float
            amplification factor for the Neumann B.C.
        solver_type : str
            string with the solver type

        return 
            up : np.array
                displacement of the component
            elapsed : float
                execution time of the standard solver
            K_inv : callable
                pseudo inverse operator

    '''
    my_comp2 = copy.deepcopy(comp)

    # solving problem with pure Neumman
    my_comp2.apply_neumann_boundaries(bc[id][0], value, 'normal')
    my_comp2.apply_neumann_boundaries(bc[id][1], value, 'normal')

    start = timeit.clock() 
    K, f_ext = my_comp2.assembly_class.assemble_k_and_f_neumann()
    K, f_int = my_comp2.assembly_class.assemble_k_and_f()
    K_inv = amfe.amna.pinv_and_null_space.compute(K,solver_opt = solver_type)
    up = K_inv.apply(f_ext)
    elapsed = timeit.clock()
    elapsed = elapsed - start
    
    return up, elapsed, K_inv


def calc_error_between_base_case_and_singular_solver(ue,up,K_inv):
    ''' This function calculates the error between the standard Amfe 
    solver and the singular solver.

    argument 
        ue : np.array
            displacement calculated by the standard solver
        up : np.array
            displacement calculated by the singular solver
        Kinv : callable
            singular solver object with kernel of the operator

        return 
            ratio : float
                relation between the norm of the error by the
                norm of the ue vector

    '''
    R = K_inv.null_space
    error = ue - up
    alpha = np.linalg.solve(R.T.dot(R),R.T.dot(error))

    u_corr = K_inv.calc_kernel_correction(alpha)
    new_error = ue - (up + u_corr)
    norm_error = np.linalg.norm(new_error)
    norm_ue = np.linalg.norm(ue)

    ratio = norm_error/norm_ue
    return ratio


def main(id,value,solver_type):
    #id = 0
    #value = 1e8
    #solver_type='splusps'

    my_comp = create_base_case(id)

    ue, elapsed_amfe = solve_base_amfe_case(my_comp,id,value)

    up, elapsed_sps, K_inv = test_singular_solver(my_comp,id,value,solver_type=solver_type)
    ratio = calc_error_between_base_case_and_singular_solver(ue,up,K_inv)

    print("Time spent in AMfe standard solver is: %f" %elapsed_amfe)
    print("Time spent in AMfe singular solver %s is: %f" %(solver_type,elapsed_sps))
    print('AMfe standard is %f faster than the singular solver' %(elapsed_sps/elapsed_amfe))
    print("erro between standard and singular solver is %f" %ratio)
    return ratio

class test_singular_solvers(unittest.TestCase):
    
    tolerance = 1.0e-2
    num_of_hashs = 50
    num_of_cases = 7
    def test_splusps(self):

        solver_type = 'splusps'
        print('#'*test_singular_solvers.num_of_hashs)
        print('Testing %s' %solver_type)
        value = 1e8
        for id in range(test_singular_solvers.num_of_cases):
            print('#'*test_singular_solvers.num_of_hashs)
            print('Running case %i' %id)
            ratio = main(id,value,solver_type)
            self.assertLessEqual(ratio, test_singular_solvers.tolerance)
            print('#'*test_singular_solvers.num_of_hashs)
    def test_cholsps(self):
        solver_type = 'cholsps'
        print('#'*test_singular_solvers.num_of_hashs)
        print('Testing %s' %solver_type)
        value = 1e8
        for id in range(test_singular_solvers.num_of_cases):
            print('#'*test_singular_solvers.num_of_hashs)
            print('Running case %i' %id)
            ratio = main(id,value,solver_type)
            self.assertLessEqual(ratio, test_singular_solvers.tolerance)
            print('#'*test_singular_solvers.num_of_hashs)

    def test_svd(self):
        print('#'*test_singular_solvers.num_of_hashs)
        solver_type = 'svd'
        print('Testing %s' %solver_type)
        value = 1e8
        for id in range(test_singular_solvers.num_of_cases):
            print('#'*test_singular_solvers.num_of_hashs)
            print('Running case %i' %id)
            ratio = main(id,value,solver_type)
            self.assertLessEqual(ratio, test_singular_solvers.tolerance)
            print('#'*test_singular_solvers.num_of_hashs)


if __name__ == '__main__':
    unittest.main()
