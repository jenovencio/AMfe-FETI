
import amfe
from amfe.amna import cholsps, splusps, pinv_and_null_space_svd, P_inverse, is_null_space
import numpy as np
import time as timeit
import unittest



def test_null_space(alg_type,K1,tolerance=1.0E-3, solver_tol=1.0E-8):
    ''' This function test the kernel calculation of 
    the pseudo inverse method which calculates the Kernel of
    K1 matrix

    arguments:

        alg_type : str
            str og the solver type to be called and tested
        K1 : np.array
            matrix to calculated the Kernel
        tolerance : float
            tolerance for the norm of the vector v in R
           by the the K1 matrix, which represents a tolerance for
           checking if v in R is really a kernel vector of K1
        solver_tol : float
            solver tolerance for a pivot or eigenvalue be considered small

    returns
        null_space_size : int
            null space size found
        elapsed: float
            time to calculate the kernel and the pseudo-inverse
            
    '''
    
    start = timeit.clock() 
    pinv_obj = amfe.amna.P_inverse()
    K_inv = pinv_obj.compute(K1, tol = solver_tol, solver_opt = alg_type)
    R1 = K_inv.null_space
    elapsed = timeit.clock()
    elapsed = elapsed - start

    null_space_size = 0
    for v in R1.T:
        if is_null_space(K1,v, tol=tolerance):
            null_space_size += 1

    if alg_type == 'cholsps':
        str_alg = 'Cholesky'
    elif alg_type == 'splusps':
        str_alg = 'SuperLU'
    elif alg_type == 'svd':
        str_alg = 'SVD'

    print('Elapsed time for kernel calculation  = %f' %elapsed)
    print('%s Kernel calculated rank =%i' %(str_alg,np.linalg.matrix_rank(R1)))
    print('%s Kernel real rank =%i' %(str_alg,null_space_size))


    return null_space_size, elapsed  





class test_null_space_solvers(unittest.TestCase):

    def calc_null_space_size_3(self,list_alf_type = ['cholsps', 'splusps', 'svd'], solver_tolerance=1.0E-8):
        ''' This function load K matrix with kernel size equal 3
        and computes the Kernel using diffente algorithms
        and check is the calculated kernel is right
        '''
        #list_alf_type = ['cholsps', 'splusps', 'svd']
        kernel_tolerance = 1.0E-2
        print_columns = 50
        print('#'*print_columns)
        print('Testing Null space calculation')
        num_of_matrices = 9
        for i in range(1,num_of_matrices+1):
            print('#'*print_columns)
            str_path = 'K_matrices_kernel_3/K_%i.pkl' %i
            print('Loaing %s' %str_path)
            K1 = amfe.load_obj(str_path)
            
            for alg_type in list_alf_type:
                print('*'*print_columns)
                print('Algotihm type choosen is %s' %alg_type)
                null_space_size, elapsed  = test_null_space(alg_type,K1,tolerance=kernel_tolerance,solver_tol= solver_tolerance)
                self.assertEqual(null_space_size,3)
                print('*'*print_columns)
            print('#'*print_columns)

    def test_null_space_size_3_cholsps(self):
        print_columns = 50
        str_label = 'cholsps'
        solver_tolerance = 1.0E-8
        hash_size = int((print_columns - len(str_label))/2)
        print('#'*print_columns )
        print('#'*hash_size + str_label.upper() + '#'*hash_size)
        print('#'*print_columns )
        list_alf_type = [str_label]
        self.calc_null_space_size_3(list_alf_type,solver_tolerance)

    def test_null_space_size_3_svd(self):
        print_columns = 50
        str_label = 'svd'
        solver_tolerance = 1.0E-8
        hash_size = int((print_columns - len(str_label))/2)
        print('#'*print_columns )
        print('#'*hash_size + str_label.upper() + '#'*hash_size)
        print('#'*print_columns )
        list_alf_type = [str_label]
        self.calc_null_space_size_3(list_alf_type,solver_tolerance)
    
    def test_null_space_size_3_slusps(self):     
        print_columns = 50
        str_label = 'splusps'
        solver_tolerance = 1.0E-8
        hash_size = int((print_columns - len(str_label))/2)
        print('#'*print_columns )
        print('#'*hash_size + str_label.upper() + '#'*hash_size)
        print('#'*print_columns )
        list_alf_type = [str_label]
        self.calc_null_space_size_3(list_alf_type,solver_tolerance)
           
    def calc_null_space_size_6(self, list_alf_type = ['cholsps', 'splusps', 'svd'], solver_tolerance=1.0E-8):
        ''' This function load K matrix with kernel size equal 3
        and computes the Kernel using diffente algorithms
        and check is the calculated kernel is right
        '''
        #list_alf_type = ['cholsps', 'splusps', 'svd']
        kernel_tolerance = 1.0E-2
        print_columns = 50
        print('#'*print_columns)
        print('Testing Null space calculation')
        num_of_matrices = 3
        for i in range(1,num_of_matrices+1):
            print('#'*print_columns)
            str_path = 'K_matrices_kernel_6/K%i.pkl' %i
            print('Loading %s' %str_path)
            K1 = amfe.load_obj(str_path)
            
            for alg_type in list_alf_type:
                print('*'*print_columns)
                print('Algotihm type choosen is %s' %alg_type)
                null_space_size, elapsed  = test_null_space(alg_type,K1,tolerance=kernel_tolerance,solver_tol= solver_tolerance)
                self.assertEqual(null_space_size,6)
                print('*'*print_columns)
            print('#'*print_columns)
            
    def test_null_space_size_6_svd(self):
        print_columns = 50
        str_label = 'svd'
        solver_tolerance = 1.0E-8
        hash_size = int((print_columns - len(str_label))/2)
        print('#'*print_columns )
        print('#'*hash_size + str_label.upper() + '#'*hash_size)
        print('#'*print_columns )
        list_alf_type = [str_label]
        self.calc_null_space_size_6(list_alf_type,solver_tolerance)     
    
    def test_null_space_size_6_slusps(self):     
        print_columns = 50
        str_label = 'splusps'
        solver_tolerance = 1.0E-8
        hash_size = int((print_columns - len(str_label))/2)
        print('#'*print_columns )
        print('#'*hash_size + str_label.upper() + '#'*hash_size)
        print('#'*print_columns )
        list_alf_type = [str_label]
        self.calc_null_space_size_6(list_alf_type,solver_tolerance)
        
    def test_null_space_size_6_cholsps(self):
        print_columns = 50
        str_label = 'cholsps'
        solver_tolerance = 1.0E-8
        hash_size = int((print_columns - len(str_label))/2)
        print('#'*print_columns )
        print('#'*hash_size + str_label.upper() + '#'*hash_size)
        print('#'*print_columns )
        list_alf_type = [str_label]
        self.calc_null_space_size_6(list_alf_type,solver_tolerance)
        
        
if __name__ == '__main__':
    #unittest.main()
    test = test_null_space_solvers()
    #test.test_null_space_size_3_slusps()
    #test.test_null_space_size_3_cholsps()
    test.test_null_space_size_6_svd()
    test.test_null_space_size_6_slusps()
    test.test_null_space_size_6_cholsps()