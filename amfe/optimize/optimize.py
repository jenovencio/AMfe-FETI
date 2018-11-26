from scipy import optimize, sparse
import numpy as np
from unittest import TestCase, main
from scipy.misc import derivative
from scipy.sparse.linalg import LinearOperator
import numdifftools as nd

def func_wrapper(fun,x0_aug_real):
    ''' This function is a wrapper function for
    complex value function. The paramenter x0_aug_real
    is a augmented real numpy array with n/2 real array and
    n/2 imaginary array. The fun parameter is a complex function.
    
    Parameters:
    ----------
        fun : callable
            complex function which return a complex array
        x0_aug_real : np.array
            real array with 2m length
            
    Returns 
    --------
        fun_aug_real : np.array
            real array with 2m length
    '''
    n = len(x0_aug_real)
    x0_real = x0_aug_real[0:int(n/2)]
    x0_imag = x0_aug_real[int(n/2):]
    x0 = x0_real + 1J*x0_imag
    fun_aug_real = fun(x0)
    return np.concatenate([ fun_aug_real.real,  fun_aug_real.imag])

def hessp_wrapper(fun,x0_aug_real,p_aug_real):
    ''' This function is a wrapper function for
    complex Hessian product with two complex parameters. 
    The paramenter x0_aug_real is a augmented real numpy array with 2m real array 
    which the Hessian matrix will be evaluated and p is real 2m vector
    for computing the Hessian matrix product:

    H(x) * p
    
    Parameters:
    ----------
        fun : callable
            complex function which return a complex array
        x0_aug_real : np.array
            real array with 2m length
        p_aug_real : np.array
            real array with 2m length    

    Returns 
    --------
        fun_aug_real : np.array
            real array with 2m length
    '''
    x = real_array_to_complex(x0_aug_real)
    p = real_array_to_complex(p_aug_real)
    fun_aug_real = fun(x,p)
    return np.concatenate([ fun_aug_real.real,  fun_aug_real.imag])

def scalar_func_wrapper(fun,x0_aug_real):
    ''' This function is a wrapper function for
    complex value function. The paramenter x0_aug_real
    is a augmented real numpy array with n/2 real array and
    n/2 imaginary array. The fun parameter is a complex function.
    
    Parameters:
    ----------
        fun : callable
            complex function which return a complex number
        x0_aug_real : np.array
            real array with 2m length
            
    Returns 
    --------
        fun_aug_real : np.array
            real array with 2 number [x_real, x_imag]
    '''
    n = len(x0_aug_real)
    x0_real = x0_aug_real[0:int(n/2)]
    x0_imag = x0_aug_real[int(n/2):]
    x0 = x0_real + 1J*x0_imag
    fun_aug_real = fun(x0)
    return np.abs(fun_aug_real)

def jac_wrapper(Jfun,x0_aug_real):
    ''' This function is a wrapper function for
    complex Jacobian evaluation. The paramenter x0_aug_real
    is a augmented real numpy array with n/2 real array and
    n/2 imaginary array. The fun parameter is a complex function.
    
    Parameters:
    ----------
        fun : callable
            complex Jacobian callable function
        x0_aug_real : np.array
            real array with 2m length
            
    Returns 
    --------
        fun_aug_real : sparse.block_diag
            real 2D block diag sparse matrix 
    '''
    n = len(x0_aug_real)
    x0_real = x0_aug_real[0:int(n/2)]
    x0_imag = x0_aug_real[int(n/2):]
    x0 = x0_real + 1J*x0_imag
    Jfun_aug_real = Jfun(x0)
    
    #J_real = sparse.csc_matrix(Jfun_aug_real.real)
    #J_imag = sparse.csc_matrix(Jfun_aug_real.imag)
    #jac_real_row_1 = sparse.hstack((J_real,J_imag ))
    #jac_real_row_2 = sparse.hstack((-J_imag,J_real))
    #jac_real = sparse.vstack((jac_real_row_1, jac_real_row_2))

    jac_real = complex_matrix_to_real_block(Jfun_aug_real)

    return jac_real

def real_block_matrix_to_complex(M_block):
    ''' this function converts a block matrix
    M_block with format:

    M_block = [[ M_real, M_imag],
               [-M_imag, M_real]]

    into a a complex matrix

    M_complex = M_real + 1J*M_imag

    Parameters:
    -----------
        M_block : np.array
            real block matrix with real and imag blocks with [2m, 2m] shaoe

    Returns
    --------
    M_complex : np.array 
        a complex matrix with [m,m] shape

    '''
    m = int(M_block.shape[0]/2)
    M_block_real = M_block[:m,:m]
    M_block_imag = M_block[:m,m:]

    M_complex = M_block_real + 1J*M_block_imag 
    return M_complex

def real_array_to_complex(v_real):
    m = int(len(v_real)/2)
    return v_real[:m] + 1J*v_real[m:]

def complex_array_to_real(v_complex):
     return np.concatenate([ v_complex.real,  v_complex.imag])  

def complex_matrix_to_real_block(M_complex,sparse_matrix=True):
    ''' this function converts a complex matrix with format

    M_complex = M_real + 1J*M_imag
    
    into a block matrix M_block with format:

    M_block = [[ M_real, M_imag],
               [-M_imag, M_real]]


    Parameters:
    -----------
    M_complex : np.array 
        a complex matrix with [m,m] shape
    sparse_matrix: Bollean,  defaut = True 

    Returns
    --------
    M_block : np.array
            real block matrix with real and imag blocks with [2m, 2m] shape

    '''
    M_real = sparse.csc_matrix(M_complex.real)
    M_imag = sparse.csc_matrix(M_complex.imag)
    M_block_real_row_1 = sparse.hstack((M_real,M_imag ))
    M_block_real_row_2 = sparse.hstack((-M_imag,M_real))
    M_block = sparse.vstack((M_block_real_row_1, M_block_real_row_2))
    if sparse_matrix:
        return M_block 
    else:
        return M_block.toarray()

def complex_derivative(fun):
    pass

def complex_jacobian(fun):
    pass

def complex_hessian(fun):
    pass

def striu_from_vector(v,n):
    '''convert a vector in to a sparse
    upper triagule matrix
    '''
    indices = np.triu_indices(n)
    return sparse.csc_matrix((v, indices))


def root(fun, x0, args=(), method='hybr', jac=None, tol=None, callback=None, options=None):
    ''' This function is a wrapper for scipy.optimize.root which 
    deals with complex functions
    '''
    if  x0.dtype== 'complex':
        
        if callable(jac):
            jac_real = lambda x : jac_wrapper(jac,x).toarray()
            func_real = lambda x : func_wrapper(fun,x)
        elif jac is None:
            jac_real = None
            func_real = lambda x : func_wrapper(fun,x)
        elif jac==True or jac>0:
            raise('Not supported! please create a callable function')

        #initial guess
        m = len(x0)
        x0_real = np.concatenate([ x0.real,  x0.imag])    
        scipy_opt_obj = optimize.root(func_real, x0=x0_real, args=args, method=method, jac=jac_real , tol=tol, callback=callback, options=options)
        
        f_opt = scipy_opt_obj.fun[:m] + 1J*scipy_opt_obj.fun[m:]
        x_opt = scipy_opt_obj.x[:m] + 1J*scipy_opt_obj.x[m:]
        
        scipy_opt_obj.fun = f_opt
        scipy_opt_obj.x = x_opt
        
        if 'fjac' in scipy_opt_obj:
            scipy_opt_obj.fjac = real_block_matrix_to_complex(scipy_opt_obj.fjac)
        #    scipy_opt_obj.complex_QR = real_block_matrix_to_complex(scipy_opt_obj.fjac*scipy_opt_obj.ipvt)

        if 'cov_x' in scipy_opt_obj:
            scipy_opt_obj.cov_x = real_block_matrix_to_complex(scipy_opt_obj.cov_x)
    else:
        scipy_opt_obj = optimize.root(fun, x0, args=args, method=method, jac=jac, tol=tol, callback=callback, options=options)
    
    return scipy_opt_obj
 
def minimize(fun, x0, args=(), method=None, jac=None, hess=None, hessp=None, bounds=None, constraints=(), tol=None, callback=None, options=None):
    ''' This function is a wrapper for scipy.optimize.minimize  which 
    deals with complex functions
    '''
    if  x0.dtype== 'complex':
        
        #f_real return a float value given a augmented array with [x_real, x_imag]
        #f_real = lambda x_aug : np.linalg.norm(np.abs(scalar_func_wrapper(fun,x_aug)))**2
        f_real = lambda x : scalar_func_wrapper(fun,x)

        m = len(x0)
        x0_real = np.concatenate([ x0.real,  x0.imag])    

        if jac is None:
            jac_real = None
        elif callable(jac):
            jac_real = lambda x : func_wrapper(jac,x)
        else:
            raise('Jacobian type not supported')

        if hess is None:
            hess_real = None
        elif callable(hess):
            hess_real = lambda x : jac_wrapper(hess,x).toarray()
        else:
            raise('Hessian type not supported')

        if hessp is None:
            hessp_real = None
        elif callable(hessp):
            hessp_real = lambda x, p : hessp_wrapper(hessp,x,p)
            #hessp_real = LinearOperator((2*m,2*m), matvec = lambda x, p : func_wrapper(hessp,x).dot(p)
        else:
            raise('Hessian product "hessp" type not supported')

        scipy_opt_obj = optimize.minimize(f_real, x0_real, args=args, method=method, jac=jac_real, hess=hess_real, hessp=hessp_real, 
                                          bounds=bounds, constraints=constraints, tol=tol, callback=callback, options=options)
        
        j_opt = scipy_opt_obj.jac[:m] + 1J*scipy_opt_obj.jac[m:]
        x_opt = scipy_opt_obj.x[:m] + 1J*scipy_opt_obj.x[m:]
        
        #scipy_opt_obj.fun = f_opt
        scipy_opt_obj.x = x_opt

        if 'jac' in scipy_opt_obj:
            scipy_opt_obj.jac = j_opt

        if 'hess' in scipy_opt_obj:
            scipy_opt_obj.hess = real_block_matrix_to_complex(scipy_opt_obj.hess)

        
    else:
        scipy_opt_obj = optimize.minimize(fun, x0, args=args, method=method, jac=jac, hess=hess, hessp=hessp, 
                                          bounds=bounds, constraints=constraints, tol=tol, callback=callback, options=options)
    
    return scipy_opt_obj

def fsolve(func, x0, args=(), fprime=None, full_output=0, col_deriv=0, xtol=1.49012e-08, maxfev=0, band=None, epsfcn=None, factor=100, diag=None):

    if  x0.dtype== 'complex':
        
        if callable(fprime):
            fprime_real = lambda x : jac_wrapper(fprime,x).toarray()

        elif fprime is None:
            fprime_real = None
            
        elif fprime==True or jac>0:
            raise('Not supported! please create a callable function')
        
        func_real = lambda x : func_wrapper(func,x)
        x0_real = complex_array_to_real(x0)
        info_dict = optimize.fsolve(func_real, x0_real, args=args, fprime=fprime_real, full_output=full_output, col_deriv=col_deriv, xtol=xtol, maxfev=maxfev, band=band, epsfcn=epsfcn, factor=factor, diag=diag)
        info_dict = list(info_dict)
        info_dict[0] = real_array_to_complex( info_dict[0])
        info_dict[1]['fjac'] = real_block_matrix_to_complex(info_dict[1]['fjac'])
        info_dict[1]['r'] = real_block_matrix_to_complex( striu_from_vector(info_dict[1]['r'],len(x0_real)).toarray() )
        info_dict[1]['qtf'] = real_array_to_complex(info_dict[1]['qtf'])
        info_dict[1]['fvec'] = real_array_to_complex(info_dict[1]['fvec'])

        return info_dict
    else:
        return  optimize.fsolve(func, x0, args=args, fprime=fprime, full_output=full_output, col_deriv=col_deriv, xtol=xtol, maxfev=maxfev, band=band, epsfcn=epsfcn, factor=factor, diag=diag)
        
    
    



class  Test_root(TestCase):
    def setUp(self):
        omega = 10
        m1 = 1.0
        m2 = 2.0
        c1 =  0.5
        c2 =  0.1
        k1 = 10.0
        k2 = 5.0
        f1 = 00.0 + 1J*0.0
        f2 = 50.0 + 1J*0.0

        K = np.array([[k1+k2,-k2],
                      [-k2,k2]])
        C = np.array([[c1+c2,-c2],
                      [-c2,c2]])
        M = np.array([[m1,-0],
                      [0,m2]])

        f = np.array([f1,f2])
        Z = K + 1J*omega*C - omega**2*M

        self.omega = omega
        self.K = K
        self.Z = Z
        self.f = f

        self.x_sol = np.linalg.solve(Z,f)
        self.fun1= lambda x : x.conj().dot(0.5*Z.dot(x) - f)
        self.fun2= lambda x : (Z.dot(x) - f).conj().T.dot((Z.dot(x) - f))

        self.f_prime1 = lambda x : Z.dot(x) - f
        self.f_prime2 = lambda x : Z.conj().T.dot((Z.dot(x) - f))


    def fun_broyden(self,x):
        f = (3 - x) * x + 1
        f[1:] -= x[:-1]
        f[:-1] -= 2 * x[1:]
        return f

    def fun_rosenbrock(self,x):
        return np.array([10 * (x[1] - x[0]**2), (1 - x[0])])

    def test_root_krylov_2D(self):

        Z = self.Z
        f = self.f
        fun = lambda x : Z.dot(x) - f
        
        x0 = np.zeros(2,dtype=np.complex)
        opt_obj = root(fun, x0=x0, method='krylov')
        x_sol = np.linalg.solve(Z,f)
        np.testing.assert_array_almost_equal(opt_obj.x.real, x_sol.real,  decimal=6 )
        np.testing.assert_array_almost_equal(opt_obj.x.imag, x_sol.imag,  decimal=6 )
    
    def test_root_lm_2D_analytical_jac_complex(self):
        
        Z = self.Z
        f = self.f
        fun = lambda x : Z.dot(x) - f
        J = lambda x : Z

        x0 = np.zeros(2,dtype=np.complex)
        opt_obj = root(fun, x0=x0, jac = J, method='lm')
        x_sol = np.linalg.solve(Z,f)
        np.testing.assert_array_almost_equal(opt_obj.x.real, x_sol.real,  decimal=6 )
        np.testing.assert_array_almost_equal(opt_obj.x.imag, x_sol.imag,  decimal=6 )

    def test_root_lm_2D_jac_complex(self):
        
        Z = self.Z
        f = self.f
        fun = lambda x : Z.dot(x) - f

        x0 = np.zeros(2,dtype=np.complex)
        opt_obj = root(fun, x0=x0, jac = None, method='lm')
        x_sol = np.linalg.solve(Z,f)
        np.testing.assert_array_almost_equal(opt_obj.x.real, x_sol.real,  decimal=6 )
        np.testing.assert_array_almost_equal(opt_obj.x.imag, x_sol.imag,  decimal=6 )

    def test_root_hybr_2D_jac_complex(self):
        
        Z = self.Z
        f = self.f
        fun = lambda x : Z.dot(x) - f
        J = lambda x : Z

        x0 = np.zeros(2,dtype=np.complex)
        opt_obj = root(fun, x0=x0, jac = J, method='hybr')
        x_sol = np.linalg.solve(Z,f)
        np.testing.assert_array_almost_equal(opt_obj.x.real, x_sol.real,  decimal=6 )
        np.testing.assert_array_almost_equal(opt_obj.x.imag, x_sol.imag,  decimal=6 )

    def test_root_broyden1_2D_jac_complex(self):
        
        Z = self.Z
        f = self.f
        fun = lambda x : Z.dot(x) - f
        J = lambda x : Z

        x0 = np.zeros(2,dtype=np.complex)
        opt_obj = root(fun, x0=x0, jac = J, method='broyden1')
        x_sol = np.linalg.solve(Z,f)
        np.testing.assert_array_almost_equal(opt_obj.x.real, x_sol.real,  decimal=6 )
        np.testing.assert_array_almost_equal(opt_obj.x.imag, x_sol.imag,  decimal=6 )

    def test_root_lm_2D_jac(self):
        
        K = self.K
        f = self.f.real
        fun = lambda x : K.dot(x) - f
        J = lambda x : K

        x0 = np.zeros(2)
        opt_obj = root(fun, x0=x0, jac = J, method='lm')
        x_sol = np.linalg.solve(K,f)
        np.testing.assert_array_almost_equal(opt_obj.x, x_sol,  decimal=6 )
     
    def test_fsolve_2D_jac(self):
        
        K = self.K
        f = self.f.real
        x_sol = np.linalg.solve(K,f)

        fun = lambda x : K.dot(x) - f
        J = lambda x : K

        x0 = np.zeros(2)
        info_dict = fsolve(fun, x0=x0, fprime = J, full_output=1)
        
        x = info_dict[0]
        info_dict = info_dict[1]

        np.testing.assert_array_almost_equal(x, x_sol,  decimal=6 )
        
    def test_fsolve_2D_complex_jac(self):
        Z = self.Z
        f = self.f
        x_sol = np.linalg.solve(Z,f)

        fun = lambda x : Z.dot(x) - f
        J = lambda x : Z

        x0 = np.zeros(2,dtype=np.complex)

        info_dict = fsolve(fun, x0=x0, fprime = J, full_output=1)
        
        x = info_dict[0]
        info_dict = info_dict[1]
        np.testing.assert_array_almost_equal(x.real, x_sol.real,  decimal=6 )
        np.testing.assert_array_almost_equal(x.imag, x_sol.imag,  decimal=6 )

    def test_root_krylov(self):
        omega = 10
        m = 1.0
        c = 0.5
        k = 10.0
        f = 50.0 + 1J*0.0
        z = k + 1J*omega*c - omega**2*m
        fun = lambda x : z*x - f
        x0 = np.array([0.0 + 1J*0.0])
        opt_obj = root(fun, x0=x0, method='krylov')
        x_sol = f/z
        self.assertAlmostEqual(opt_obj.x.real[0], x_sol.real,  places=6 )
        self.assertAlmostEqual(opt_obj.x.imag[0], x_sol.imag,  places=6 )

    def test_minimize_cg(self):
        n = 2
        x0 = np.array([0.0]*n)
        xopt = np.array([1.0]*n)
        opt_obj = minimize(optimize.rosen,x0,method='cg')

    def test_minimize_complex_cg(self):
        Z = self.Z
        f = self.f
        x_sol = np.linalg.solve(Z,f)

        fun1 = lambda x : x.dot(Z.dot(x) - f)
        fun2 = lambda x : np.linalg.norm(Z.dot(x) - f)**2

        x0 = np.ones(2,dtype=np.complex)
        opt_obj = minimize(fun2, x0=x0, jac = None, method='cg', tol=1E-12)
        print('Number of function evaluatons with numerical jacobian %i' %opt_obj.nfev)

        np.testing.assert_array_almost_equal(opt_obj.x.real, x_sol.real,  decimal=6 )
        np.testing.assert_array_almost_equal(opt_obj.x.imag, x_sol.imag,  decimal=6 )

    def test_minimize_complex_cg_with_jac(self):
        Z = self.Z
        f = self.f
        x_sol = np.linalg.solve(Z,f)
        fun1= lambda x : x.conj().dot(0.5*Z.dot(x) - f)
        fun2= lambda x : (Z.dot(x) - f).conj().T.dot((Z.dot(x) - f))

        f_prime1 = lambda x : Z.dot(x) - f
        f_prime2 = lambda x : Z.conj().T.dot((Z.dot(x) - f))

        x0 = np.ones(2,dtype=np.complex)
        #opt_obj = minimize(fun1, x0=x0, jac = f_prime1, method='cg', tol=1E-12)
        opt_obj = minimize(fun2, x0=x0, jac = f_prime2, method='cg', tol=1E-12)
        print('Alg: cg ->Number of function evaluatons with analytical jacobian %i' %opt_obj.nfev)

        np.testing.assert_array_almost_equal(opt_obj.x.real, x_sol.real,  decimal=6 )
        np.testing.assert_array_almost_equal(opt_obj.x.imag, x_sol.imag,  decimal=6 )

    def test_minimize_cg_with_jac(self):
            
        K = self.K
        f = self.f.real
        x_sol = np.linalg.solve(K,f)

        fun1= lambda x : x.dot(0.5*K.dot(x) - f)
        fun2= lambda x : 0.5*np.linalg.norm(K.dot(x) - f)**2

        f_prime1 = lambda x : (K.dot(x) - f)
        f_prime2 = lambda x : K.T.dot(K.dot(x) - f)

        x0 = np.ones(2)
        opt_obj1 = minimize(fun1, x0=x0, jac = f_prime1, method='cg', tol=1E-12)
        np.testing.assert_array_almost_equal(opt_obj1.x.real, x_sol.real,  decimal=6 )
        np.testing.assert_array_almost_equal(opt_obj1.x.imag, x_sol.imag,  decimal=6 )

        opt_obj2 = minimize(fun2, x0=x0, jac = f_prime2, method='cg', tol=1E-12)
            
        np.testing.assert_array_almost_equal(opt_obj2.x.real, x_sol.real,  decimal=6 )
        np.testing.assert_array_almost_equal(opt_obj2.x.imag, x_sol.imag,  decimal=6 )

    def test_minimize_newton_cg_with_jac_and_hessp(self):
            
        K = self.K
        f = self.f.real
        x_sol = np.linalg.solve(K,f)

        fun2= lambda x : 0.5*np.linalg.norm(K.dot(x) - f)**2

        f_prime2 = lambda x : K.T.dot(K.dot(x) - f)

        hessp2 = lambda x, p  : K.T.dot(K.dot(p))

        x0 = np.ones(2)
       
        opt_obj2 = minimize(fun2, x0=x0, jac = f_prime2, hessp=hessp2,method='Newton-CG', tol=1E-12)
            
        np.testing.assert_array_almost_equal(opt_obj2.x.real, x_sol.real,  decimal=6 )
        np.testing.assert_array_almost_equal(opt_obj2.x.imag, x_sol.imag,  decimal=6 )

    def test_minimize_complex_ncg_with_jac_hess(self):
        Z = self.Z
        f = self.f
        x_sol = np.linalg.solve(Z,f)
        fun1= lambda x : x.conj().dot(0.5*Z.dot(x) - f)
        fun2= lambda x : (Z.dot(x) - f).conj().T.dot((Z.dot(x) - f))

        f_prime1 = lambda x : Z.dot(x) - f
        f_prime2 = lambda x : Z.conj().T.dot((Z.dot(x) - f))

        hess1 = lambda x : Z
        hess2 = lambda x : Z.conj().T.dot(Z)

        x0 = np.ones(2,dtype=np.complex)
        #opt_obj = minimize(fun1, x0=x0, jac = f_prime1, method='cg', tol=1E-12)
        opt_obj = minimize(fun2, x0=x0, jac = f_prime2, hess=hess2, method='trust-ncg', tol=1E-12)
        print('Alg: trust-ncg -> Number of function evaluatons with analytical Jacobian and Hessian %i' %opt_obj.nfev)

        np.testing.assert_array_almost_equal(opt_obj.x.real, x_sol.real,  decimal=6 )
        np.testing.assert_array_almost_equal(opt_obj.x.imag, x_sol.imag,  decimal=6 )

    def test_minimize_complex_newton_cg_with_jac_hess(self):
        Z = self.Z
        f = self.f
        x_sol = np.linalg.solve(Z,f)
        fun1= lambda x : x.conj().dot(0.5*Z.dot(x) - f)
        fun2= lambda x : (Z.dot(x) - f).conj().T.dot((Z.dot(x) - f))

        f_prime1 = lambda x : Z.dot(x) - f
        f_prime2 = lambda x : Z.conj().T.dot((Z.dot(x) - f))

        hess1 = lambda x : Z
        hess2 = lambda x : Z.conj().T.dot(Z)

        x0 = np.ones(2,dtype=np.complex)
        #opt_obj = minimize(fun1, x0=x0, jac = f_prime1, method='cg', tol=1E-12)
        opt_obj = minimize(fun2, x0=x0, jac = f_prime2, hess=hess2, method='Newton-CG', tol=1E-12)
        print('Alg: Newton-CG -> Number of function evaluatons with analytical Jacobian and Hessian %i' %opt_obj.nfev)

        np.testing.assert_array_almost_equal(opt_obj.x.real, x_sol.real,  decimal=6 )
        np.testing.assert_array_almost_equal(opt_obj.x.imag, x_sol.imag,  decimal=6 )

    def test_minimize_complex_newton_cg_with_jac_hessp(self):
        Z = self.Z
        f = self.f
        x_sol = self.x_sol

        fun2= self.fun2

        f_prime2 = self.f_prime2


        hessp2 = lambda x, p : Z.conj().T.dot(Z).dot(p)

        x0 = np.ones(2,dtype=np.complex)
        opt_obj = minimize(fun2, x0=x0, jac = f_prime2, hessp=hessp2, method='Newton-CG', tol=1E-12)
        print('Alg: Newton-CG -> Number of function evaluatons with analytical Jacobian and Hessian product %i' %opt_obj.nfev)

        np.testing.assert_array_almost_equal(opt_obj.x.real, x_sol.real,  decimal=6 )
        np.testing.assert_array_almost_equal(opt_obj.x.imag, x_sol.imag,  decimal=6 )

    def test_matrix_conversion(self):

         Z = self.Z
         Z_block = complex_matrix_to_real_block(Z,sparse_matrix=False)

         Z_ = real_block_matrix_to_complex(Z_block)

         abs_dif = abs(Z - Z_).flatten()

         np.testing.assert_array_almost_equal( abs_dif, np.zeros(len(abs_dif)),  decimal=10 )

if __name__ == '__main__':
    main()