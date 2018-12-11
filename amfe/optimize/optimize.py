from scipy import optimize, sparse
import numpy as np
from unittest import TestCase, main
from scipy.misc import derivative
from scipy.sparse.linalg import LinearOperator
from scipy import linalg
import numdifftools as nd
import matplotlib.pyplot as plt

import sys
import os

# getting amfe folder
if sys.platform[:3]=='win':
    path_list = os.path.dirname(__file__).split('\\')[:-1]
    amfe_dir = '\\'.join(path_list)
    sys.path.append(amfe_dir)
elif sys.platform[:3]=='lin':
    path_list = os.path.dirname(__file__).split('/')[:-1]
    amfe_dir = '/'.join(path_list)
    sys.path.append(amfe_dir)
else :
    raise('Plataform %s is not supported  ' %sys.platform)

from amfe.frequency_module.frequency import cos_bases_for_MHBM, create_Z_matrix, linear_harmonic_force, hbm_complex_bases
from amfe.operators.operators import ReshapeOperator

def func_wrapper(fun,x0_aug_real,extra_arg=None):
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
        extra_arg : np.array
            real array with extra function parameters 
            (real paramenters in the original function )
    Returns 
    --------
        fun_aug_real : np.array
            real array with 2m length
    '''
    n = len(x0_aug_real)
    x0_real = x0_aug_real[0:int(n/2)]
    x0_imag = x0_aug_real[int(n/2):]
    x0 = x0_real + 1J*x0_imag
    if extra_arg is None:
        fun_aug_real = fun(x0)
    else:
        fun_aug_real = fun(x0,extra_arg)

    return complex_array_to_real(fun_aug_real)

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

def jac_wrapper(Jfun,x0_aug_real,extra_arg=None):
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
    if extra_arg is None:
        Jfun_aug_real = Jfun(x0)
    else:
        Jfun_aug_real = Jfun(extra_arg)(x0)
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
    try:
        m = int(len(v_real)/2)
        return v_real[:m] + 1J*v_real[m:]
    except: 
        print('Warning! Could not convert to complex number')
        return v_real

def complex_array_to_real(v_complex):
    if v_complex.shape:
        return np.concatenate([ v_complex.real,  v_complex.imag])  
    else:
        return np.array([ v_complex.real,  v_complex.imag])  

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

def complex_derivative(fun,n=1):

    fun_real = lambda x_real : complex_array_to_real(fun(x_real[0]+1J*x_real[1]))
    gradient_real = nd.Jacobian(fun_real,n=n)
    return lambda x : gradient_real(np.array([x.real,x.imag]))[0].dot([1,-1J])
  
def complex_jacobian(fun,n=1):
    
    fun_real = lambda x_real : complex_array_to_real(fun(real_array_to_complex(x_real)))
    gradient_real = nd.Jacobian(fun_real,n=n)
    return lambda x : real_block_matrix_to_complex(gradient_real(complex_array_to_real(x)).T)

def complex_hessian(fun):
    ''' This function computes the Hessian matrix of
    a complex function which maps complex vector of size n
    into the complex plane 
    fun : C^n -> C

    Paramenters:
    --------
        fun : callable
            function which maps complex vector of size n
            into the complex plane 
    Returns:
    --------
        hess : callable
            complex Hessian matrix 
    '''
    
    fun_real = lambda x_real : fun(real_array_to_complex(x_real)).real
    fun_imag = lambda x_real : fun(real_array_to_complex(x_real)).imag

    Hessian_real = nd.Hessian(fun_real)
    Hessian_imag = nd.Hessian(fun_imag)

    return lambda x : (real_block_matrix_to_complex(Hessian_real(complex_array_to_real(x))) + \
                        1J*real_block_matrix_to_complex(Hessian_imag(complex_array_to_real(x)))).T

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

def fsolve(func, x0, args=(), fprime=None, full_output=True, col_deriv=0, xtol=1.49012e-08, maxfev=0, band=None, epsfcn=None, factor=100, diag=None):

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
        if full_output:
            info_dict = list(info_dict)
            print(info_dict)
            info_dict[0] = real_array_to_complex( info_dict[0])
            info_dict[1]['fjac'] = real_block_matrix_to_complex(info_dict[1]['fjac'])
            info_dict[1]['r'] = real_block_matrix_to_complex( striu_from_vector(info_dict[1]['r'],len(x0_real)).toarray() )
            info_dict[1]['qtf'] = real_array_to_complex(info_dict[1]['qtf'])
            info_dict[1]['fvec'] = real_array_to_complex(info_dict[1]['fvec'])
        else:
            info_dict = real_array_to_complex( info_dict)

        return info_dict
    else:
        return  optimize.fsolve(func, x0, args=args, fprime=fprime, full_output=full_output, col_deriv=col_deriv, xtol=xtol, maxfev=maxfev, band=band, epsfcn=epsfcn, factor=factor, diag=diag)
        
def continuation(fun,x0,p_range,p0=None, jacx=None, jacp=None ,step=1.0,max_int=500,max_int_corr=50,tol=1.0E-6,max_dp=1.0,
                 correction_method='moore_penrose',root_method='lm',print_mode=False):
    ''' This function applies a continuation technique
    in the fun(x,p) which 

    Parameters:
    ------------
        fun : callable
            callable function with two parameters x and p
        x0 : np.array
            initial guess for the initial point
        p_range : tuple 
            tuple with lower and upper limit for the p parameter

    
    '''

    y_list = [] # store coverged points
    info_dict = {'success' : False} # store algorithm infos

    if not isinstance(x0, np.ndarray):
        x0 = np.array([x0])

    if p0 is None:
        p0 = p_range[0]

    # converting complex function to real, due to a mix of real and complex varialbles
    if x0.dtype == 'complex':
        fun_real = lambda x, p : func_wrapper(fun, x, p)
        x0_real = complex_array_to_real(x0)
    else:
        fun_real = fun
        x0_real = x0

    res = Result_Obj(x0.dtype)
    x_size = len(x0_real)
    p_size = 1
    # creating parametric functions
    Fx = lambda p : lambda x : fun_real(x,p)
    Fp = lambda x : lambda p : fun_real(x,p)

    if jacx is None:
        JFx = lambda p : nd.Jacobian(Fx(p))
    elif callable(jacx):
        JFx = lambda p : lambda x : jac_wrapper(jacx,x,p).toarray() 
    else:
        raise('Jacobian not supported')

    if jacp is None:
        JFp = lambda x : nd.Jacobian(Fp(x))
    elif callable(jacp):
        JFp = lambda x : jac_wrapper(jacp,x,p).toarray() 
    else:
        raise('Jacobian not supported')

    #build jacobian aprox. where y = [x,p]
    hx = lambda y : JFx(y[-1])(y[:-1])
    hp = lambda y : JFp(y[:-1])(y[-1])
    fun_norm = lambda y : np.linalg.norm(fun_real(y[:-1],y[-1]))

    # building G, Gy, and R matrix for correction phase
    # see referece https://doi.org/10.1016/j.cma.2015.07.017
    Gy = lambda y, v : np.vstack((np.hstack((hx(y),hp(y).T)),v))
    G = lambda y, v : np.concatenate((fun_real(y[:-1],y[-1]),np.array([0.0])))
    R = lambda y, v : np.concatenate((np.hstack((hx(y),hp(y).T)).dot(v),np.array([0.0])))

    # find initial fixed point for continuation
    opt_obj = root(Fx(p0),x0=x0_real,method=root_method)
    x0_opt = opt_obj.x
    y0 = np.concatenate([x0_opt,np.array([p0])])
    #y_list.append(np.concatenate([real_array_to_complex(y0[:-1]),np.array([y0[-1]])]))
    res.append(y0)

    # finding tagent vector t:
    t0 = np.ones(x_size+p_size)
    b = np.zeros(x_size+p_size)
    b[-1] = 1
    v0 = np.linalg.solve(Gy(y0,t0),b)

    # find a predictor 
    y_pred = y0 + step*v0
    last_p = p0
    default_step = step
    for i in range(max_int):
        # corrector algotirhm
        if correction_method=='moore_penrose':
            y,v,error_norm, error_list, success = moore_penrose(fun_real,y_pred,v0,G,Gy,R,b,max_int=max_int_corr,tol=tol)

        elif correction_method=='matcont':
            y,v,error_norm, error_list, success = matcont(fun_real,y_pred,v0,G,Gy,R,b,max_int=max_int_corr,tol=tol)

        elif correction_method=='optimize_matcont':
            y,v,error_norm, error_list, success = optimize_matcont(fun_real,y_pred,v0,G,Gy,R,b,max_int=max_int_corr,tol=tol)

        elif  correction_method=='fixed_parameter':
            #hxp = lambda x, p : JFx(p)(x)
            hxp = None
            y,v,error_norm, error_list, success = fixed_parameter(fun_real,y_pred,v0,G,Gy,R,b,hx=hxp,max_int=max_int_corr,tol=tol)
        
        elif  correction_method=='fixed_direction':
            hxp = None
            y,v,error_norm, error_list, success = fixed_direction(fun_real,y_pred,v0,G,Gy,R,b,hx=hxp,max_int=max_int_corr,tol=tol)
        else: 
            raise ValueError('Corrector method not supported')
        p = y[-1] 
        dp = np.abs(p - last_p)

        if success and (dp<=max_dp):
            # find a new predictor 
            y_pred = y + step*v
            
            # update variables for next iteration
            y0 = y
            v0 = v
            last_p = p
            if print_mode:
                print('Iteration %i has converged , p = %3.5f' %(i,p))
            #y_list.append(np.concatenate([real_array_to_complex(y0[:-1]),np.array([y0[-1]])]))
            res.append(y0)
            step = default_step
            if p>=p_range[1] or p<=p_range[0] :
                print('Continuation algorithm has reached the limits of paramenter range')
                info_dict['success'] = True
                break

        else:
            # reducing the correction step
            step = 0.5*step
            y_pred = y0 + step*v0

    y_array = np.array(res.get_results()) 
    return y_array.T[0:-1],y_array.T[-1], info_dict 
        

class Result_Obj():

    def __init__(self, dtype=np.float):
        self.res = []
        self.dtype = dtype

    def append(self,y,dtype=None):
        if dtype is None:
            dtype = self.dtype

        if dtype==np.float or dtype==np.int:
            self.res.append(y)
        elif dtype==np.complex:
            self.res.append(np.concatenate([real_array_to_complex(y[:-1]),np.array([y[-1]])]))
        else:
            print('dtype = %s' %str(dtype))
            raise('dtype is not supported')

    def get_results(self):
        return self.res

def corrector(fun,y0,v0,G,Gy,R,b,max_int=10,tol=1.0E-6, correction_method='moore_penrose'):

    fun_norm = lambda y : np.linalg.norm(fun(y[:-1],y[-1]))
    y = y0 
    v = v0
    success = False
    error_list = []
    for i in range(max_int):
        error_norm = fun_norm(y)
        error_list.append(error_norm)
        if error_norm<tol:
            success = True
            break

        if correction_method == 'moore_penrose':
            Gy_eval = Gy(y,v)
            try:
                v = np.linalg.solve(Gy_eval ,b)
            except:
                print('computing pseudo inverse in paramenter = %f' %y[-1])
                G_inv = np.linalg.pinv(Gy_eval)
                v = G_inv.dot(b)
        else:

            Gy_eval = Gy(y,v)
            R_eval = R(y,v)
            delta_v = np.linalg.solve(Gy_eval,R_eval)
            v -= delta_v
        
        G_eval = G(y,v)
        try:
            delta_y = np.linalg.solve(Gy_eval,G_eval)
        except:
            delta_y = G_inv.dot(G_eval)

        y -= delta_y
        #y[-1] = y[-1].real

    return y,v,error_norm, error_list, success

def matcont(fun,y0,v0,G,Gy,R,b,max_int=10,tol=1.0E-6):

    fun_norm = lambda y : np.linalg.norm(fun(y[:-1],y[-1]))
    y = y0 
    v = v0
    success = False
    error_list = []
    for i in range(max_int):
        error_norm = fun_norm(y)
        error_list.append(error_norm)
        if error_norm<tol:
            success = True
            break

        # update y solution with previus tagent vector
        Gy_eval = Gy(y,v)
        G_eval = G(y,v)
        R_eval = R(y,v)
        lu_factor = linalg.lu_factor(Gy_eval)

        # update the solution y and tanget vector v
        delta_y = linalg.lu_solve(lu_factor,G_eval)
        delta_v = linalg.lu_solve(lu_factor,R_eval)
        y -= delta_y
        v -= delta_v
        
    return y,v,error_norm, error_list, success

def optimize_matcont(fun,y0,v0,G,Gy,R,b,max_int=10,tol=1.0E-6):

    fun_norm = lambda y : np.linalg.norm(fun(y[:-1],y[-1]))
    y = y0 
    v = v0
    n = y.shape[0]
    success = False
    error_list = []
    error_norm = fun_norm(y0)
    options_dict = {'maxiter': max_int}

    x_aug_0 = np.concatenate((y0,v0))
    R_aug = lambda x_aug : np.concatenate((G(x_aug[:n],x_aug[n:]),R(x_aug[:n],x_aug[n:])))
    jac = lambda x_aug : linalg.block_diag(*([Gy(x_aug[:n],x_aug[n:])]*2))
    
    opt_obj = root(R_aug,x_aug_0,method='lm',jac=jac, tol = tol, options=options_dict)
    if opt_obj.success:
        x_opt = opt_obj.x
        y,v = x_opt[:n], x_opt[n:]
        error_norm = fun_norm(y)
        success = True

    return y,v,error_norm, error_list, success

def moore_penrose(fun,y0,v0,G,Gy,R,b,max_int=10,tol=1.0E-6):

    fun_norm = lambda y : np.linalg.norm(fun(y[:-1],y[-1]))
    y = y0 
    v = v0
    success = False
    error_list = []
    for i in range(max_int):
        error_norm = fun_norm(y)
        error_list.append(error_norm)
        if error_norm<tol:
            success = True
            break

        # update y = [x,p]
        Gy_eval = Gy(y,v)
        G_eval = G(y,v)
        try:    
            delta_y = np.linalg.solve(Gy_eval ,G_eval)
            
        except:
            print('Using iterative solver = %f' %y[-1])
            delta_y, success = sparse.linalg.cg(Gy_eval, G_eval,y,tol=tol)

        y -= delta_y
        # update tangent vector at new y = [x,p]
        R_eval = R(y,v)
        Gy_eval = Gy(y,v)

        try:
            delta_v = np.linalg.solve(Gy_eval ,R_eval)
        except:
            print('Using iterative solver = %f' %y[-1])
            delta_v, sucess = sparse.linalg.cg(Gy_eval, R_eval,v,tol=tol)

        v -= delta_v 

    return y,v,error_norm, error_list, success

def fixed_parameter(fun,y0,v0,G,Gy,R,b,hx=None,max_int=20,tol=1.0E-6):

    fun_norm = lambda y : np.linalg.norm(fun(y[:-1],y[-1]))
    y = y0 
    v = v0
    success = False
    error_list = []

    p = y[-1]
    x0_real = y[:-1]
    options_dict = {'ftol': tol, 'maxiter': max_int}
    Fx = lambda x : fun_real(x,p)
    
    error_norm = fun_norm(y)
    if error_norm<tol:
        success = True
    
    else:
        #opt_obj = root(Fx(p0),x0=x0_real,method='lm',jac=hx, tol=tol)
        opt_obj = root(fun,x0=x0_real,args=p,method='lm',jac=hx, tol=tol, options=options_dict)
       
        if opt_obj.success:
            #update variables:
            y[:-1] = opt_obj.x
            error_norm  = fun_norm(y)
            # update new tangent vector
            Gy_eval = Gy(y,v)

            #delta_v = np.linalg.solve(Gy_eval,R(y,v))    
            #v =-delta_v
            v = np.linalg.solve(Gy_eval,b)    
            delta_v = v - v0
            norm_v = np.linalg.norm(v)
            if 0.5 <= norm_v <= 1.5:
                pass
            else:
                print('Warning! Tangent vector has norm = %e ' %norm_v)
                v /=norm_v
            
            print('delta p = %e' %delta_v[-1])

            success = True
        else:
            success = False

    return y,v,error_norm, error_list, success

def fixed_direction(fun,y0,v0,G,Gy,R,b,hx=None,max_int=20,tol=1.0E-6):

    fun_aug = lambda y : fun(y[:-1],y[-1])
    fun_norm = lambda y : np.linalg.norm(fun_aug(y))
    y = y0 
    v = v0
    success = False
    error_list = []
    options_dict = {'maxiter': max_int}
   
    
    error_norm = fun_norm(y)
    if error_norm<tol:
        success = True
    
    else:       
        # update new correction vector
        # update y = [x,p]
        Gy_eval = Gy(y0,v0)
        G_eval = G(y0,v0)
        d = np.linalg.solve(Gy_eval, G_eval)
        
        fprime = nd.Gradient(fun_norm)
        d = fprime(y0)
        #d /= np.linalg.norm(d)

        line = lambda alpha : fun_norm(y0+alpha*(-d))

        #line_tuple = optimize.line_search(fun_norm, fprime, y, d/norm_d )
        opt_obj = optimize.minimize(line,0,tol=tol,options=options_dict)

        if opt_obj.success:
            #update variables:
            y = y0+opt_obj.x*d
            error_norm  = fun_norm(y)
            # update new tangent vector
            Gy_eval = Gy(y,v0)
            R_eval = R(y,v0)
            delta_v  = np.linalg.solve(Gy_eval,R_eval)    
            v  -= delta_v
            norm_v = np.linalg.norm(v)
            if 0.5 <= norm_v <= 1.5:
                pass
            else:
                print('Warning! Tangent vector has norm = %e ' %norm_v)
                v /=norm_v
            
            print('delta p = %e' %delta_v[-1])

            success = True
        else:
            success = False

    return y,v,error_norm, error_list, success


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

    def test_complex_derivative(self):

        fun = lambda x : np.exp(x)
        dfun = complex_derivative(fun)

        val_list = [0,1,np.pi]

        for x in val_list:
            for y in val_list:
                z = x + 1J*y
                dval = dfun(z)        
                dtarget = np.exp(x)*(np.cos(y) + 1J*np.sin(y))
        
                np.testing.assert_array_almost_equal(dval, dtarget,  decimal=10 )

    def test_complex_jacobian(self):
        Z = self.Z
        f = self.f
        x_sol = np.linalg.solve(Z,f)
        
        R = lambda x : Z.dot(x) - f
       
        x0 = np.ones(2,dtype=np.complex)
        jfun = complex_jacobian(R)
        jval = jfun(x0)

        np.testing.assert_array_almost_equal(jval.flatten(), Z.flatten(),  decimal=10 )

    def test_complex_hessian(self):
        
        Z = self.Z
        f = self.f
        x_sol = np.linalg.solve(Z,f)
        
        R = lambda x : Z.dot(x) - f
        fun = lambda x : R(x).conj().T.dot(R(x))
        H_target = 2*Z.conj().T.dot(Z)

        x0 = np.ones(2,dtype=np.complex)
        jfun = complex_hessian(fun)
        jval = jfun(x0)

        np.testing.assert_array_almost_equal(jval.flatten(), H_target.flatten(),  decimal=10 )

    def test_scalar_root(self):

        r = lambda x : (x-1.0)*(x-10)*(x-100)
        x0 = np.array(0.0)
        opt_obj = root(r,x0)
        x_opt = opt_obj.x 

        np.testing.assert_array_almost_equal(np.array(1.0), x_opt ,  decimal=10 )

    def test_ellipse_continuation(self):

        xc = 4
        yc = 2
        r2 = 200
        A = np.array([[10.,-5.8],
                      [-5.8,5.]])

        v = lambda x, p : np.array([(p-xc),(x-yc)])
        paraboloid_vec = lambda x, p : np.array([v(x,p).T.dot(A).dot(v(x,p)) - r2])

        ye_list, pe_list, info_dict = continuation(paraboloid_vec,x0=3.0,p_range=(-15,15),p0=0.0,step=1)
        
        r = np.array(list(map(paraboloid_vec,ye_list[0], pe_list))).flatten()
        np.testing.assert_array_almost_equal( 0.0*r,  r ,  decimal=6 )

    def test_spiral(self):
        a1, b1 = 0, 0.5
        w = 2
        xs = lambda s : (a1 + b1*s)*np.cos(w*s)
        ys = lambda s : (a1 + b1*s)*np.sin(w*s)               
        spiral = lambda s : (xs(s),ys(s)) 
        spiral_res = lambda x, y, s : (x - (a1 + b1*s)*np.cos(w*s))**2 + (y - (a1 + b1*s)*np.sin(w*s) )**2 
        #spiral_res_vec = lambda x, s : spiral_res(x[0],x[1],s)

        spiral_res_vec = lambda x, s : np.array([(x[0] - (a1 + b1*s)*np.cos(w*s)), (x[1] - (a1 + b1*s)*np.sin(w*s))]) 

        x0=np.array([0.0,0.0])

        x_sol, p_sol, info_dict = continuation(spiral_res_vec,x0=x0,p_range=(-10.0,10.0),p0=0.0,max_dp=0.1)

        x_target = np.array(list(map(xs,p_sol)))
        y_target = np.array(list(map(ys,p_sol)))

        np.testing.assert_array_almost_equal(x_target, x_sol[0] ,  decimal=6 )
        np.testing.assert_array_almost_equal(y_target, x_sol[1] ,  decimal=6 )

    def test_3d_spiral(self):
        z = lambda p : p 
        r = lambda p : (p-8)**2 + 1
        x = lambda p : r(p) * np.sin(p)
        y = lambda p : r(p) * np.cos(p)
        spiral3d = lambda xvec, p : np.array([xvec[0] - x(p),xvec[1] - y(p),xvec[2] - z(p)])

        x0=np.array([0.0,0.0,0.0])
        x_sol, p_sol, indo_dict = continuation(spiral3d,x0=x0,p_range=(0,20.0),p0=0,max_int=1000,max_dp=2)

        x_target = np.array(list(map(x,p_sol)))
        y_target = np.array(list(map(y,p_sol)))
        z_target = np.array(list(map(z,p_sol)))

        np.testing.assert_array_almost_equal(x_target, x_sol[0] ,  decimal=6 )
        np.testing.assert_array_almost_equal(y_target, x_sol[1] ,  decimal=6 )
        np.testing.assert_array_almost_equal(z_target, x_sol[2] ,  decimal=6 )

    def test_duffing(self):
        nH = 2
        B = hbm_complex_bases(1,number_of_harm=nH,n_points=50,static=False,normalized=False)
        P = 0.1
        beta = 1.0
        M = 1.0
        K = 1.0
        C = 0.08

        fnl = lambda u : beta*(u**3)
        fl = linear_harmonic_force(P, f0 = 1, n_points=50, cos=True)
        fl_ = B.conj().T.dot(fl)
        fnl_ = lambda u_ : B.conj().T.dot(fnl(B.dot(u_).real)) - fl_
        Z = lambda w :create_Z_matrix(K,C,M,f0= w/(2.0*np.pi),nH=nH, static=False)
        #Z = lambda w: np.array([-(w)**2*M + 1J*(w)*C + K]) 
        R = lambda u_, w : Z(w).dot(u_) + fnl_(u_)

        x0 = np.array([0]*nH,dtype=np.complex)
        y_d, p_d, indo_dict = continuation(R,x0=x0,p_range=(0.01,3.5), p0=0.1, max_int=400, max_dp=0.2,step=0.2, max_int_corr=20, tol=1.0E-10)

        calc_error = np.array(list(map(R,y_d.T,p_d))).flatten()
        np.testing.assert_array_almost_equal(calc_error, 0.0*calc_error ,  decimal=9 )

    def test_2dof_duffing(self):
        nH = 2
        n_points = 200
        amplitude_dof_1 = 1.0
        amplitude_dof_2 = 0.0
        P = np.array([amplitude_dof_1, amplitude_dof_2], dtype = np.complex)
        beta = 1.0
        m1 = 1.0
        m2 = m1
        k1 = 1.0
        k2 = k1
        c1 = 0.05
        c2 = c1
        

        n_dofs = P.shape[0]

        q = hbm_complex_bases(1,number_of_harm=nH,n_points=n_points,static=False,normalized=False)

        K = np.array([[k1+k2, -k2],
              [-k2,k2+k1]])

        M = np.array([[m1,0.0],
                      [0.0,m2]])

        C = np.array([[c1+c2,-c2],
                      [-c2,c1+c2]])

        B_delta = np.array([[-1, 1],
                            [-1, 1]])

        H = np.array([[-1, 0],
                      [ 0, 1]])

        Ro = ReshapeOperator(n_dofs,n_points)

        
        I = np.eye(n_dofs)
        I_harm = np.eye(nH)
        Q = np.kron(I,q[:,0])
        for i in range(1,nH):
            Q = np.concatenate([Q,np.kron(I,q[:,i])])
        Q = Q.T
        Tc = H.dot(B_delta)
        P_aug = list(0*P)*nH
        P_aug[0:n_dofs] = list(P)
        P_aug = np.array(P_aug)
        fl = Q.dot(P_aug).real

        fl_ = Q.conj().T.dot(fl) # force in frequency domain
        fnl = lambda u : beta*(Tc.dot(u)**3)
        fnl_ = lambda u_ : Q.conj().T.dot(Ro.T.dot(fnl(Ro.dot(Q.dot(u_).real)))) - fl_
        Z = lambda w : create_Z_matrix(K,C,M,f0= w/(2.0*np.pi),nH=nH, static=False)
        R = lambda u_, w : Z(w).dot(u_) + fnl_(u_)

        x0 = np.array([0.0]*n_dofs*nH,dtype=np.complex)
        p0 = 0.01
        
        y_d, p_d, info_dict = continuation(R,x0=x0,p_range=(0.01,3.0), p0=p0, max_int=500, max_dp=0.05,step=0.05, max_int_corr=30, tol=1.0E-10)

        dif_amplitude =  P - fl_[:n_dofs]
        np.testing.assert_array_almost_equal(dif_amplitude, np.array([0.0]*n_dofs) ,  decimal=2 )

        calc_error = np.array(list(map(R,y_d.T,p_d))).flatten()
        np.testing.assert_array_almost_equal(calc_error, 0.0*calc_error ,  decimal=9 )
        

    def test_2dof_duffing_fixed_parameter_corrector(self):
        nH = 2
        n_points = 200
        amplitude_dof_1 = 1.0
        amplitude_dof_2 = 0.0
        P = np.array([amplitude_dof_1, amplitude_dof_2], dtype = np.complex)
        beta = 1.0
        m1 = 1.0
        m2 = m1
        k1 = 1.0
        k2 = k1
        c1 = 0.05
        c2 = c1
        

        n_dofs = P.shape[0]

        q = hbm_complex_bases(1,number_of_harm=nH,n_points=n_points,static=False,normalized=False)

        K = np.array([[k1+k2, -k2],
              [-k2,k2+k1]])

        M = np.array([[m1,0.0],
                      [0.0,m2]])

        C = np.array([[c1+c2,-c2],
                      [-c2,c1+c2]])

        B_delta = np.array([[-1, 1],
                            [-1, 1]])

        H = np.array([[-1, 0],
                      [ 0, 1]])

        Ro = ReshapeOperator(n_dofs,n_points)

        
        I = np.eye(n_dofs)
        I_harm = np.eye(nH)
        Q = np.kron(I,q[:,0])
        for i in range(1,nH):
            Q = np.concatenate([Q,np.kron(I,q[:,i])])
        Q = Q.T
        Tc = H.dot(B_delta)
        P_aug = list(0*P)*nH
        P_aug[0:n_dofs] = list(P)
        P_aug = np.array(P_aug)
        fl = Q.dot(P_aug).real

        fl_ = Q.conj().T.dot(fl) # force in frequency domain
        fnl = lambda u : beta*(Tc.dot(u)**3)
        fnl_ = lambda u_ : Q.conj().T.dot(Ro.T.dot(fnl(Ro.dot(Q.dot(u_).real)))) - fl_
        Z = lambda w : create_Z_matrix(K,C,M,f0= w/(2.0*np.pi),nH=nH, static=False)
        R = lambda u_, w : Z(w).dot(u_) + fnl_(u_)

        x0 = np.array([0.0]*n_dofs*nH,dtype=np.complex)
        p0 = 0.01
        
        y_d, p_d, info_dict = continuation(R,x0=x0,p_range=(0.01,3.0), p0=p0, max_int=500, correction_method='matcont',
                                           max_dp=0.05,step=0.05, max_int_corr=30, tol=1.0E-10)

        dif_amplitude =  P - fl_[:n_dofs]
        np.testing.assert_array_almost_equal(dif_amplitude, np.array([0.0]*n_dofs) ,  decimal=2 )

        calc_error = np.array(list(map(R,y_d.T,p_d))).flatten()
        np.testing.assert_array_almost_equal(calc_error, 0.0*calc_error ,  decimal=9 )


    def implicit_continuation(self):
        # case constants
        beta = 1.0
        m1 = 1.0
        m2 = m1
        k1 = 1.0
        k2 = k1
        c1 = 0.05
        c2 = c1

        # harmonic bases
        w0 = 0.2
        nH = 1
        n_points= 1000


        # system 1
        K1 = np.array([k1])

        M1 = np.array([m1])

        C1 = np.array([c1])

        q = hbm_complex_bases(1,number_of_harm=nH,n_points=n_points,static=False,normalized=False)
        n_dofs = 2
        I = np.eye(n_dofs)
        I_harm = np.eye(nH)
        #Q = np.kron(q ,[1]*nH)
        Q = np.kron(I,q[:,0])
        for i in range(1,nH):
            Q = np.concatenate([Q,np.kron(I,q[:,i])])
        Q = Q.T
        Ro = ReshapeOperator(n_dofs,n_points)

        Z1 = lambda w: -(w)**2*M1 + 1J*(w)*C1 + K1

        fl1_ = np.array([0.0],dtype= np.complex)

        fnl1_ = lambda u1_ : np.array([0.0],dtype= np.complex)

        B1 = np.array([1])

        R1 = lambda u1_,lm, w : (Z1(w).dot(u1_) - fl1_ + fnl1_(u1_) - B1.T.dot(lm))

        x0=np.array([0.0,0.0],dtype= np.complex)
        u1_implicit_numeric = lambda lm, w : optimize.root( lambda u1_ : R1(u1_, lm, w), x0=np.array([0.0],dtype= np.complex) ).x
        u1_implicit_analyic = lambda lm, w : (fl1_ + B1.T.dot(lm))/Z1(w)
        u1_implicit = u1_implicit_analyic
        du1dlm = lambda lm, w :  B1.T/Z1(w)
    
        # system 2

        K2 = np.array([[k2, -k2],
                      [-k2, k2]])

        M2 = np.array([[0.0,0.0],
                       [0.0,m2]])

        C2 = np.array([[c2,-c2],
                       [-c2,c2]])


        Z2 = lambda w: -(w)**2*M2 + 1J*(w)*C2 + K2


        fl2_ = np.array([0.0,1.0], dtype= np.complex)/np.diag(Q.conj().T.dot(Q).real)[0]

        Bc = np.array([[0, 0],
                       [0, 1]])

        fnl = lambda u : Bc.dot(beta*(u**3))
        #Jfnl = lambda u : 3.0*Bc.dot(beta*(u**2)*I)

        fnl2_ = lambda u_ : Q.conj().T.dot(Ro.T.dot(fnl(Ro.dot(Q.dot(u_).real))))
        #Jfnl2_ = lambda u_ : Q.conj().T.dot(Ro.T.dot(Jfnl(Ro.dot(Q.dot(u_).real))))
        Jfnl2_ = complex_jacobian(fnl2_)

        B2 = np.array([-1, 0])

        R2 = lambda u2_,lm, w : (Z2(w).dot(u2_) - fl2_ + fnl2_(u2_) - B2.T.dot(lm))
        dR2du_ = lambda w : lambda u2_ :Z2(w)+Jfnl2_(u2_)
        u2_implicit = lambda lm, w : root( lambda u2_ : R2(u2_, lm, w), x0=np.array([0.0,0.0],
                                                    dtype= np.complex), method='df-sane',tol=1.0E-12 ).x
        #u2_implicit = lambda lm, w : optimize.root( lambda u2_ : R2(u2_, lm, w), x0=np.array([0.0,0.0],
        #                                            dtype= np.complex),jac = dR2du_(w), method='broyden1',tol=1.0E-12 ).x
        du2dlm = lambda lm, w : np.linalg.solve(Z2(w)+Jfnl2_(u2_implicit(lm,w)), B2.T)


        # compatibility contraint
        R3 = lambda lm, w : B1.dot(u1_implicit(lm,w)) + B2.dot(u2_implicit(lm,w))
        dR3dlm = lambda w :lambda lm : B1.dot(du1dlm(lm[0],w)) + B2.dot(du2dlm(lm[0],w))
        R3lm = lambda w : lambda lm : R3(lm[0],w)

        R3_cont = lambda lm, w : B1.dot(u1_implicit(lm[0],w)) + B2.dot(u2_implicit(lm[0],w))
        lm0 = np.array([0.0]*nH,dtype=np.complex)
        y_d, p_d, info_dict =continuation(R3_cont, x0=lm0 ,p_range=(0.2,3.0), jacx=dR3dlm,
                                            max_int=2, max_dp=0.01,step=0.05, max_int_corr=30, tol=1.0E-8,root_method='df-sane')

if __name__ == '__main__':
    #main()
    
    test_obj = Test_root()
    test_obj.test_2dof_duffing()
    #test_obj.test_implicit_continuation()
    #test_obj.test_2dof_duffing_fixed_parameter_corrector()
