# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 11:22:46 2017

@author: Guilherme Jenovencio


This module is the Applied Mechanical Numerical Algebra Library (AMNA)

This library intend to solve linear problems and non-linear problem filling the 
gaps of where standards libraries like numpy, scipy cannot handle efficiently,
for example, solving singular system

methods:
    cholsps: Cholesky decomposition for Symmetry Positive Definite Matrix

"""

import logging 
import numpy as np
from scipy.sparse import csc_matrix, issparse, linalg as sla
from scipy import linalg

def cholsps(A, tol=1.0e-8):    
    ''' This method return the upper traingular matrix of cholesky decomposition of A.
    This function works for positive semi-definite matrix. 
    This functions also return the null space of the matrix A.
    Input:
    
        A -> positive semi-definite matrix
        tol -> tolerance for small pivots
        
    Ouputs:
        U -> upper triangule of Cholesky decomposition
        idp -> list of non-zero pivot rows
        R -> matrix with bases of the null space
        
    '''
    [n,m] = np.shape(A)
    
    if n!=m:
        print('Matrix is not square')
        return
    
    
    L = np.zeros([n,n])
    #L = sparse.lil_matrix((n,n),dtype=float) 
    #A = sparse.csr_matrix(A)
    idp = [] # id of non-zero pivot columns
    idf = [] # id of zero pivot columns
    
    if issparse(A):
        Atrace = np.trace(A.A)
        A = A.todense()
    else:    
        Atrace = np.trace(A)
        
    tolA = tol*Atrace/n
    
    for i in range(n):
        Li = L[i,:]
        Lii = A[i,i] - np.dot(Li,Li)
        if Lii>tolA:
            L[i,i] = np.sqrt(Lii)
            idp.append(i)
        elif abs(Lii)<tolA:
            L[i,i] = 0.0
            idf.append(i)
    
        elif Lii<-tolA:
            logging.debug('Matrix is not positive semi-definite.' + \
                          'Given tolerance = %2.5e' %tol)
            return L, [], None
    
        for j in range(i+1,n):
            if L[i, i]>tolA:
                L[j, i] = (A[j, i] - np.dot(L[i,:],L[j,:]))/L[i, i]
            
            
    # finding the null space
    rank = len(idp)
    rank_null = n - rank
    
    U = L.T 
    R = None
    if rank_null>0:
        Im = np.eye(rank_null)
        
        # Applying permutation to get an echelon form
        
        PA = np.zeros([n,n])
        PA[:rank,:] = U[idp,:]
        PA[rank:,:] = U[idf,:]
        
        # creating block matrix
        A11 = np.zeros([rank,rank])
        A12 = np.zeros([rank,rank_null])
        
        A11 = PA[:rank,idp]
        A12 = PA[:rank,idf]
        
        
        R11 = np.zeros([rank,rank_null])
        R = np.zeros([n,rank_null])
        
        # backward substitution
        for i in range(rank_null):
            for j in range(rank-1,-1,-1):
                if j==rank-1:
                    R11[j,i] = -A12[j,i]/A11[j,j]
                else:
                    R11[j,i] = (-A12[j,i] - np.dot(R11[j+1:rank,i],A11[j,j+1:rank]) )/A11[j,j]
                
        # back to the original bases
        R[idf,:] = Im
        R[idp,:] = R11
        
        logging.debug('Null space size = %i' %len(idf))
            
    return U, idf, R   

def splusps(A,tol=1.0e-6):
    ''' This method return the upper traingular matrix based on superLU of A.
    This function works for positive semi-definite matrix. 
    This functions also return the null space of the matrix A.
    Input:
    
        A -> positive semi-definite matrix
        tol -> tolerance for small pivots
        
    Ouputs:
        U -> upper triangule of Cholesky decomposition
        idp -> list of non-zero pivot rows
        R -> matrix with bases of the null space
    '''
    [n,m] = np.shape(A)
    
    if n!=m:
        print('Matrix is not square')
        return
    
    
    idp = [] # id of non-zero pivot columns
    idf = [] # id of zero pivot columns
    
    if not isinstance(A,csc_matrix):  
        A = csc_matrix(A)

    lu = sla.splu(A)

    #L = lu.L
    U = lu.U
    Pr = csc_matrix((n, n))
    Pc = csc_matrix((n, n))
    Pc[np.arange(n), lu.perm_c] = 1
    Pr[lu.perm_r, np.arange(n)] = 1

    #L1 = (Pr.T * L).A
    #L2 = (U*Pc.T).A

    Utrace = np.trace(U.A)

    diag_U = np.diag(U.A)/Utrace

    idf = np.where(abs(diag_U)<tol)[0].tolist()
    
    R = calc_null_space_of_upper_trig_matrix(U,idf)
    R = Pc.A.dot(R)

    #for v in R.T:
    #    is_null_space(A,v, tol)

    return  lu, idf, R

def calc_null_space_of_upper_trig_matrix(U,idf=None):
    ''' This function computer the Null space of
    a Upper Triangule matrix which is can be a singular
    matrix

    argument
        U : np.matrix
            Upper triangular matrix
        idf: list
            index to fixed if the matrix is singular
    
    return
        R : np.matrix
            null space of U
    
    '''

    
    # finding the null space
    n,n = U.shape
    rank_null =len(idf)
    rank = n - rank_null
    
    U[np.ix_(idf),np.ix_(idf)] = 0

    # finding the null space
    idp = set(range(n))
    for fid in idf:
        idp.remove(fid)
    
    idp = list(idp)

    R = None
    if rank_null>0:
        Im = np.eye(rank_null)
        
        # Applying permutation to get an echelon form
        
        PA = np.zeros([n,n])
        PA[:rank,:] = U.A[idp,:]
        PA[rank:,:] = U.A[idf,:]
        
        # creating block matrix
        A11 = np.zeros([rank,rank])
        A12 = np.zeros([rank,rank_null])
        
        A11 = PA[:rank,idp]
        A12 = PA[:rank,idf]
        
        
        R11 = np.zeros([rank,rank_null])
        R = np.zeros([n,rank_null])
        
        # backward substitution
        for i in range(rank_null):
            for j in range(rank-1,-1,-1):
                if j==rank-1:
                    R11[j,i] = -A12[j,i]/A11[j,j]
                else:
                    R11[j,i] = (-A12[j,i] - np.dot(R11[j+1:rank,i],A11[j,j+1:rank]) )/A11[j,j]
                
        # back to the original bases
        R[idf,:] = Im
        R[idp,:] = R11

        return R

def pinv_and_null_space_svd(K,tol=1.0E-8):
    ''' calc pseudo inverve and
    null space using SVD technique
    '''

    if issparse(K):
        K = K.todense()
        
    V,val,U = np.linalg.svd(K)
        
    total_var = np.sum(val)
        
    norm_eigval = val/val[0]
    idx = [i for i,val in enumerate(norm_eigval) if val>tol]
    val = val[idx]
        
        
    invval = 1.0/val[idx]

    subV = V[:,idx]
        
    Kinv =  np.matmul( subV,np.matmul(np.diag(invval),subV.T))
        
    last_idx = idx[-1]
    R = np.array(V[:,last_idx+1:])

    return Kinv,R
        

def is_null_space(K,v, tol=1.0E-3):
    ''' this function checks if 
    a vector is belongs to the null space of
    K matrix

    argument:
    K : np.matrix
        matrix to check the kernel vector
    v : np.array
        vector to be tested
    tol : float
        tolerance for the null space

       '''

    norm_v = np.linalg.norm(v)
    r = K.dot(v)
    norm_r = np.linalg.norm(r)

    ratio = norm_r/norm_v

    if ratio<=tol:
        return True
    else:
        return False

class pinv_and_null_space():
    ''' This class intend to solve singular systems
    build the null space of matrix operator and also 
    build the inverse matrix operator
    
    Ku = f
    
    where K is singular, then the general solution is
    
    u = K_pinvf + alpha*R
    
    argument
        K : np.array
            matrix to be inverted
        tol : float
            float tolerance for buiding the null space
        
    return:
        K_pinv : object
        object containg the null space and the inverse operator
    '''
    solver_opt = 'splusps'
    list_of_solvers = ['cholsps','splusps','svd']
    pinv = None
    null_space = None
    free_index = []
    
    def compute(K,tol=1.0E-8,solver_opt=None):
        ''' This method computes the kernel and inverse operator
        '''
        
        if solver_opt is None:
            solver_opt = pinv_and_null_space.solver_opt
        
        if solver_opt=='splusps':
            lu, idf, R = splusps(K)
            lu.U[idf,:] = 0.0
            lu.U[:,idf] = 0.0
            lu.U[idf,idf] = 1.0
            K_pinv = lu.solve
            
            
        elif solver_opt=='cholsps':
            U,idf,R =cholsps(K)
            U[idf,:] = 0.0
            U[:,idf] = 0.0
            U[idf,idf] = 1.0
            K_pinv = lambda f : linalg.cho_solve((U,False),f) 
            
        elif solver_opt=='svd':
            K_inv, R = pinv_and_null_space_svd(K)
            K_pinv = np.array(K_inv).dot
            idf = []
        
        else:
            raise('Solver %s not implement. Check list_of_solvers.')
        
        pinv_and_null_space.pinv = K_pinv
        pinv_and_null_space.null_space = R
        pinv_and_null_space.free_index = idf
        
        return pinv_and_null_space
        
    def apply(f,alpha=np.array([])):
        ''' function to apply K_pinv
        and calculate a solution based on alpha
        by the default alpha is set to the zero vector
        
        argument  
            f : np.array
                right hand side of the equation 
            alpha : np.array
                combination of the kernel of K alpha*R
        '''
        K_pinv = pinv_and_null_space.pinv
        idf = pinv_and_null_space.free_index
        
        # f must be orthogonal to the null space R.T*f = 0 
        if idf:
            f[idf] = 0.0
        
        u_hat = K_pinv(f)
        
        if alpha:
            u_hat += pinv_and_null_space.calc_kernel_correction(alpha)
            
        return u_hat
        
    def calc_kernel_correction(alpha):
        ''' apply kernel correction to
        calculate another particular solution
        '''
        R = pinv_and_null_space.null_space
        u_corr = R.dot(alpha)
        return u_corr
