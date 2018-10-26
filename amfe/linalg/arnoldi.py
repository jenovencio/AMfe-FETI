# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 10:26:53 2018

@author: ge72tih
"""
import numpy as np
from scipy.sparse.linalg import LinearOperator
import scipy.sparse as sparse
import scipy.linalg as linalg

def arnoldi_iteration(A,b,nimp):
    """
    Input
    A: (nxn matrix)
    b: (initial vector)
    k: number of iterations
    
    Returns Q, h
    
    """
    m = A.shape[0] # Shape of the input matrix

    h = np.zeros((nimp+1, nimp))    # Creates a zero matrix of shape (n+1)x n
    Q = np.zeros((m, nimp+1))       # Creates a zero matrix of shape m x n

    q  = b/np.linalg.norm(b)        # Normalize the input vector
    q = np.reshape(q,(1,m))
    Q[:, 0] = q[0]                     # Adds q to the first column of Q
    
    for n in range(nimp):           
        v = q.dot(A)                # A*q_0
        for j in range(n+1):
            h[j, n] = float(v.dot(Q[:,j]))
            v = v - h[j,n]*Q[:,j]   

        h[n+1, n] = np.linalg.norm(v)
        q = v / h[n+1, n]
        Q[:, n+1] = q
    return Q[:,:nimp], h[:nimp,:nimp]


def inverse_arnoldi_iteration(A,b,nimp):
    """
    Input
    A: (nxn matrix)
    b: (initial vector)
    k: number of iterations
    
    Returns Q, h
    
    """
    m = A.shape[0] # Shape of the input matrix

    h = np.zeros((nimp+1, nimp))    # Creates a zero matrix of shape (n+1)x n
    Q = np.zeros((m, nimp+1))       # Creates a zero matrix of shape m x n

    q  = b/np.linalg.norm(b)        # Normalize the input vector
    q = np.reshape(q,(1,m))
    Q[:, 0] = q[0]                     # Adds q to the first column of Q
    
    for n in range(nimp):           
        v = np.linalg.solve(A, np.array(q).flatten())                # A^-1*q_0
        for j in range(n+1):
            h[j, n] = float(v.dot(Q[:,j]))
            v = v - h[j,n]*Q[:,j]   

        h[n+1, n] = np.linalg.norm(v)
        q = v / h[n+1, n]
        Q[:, n+1] = q
    return Q[:,:nimp], h[:nimp,:nimp]    
    
def general_inverse_arnoldi_iteration(A,M,b,nimp,tol=1.0e-8):
    """
    Input
    A: (nxn matrix)
    b: (initial vector)
    k: number of iterations
    
    Returns Q, h
    
    """
    A = np.array(A)
    M =np.array(M)
    b = np.array(b)
    
    m = A.shape[0] # Shape of the input matrix

    T = np.zeros((nimp+1, nimp))    # Creates a zero matrix of shape (n+1)x n
    Q = np.zeros((m, nimp+1))       # Creates a zero matrix of shape m x n
    Z = np.zeros((m, nimp+1))       # Creates a zero matrix of shape m x n
    
    z0 = b
    m0 = M.dot(b)
    beta0 = np.sqrt(z0.dot(m0))
    
    z  = z0/beta0       # Normalize the input vector
    q = np.reshape(m0,(1,m))
    z = np.reshape(z0,(1,m))
    Q[:, 0] = q[0]                     # Adds q to the first column of Q
    Z[:, 0] = z[0]                     # Adds m0 to the first column of Z
    
    for n in range(nimp):           
        z = np.linalg.solve(A, Q[:, n])           # A*q_0
        for j in range(n+1):
            T[j, n] = float(z.dot(Q[:,j]))
            z = z - T[j,n]*Z[:,j]   
            
        mj = M.dot(z)
        beta = np.sqrt(z.dot(mj))
        if beta>tol:
            T[n+1, n] = beta
        else:
            print('choose new v orthogonal to the previous iterations')
            T[n+1, n] = beta
            
        Q[:, n+1] = mj/T[n+1, n]
        Z[:, n+1] = z/T[n+1, n]
    return Z[:,:nimp], T[:nimp,:nimp]
    
    
    
    
    
    
def norm_A(x,A):
    ''' compute the A norm of a vector
    A must be a symmetric matrix
    
    parameters:
       x: np.array    
        vector to compute the A norm
       A : 2d np.array
       
    return 
    || x || _A
    
    '''
    x = np.array(x).flatten()
    A = np.array(A)
    return np.sqrt(x.dot(A.dot(x)))
    
    
def generalized_arnoldi_iteration(A,b,nimp,normOP=np.linalg.norm):
    """
    Input
    A: Linear operator
    b: (initial vector)
    normOP : a norm function
    nimp: number of iterations
    
    Returns Q, h
    
    """
    m = len(b) # Shape of the input matrix

    h = np.zeros((nimp+1, nimp))    # Creates a zero matrix of shape (n+1)x n
    Q = np.zeros((m, nimp+1))       # Creates a zero matrix of shape m x n

    q  = b/normOP(b)        # Normalize the input vector
    q = np.reshape(q,(1,m))
    Q[:, 0] = q[0]                     # Adds q to the first column of Q
    
    for n in range(nimp):           
        v = np.array(A.dot(q.T)).flatten()                # A*q_0
        for j in range(n+1):
            h[j, n] = float(v.dot(Q[:,j]))
            v = v - h[j,n]*Q[:,j]   

        h[n+1, n] = normOP(v)
        q = v / h[n+1, n]
        Q[:, n+1] = q
    return Q[:,:nimp], h[:nimp,:nimp]    
    


        
def lanczos(K,M=None,modes=10,b=None):
    ''' This algorithm uses general_inverse_arnoldi_iteration
    together with scipy eigen solver to compute the
    generalize eigen value problem
    
    parameters:
        K : scipy sparse matrix
            Left hand side Matrix for the generalized eigen problem
        M :  scipy sparse matrix
            Right hand side Matrix for the generalized eigen problem
        b : np.array
            arbitraty vector to initialize Arnold algorithm
        modes : int
            number of modes to compute
            
    output 
        eigval_TD : np.array
            eigen values of the generalized problem
            
        eigvec_TD : matrix
            eigen vector of the generalized problem
        '''
    ndof = K.shape[0]
    if M is None:
        M = sparse.eye(ndof)
    
    sys = LinearSys(K,M)
    Dop = LinearOperator((ndof,ndof), matvec=sys.solve)    
    
    if b is None:
        b = np.random.rand(ndof)
        
    DQ, DT = generalized_arnoldi_iteration(Dop,b,modes)
    eigval_TD, eigvec_TD = linalg.eig(DT)
    eigvec_TD = DQ.dot(eigvec_TD)
    
    return 1.0/eigval_TD, eigvec_TD
    
    
def is_eigvec(A,vec,M=None,tol=1.0e-6,show=True):
    ''' check is a vec is a eigenvector of A or
    a eigenvector of Ax = lambdaMx
    
    paramters:
        A : np.array
            right hand side matrix of the eigenvalue problem
        vec : np.array
            eigenvector to be tested
        M : np.array
            left hand side of the generalized eigenvalue problem
        tol : float
            tolerante in the norm difference between vector 
        show : Boolen
            print norm on the screen
            
    returns:
        Boolean
            True : if is vec is eigenvector
            Flase : if s vec is not eigenvector
    '''
    
    ndof = len(vec)
    if M is None:
        M = sparse.eye(ndof)
    
    vec1 = np.array(M.dot(vec)).flatten()
    vec2 = np.array(A.dot(vec)).flatten()
    
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    unit_vec1 = vec1/norm_vec1
    unit_vec2 = vec2/norm_vec2

    norm_dif = np.linalg.norm(unit_vec1 - unit_vec2)
    if show:
        print('The L2 norm of the vectors ||(v2/||v2|| - v1/|| v1 ||)|| where A*v=scale*M*v , v1= scale * v2 is %f' %norm_dif)
    if norm_dif>tol:
        return False
    else:
        return True    
        
def nullspace(B, tol= 10.e-10):
    ''' Compute the null space of A 
    where A is a complex matrix
    and it has dimention mxn where m<n
    
    parameters
        A : np.matrix
            matrix to compute the null space
    return
        R : np.matrix
            a orthogonal bases of the null space
    
    '''
    
    try:
        BTB = B.conj().T.dot(B).todense()
    except:
        BTB = B.conj().T.dot(B)
        
    w, v_null = linalg.eig(BTB)

    w_real = w.real
    w_imag = w.imag
    nnz = (w_real >= tol).sum()
    R = v_null[:,nnz:]
    return R        
    
def is_moore_pensore_inverse(A_inv, A,tol=1.e-12):
    max_elem = np.max(abs(A - A.dot(A_inv.dot(A))))
    if max_elem<tol:
        return True
    else:
        return False
        
def power_iteration(A, b =None, max_nint=100,tol=1e-10):
    ''' Computer a power iteration for a matrix A or a Linear
    Operator
        parameters
            A
            b
            max_int
            tol
    returns
        b_k : np.array
            the biggest eigenvector of A
    '''
   
    if b is None:
        b_k = np.random.rand(A.shape[1])
    else:
        b_k = b
        
    for _ in range(max_nint):
        # calculate the matrix-by-vector product Ab
        b_k1 = A.dot(b_k)

        # calculate the norm
        b_k1_norm = np.linalg.norm(b_k1)
        
        if np.linalg.norm(b_k1 - b_k)<tol:
            b_k = b_k1 / b_k1_norm
            break
        
        # re normalize the vector
        b_k = b_k1 / b_k1_norm

    return b_k
    
class LinearSys():
    def __init__(self,A,M):
        self.A = A
        self.M = M
        
    def solve(self,b):
        A = self.A
        M = self.M
        b = np.array(b)
        b_prime = np.array(M.dot(b)).flatten()
        return sparse.linalg.spsolve(A,b_prime)
    
    def normM(self,b):
        M = self.M
        b_prime = np.array(M.dot(b)).flatten()
        return b.dot(b_prime)    
        
    def getLinearOperator(self):
        ndof = self.A.shape[0]
        return LinearOperator((ndof,ndof), matvec=self.solve)  
        
class ProjLinearSys():      
    def __init__(self,A,M,P):
        self.A = A
        self.M = M
        self.P = P
        
    def solve(self,b):
        A = self.A
        M = self.M
        P = self.P
        b = np.array(b)
        b_prime = np.array(P.dot(M.dot(b))).flatten()
        return np.array(P.conj().T.dot(sparse.linalg.spsolve(A,b_prime)))
    
    def normM(self,b):
        M = self.M
        b_prime = np.array(M.dot(b)).flatten()
        return b.dot(b_prime)    
        
    def getLinearOperator(self):
        ndof = self.A.shape[0]
        return LinearOperator((ndof,ndof), matvec=self.solve)  
    
def compute_modes(K,M,num_of_modes=10,which='SM'):

    eigval, V = sparse.linalg.eigsh(K, k=num_of_modes, M=M,which=which)
    sort_id = np.argsort(eigval)

    eigval= eigval[sort_id]
    V = V[:,sort_id]
    return eigval,V