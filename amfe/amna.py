# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 11:22:46 2017

@author: Guilherme Jenovencio


This module is the Applied Mechanical Numerival Algebra Library (AMNA)

This library intend to solve linear problems and non-linear problem filling the 
gaps of where standards libraries like numpy, scipy cannot handle efficiently,
for example, solving singular system

methods:
    cholsps: Cholesky decomposition for Symmetry Positive Definite Matrix

"""



import numpy as np
import scipy.sparse as sparse

def cholsps(A,tol=1e-10):    
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
    
    if isinstance(A,sparse.csr.csr_matrix):
        Atrace = np.trace(A.A)
    else:    
        Atrace = np.trace(A)
        
    tolA = tol*Atrace
    
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
            print('Matrix is not positive semi-definite')
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
    
            
    return U, idf, R   