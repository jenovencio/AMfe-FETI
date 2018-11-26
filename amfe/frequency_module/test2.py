import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from scipy.misc import derivative
from scipy import optimize
import sys

sys.path.append(r'H:\TUM-PC\Dokumente\Python Scripts\notebooks')

from frequency_module.frequency import *


def create_Z_matrix(K,C,M,f0=1.0,nH=1,beta=None,complex_data= True):
    
    Z_list = []
    number_of_harm = nH 
    w0 = 2.0*np.pi*f0
    if complex_data:
        freq_list = w0*np.arange(-number_of_harm,number_of_harm+1,1)
    else:
        freq_list = w0*np.arange(0,number_of_harm+1,1)
    
    for w_i in freq_list:
        Z_i = K + 1J*w_i*C - w_i*w_i*M
        Z_list.append(Z_i)
    
    return la.block_diag(*Z_list)
    

  
    
def Fnl(u_,B,beta=1,power=3,nH =None, f0=None):  
    
    u = np.array(B.dot(u_)).flatten().real
    
    f = beta*u**power
    
    f_ = np.array(B.conj().T.dot(f)).flatten()
    
    return f_

def Fl(a, f0 = 1, n_points=100, cos=True, nH = None, beta=None):
    t0 = 0
    two_pi = 2.0*np.pi
    w0 = two_pi*f0
    
    
    if f0==0.0:
        return np.zeros(n_points)
    
    T = two_pi/w0
    x = np.linspace(t0,T,n_points)
    
    if cos:
        f = a*np.cos(w0*x)
    else:
        f = a*np.sin(w0*x)
    return f
    

k = 100
m = 1
c = 0.0
wn = np.sqrt(k/m)
nH = 1
f0 = 1
a = 1
n_points=100
wn = np.sqrt(k/m)
print('Natural frequency is w0 = %f' %wn)


B = complex_bases_for_MHBM(f0,number_of_harm=nH,n_points=n_points,complex_data=True)  

def LinearDyn(f0,a=1):
    Z = create_Z_matrix(k,c,m,f0=f0,nH=nH)     
    flinear = np.array(B.conj().T.dot((Fl(a,cos=False,f0=f0)))).flatten()
    u0_ = np.linalg.solve(Z,flinear)
    return u0_


f_list =  np.arange(0,10,1)

a0_list = []
s_list = []
col_id = 2
for fi in f_list:
    a = LinearDyn(fi)
    a0_list.append(np.abs(a[col_id]))
    s_list.append(np.angle(a[col_id]))