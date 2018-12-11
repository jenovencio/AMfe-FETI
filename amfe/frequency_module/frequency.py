import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from scipy import sparse

from unittest import TestCase, main
from numpy.testing import assert_array_equal

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

from operators.operators import HBMOperator, ReshapeOperator     

def harmonic_bases(x,freq=[1]):
    ''' create a harmonic bases [1, sin(w_i*x), cos(w_i*x)]
    given a frequency list
    
    arguments :
        freq : list
        
    return 
        phi : np.array
            bases of a Fourier Expansion 
            the size of the bases is 1 + 2*n_harm
            where n_harm is the size of freq list
    
    '''
    two_pi = 2.0*np.pi
    n_harm = len(freq)
    n = len(x)
    
    #phi_static = np.zeros([n])
    #phi_static[:] = 1.0
    
    
    phi_dynamic = np.zeros([n,n_harm],dtype=complex)
    
    for k,omega in enumerate(freq): 
        if freq[k]!=0.0:
            a = 1.0
        else:
            a = 0.5
        
        for i,dt in enumerate(x):     
            phi_dynamic[i,k] += a*np.cos(two_pi*freq[k]*dt) + a*np.sin(two_pi*freq[k]*dt)*1J
        
    return np.matrix(phi_dynamic)


def complex_bases(x,freq=[1]):
    ''' create a harmonic bases [1, sin(w_i*x), cos(w_i*x)]
    given a frequency list
    
    arguments :
        freq : list
        
    return 
        phi : np.array
            bases of a Fourier Expansion 
            the size of the bases is 1 + 2*n_harm
            where n_harm is the size of freq list
    
    '''
    two_pi = 2.0*np.pi
    n_harm = len(freq)
    n = len(x)
    
    #phi_static = np.zeros([n])
    #phi_static[:] = 1.0
    
    freq_list = np.concatenate((-1.0*np.array(freq),np.array([0]),np.array(freq)))
    
    phi_dynamic = np.zeros([n,2*n_harm+1],dtype=complex)
    
    for k,omega in enumerate(freq_list): 
        for i,dt in enumerate(x):     
            phi_dynamic[i,k] += np.exp(1J*two_pi*omega*dt) 
            
    return np.matrix(phi_dynamic)    
    
    
def complex_bases_for_MHBM(f0,number_of_harm=1,n_points=100,complex_data=False):
    ''' create a harmonic bases [1, exp(jw0t), exp(j2w0t), ...., exp(j(number_of_harm)w0t)]
    
    
        
    return 
        phi : np.array
            bases of a Fourier Expansion 
            the size of the bases is 1 + 2*n_harm
            where n_harm is the size of freq list
    
    '''
    t0 = 0
    two_pi = 2.0*np.pi
    w0 = two_pi*f0
    T = two_pi/w0
    
    x = np.linspace(t0,T,n_points)
    if complex_data:
        freq_list = w0*np.arange(-number_of_harm,number_of_harm+1,1)
        mult = 1.0
    else:
        freq_list = w0*f0*np.arange(0,number_of_harm+1,1)
        mult =np.sqrt(2.0)
    
    nH = len(freq_list)
    phi_dynamic = np.zeros([n_points,nH],dtype=np.complex)
    
    for k,omega in enumerate(freq_list): 
        for i,dt in enumerate(x):     
            phi_dynamic[i,k] += np.exp(1J*omega*dt) 
            
    return mult*np.array(phi_dynamic)/np.sqrt(n_points)
    
    
def cos_bases_for_MHBM(f0,number_of_harm=1,n_points=100, static=False):
    ''' create a harmonic bases [1, exp(jw0t), exp(j2w0t), ...., exp(j(number_of_harm)w0t)]
    
    
        
    return 
        phi : np.array
            bases of a Fourier Expansion 
            the size of the bases is 1 + 2*n_harm
            where n_harm is the size of freq list
    
    '''
    t0 = 0
    two_pi = 2.0*np.pi
    w0 = two_pi*f0
    #T = two_pi/w0
    T = 1.0/f0
    x = np.linspace(t0,T,n_points)
    
    if static:
        freq_list = w0*np.arange(0,number_of_harm+1,1)
        mult =np.sqrt(2.0)
    else:
        freq_list = w0*np.arange(1,number_of_harm+1,1)
        mult =np.sqrt(2.0)
    
    nH = len(freq_list)
    phi_dynamic = np.zeros([n_points,nH],dtype=np.float)
    
    for k,omega in enumerate(freq_list): 
        for i,dt in enumerate(x):     
            phi_dynamic[i,k] += np.cos(omega*dt) 
            
    return mult*np.array(phi_dynamic)/np.sqrt(n_points)
    
class Fourier():


    def __init__(self,T,dt=1.0):
        ''' initialize fourier class with time windows
        and time increament
        
        time_list and freq_list will be created
        
        '''
        self.freq_list, self.time_list = create_freq_list(T,dt,get_time=True)
       
        self.n = len(self.time_list)
        self.n_harm = len(self.freq_list)
    
        self.fourier_coef = []
        self.bases = []
        
    def compute_bases(self,ortho=False):
        ''' This function creates the Fourier bases

        phi = [1 + 0j, cos(w1t) + jsin(w1t), ..., cos(wNt) + jsin(wNt)]

         parameters:
            ortho: Boolean
                if True return a orthonormal bases for Fourier Transform

        return a matrix with the Discrete Fourier Bases
        
        '''
        
        self.ortho = ortho
        freq = self.freq_list
        n = self.n
        n_harm = self.n_harm
        self.bases = np.zeros([n,n_harm],dtype=complex)
        
        two_pi = 2.0*np.pi
        for k in range(len(freq)):
            for i,dt in enumerate(self.time_list):     
                self.bases[i,k] += np.cos(two_pi*freq[k]*dt) + np.sin(two_pi*freq[k]*dt)*1J
        
        if ortho:
            self.bases = self.bases/np.sqrt(n)

        return self.bases
    
    def fourier_transform(self,signal,ortho=False):
        
        if self.bases==[]:
            self.compute_bases(ortho)
        
        if ortho!=self.ortho:
            self.compute_bases(ortho)

        self.fourier_coef = self.bases.conj().T.dot(signal)
        #self.fourier_coef = np.linalg.solve(self.bases.conj().T.dot(self.bases), self.bases.conj().T.dot(signal))
        
        return self.fourier_coef
        
    def get_derivative(self,order=1):
        ''' This functions get the derivative of the Fourier bases
        
        paramenters:
            order : int
                order the derivative. Default = 1
        
        return 
            derivative : np.array
                derivative of the Fourier bases
        '''

        if self.bases==[]:
            self.compute_bases()

        wn = np.diag(1J*2.0*np.pi*np.array(self.freq_list))**order

        return self.bases.dot(wn)

    def get_amplitutes(self):

        if self.ortho:
            amplitute = 2*np.sqrt(self.n)*np.abs(self.fourier_coef)/self.n
        else:
            amplitute = 2*abs(self.fourier_coef)/self.n

        return amplitute
    
    def get_shift(self):
        shift = np.angle(self.fourier_coef)
        return shift

    def inv_fourier_transform(self,a_coef,ortho=False):
        ''' Given a fourier coef build the time signal

        paramenters
            a_coef : complex np.array
                Fourier coeficients

        return time signal

        ''' 
        if not(self.bases):
            self.compute_bases(ortho)
            
        two = 2.0
        if self.ortho:
            s = two*self.bases.dot(a_coef).real
        else:
            s = two*self.bases.dot(a_coef).real/(self.n)

        return s

    def get_real_bases_matrix(self):
        ''' fourier bases are build with complex number
        
        phi = phi_real + j*phi_imag
        
        this function creates a real phi bases which has the form
        
        phi = [phi_real   -phi_imag
               phi_imag    phi_real]
        
        '''
        row1 = np.hstack((self.bases.real,-self.bases.imag))
        row2 = np.hstack((self.bases.imag, self.bases.real))
        self.real_bases = np.vstack((row1,row2))
        return self.real_bases
    
    def get_real_fourier_coef(self):
        ''' Forier coef are complex numbers such as
        
        f_coef = a + jb
        
        this function creates a array containing 
        a array with [a, b]
        
        '''
            
        return np.hstack((self.fourier_coef.real, self.fourier_coef.imag))        
        
            
def create_freq_list(T,dt=1,get_time=False):
    ''' This fucntion creates a list of frequency based on
    a time window (0,T) and the time increament dt

    parameters:
        T : float
            end of the time windown
        dt : float
            time increament
        
        get_time : Boolean
            if True returns freq_list and time_list
            if False returns freq_list

    returns
        freq_list
            list with fourier transform frequencies
        time_list
            list of the discrite time points
    '''
    
    time_list = np.arange(0.0,T+dt,dt)
    freq_list = []
    
    if time_list[-1]!=T:
        time_list = time_list[:-1]
        
    N = len(time_list)
    #delta_f = 1.0/(2.0*T)

    if N%2==0:
        delta_f = 1.0/(2.0*T)
    else:
        delta_f = 1.0/(2.0*(T+dt))

    #if N%2==0:
    #    freq_num = int(N/2)
    #else:
    #    freq_num = int((N-1)/2)
    
    #for i in range(freq_num):
    #    freq_list.append(i*delta_f)
        
    freq_list = np.fft.rfftfreq(N,d=dt)

    if get_time:
        return freq_list,time_list
    else:    
        return freq_list

        
def create_Z_matrix(K,C,M,f0=1.0,nH=1, static=True, complex_data= False):
    
    #Z_list = []
    number_of_harm = nH 
    freq_list = hbm_freq(f0,number_of_harm=nH,complex_data=complex_data,static=static)
    num_of_freq = len(freq_list)
    
    # Z = K (x) I + wj (x) C - w2 (x) M
    if sparse.issparse(K):
        Z = sparse.kron(np.diag([1]*num_of_freq),K) + \
        sparse.kron(1J*np.diag(freq_list),C) - \
        sparse.kron(np.diag(freq_list)**2,M)
    else:
        Z = np.kron(np.diag([1]*num_of_freq),K) + \
        np.kron(1J*np.diag(freq_list),C) - \
        np.kron(np.diag(freq_list)**2,M)
    
    #for w_i in freq_list:
    #    Z_i = K + 1J*w_i*C - w_i*w_i*M
    #    Z_list.append(Z_i)
    #return sparse.block_diag(Z_list).tocsc()
    return Z
    
    
def hbm_freq(f0,number_of_harm=1,static=False,complex_data=False):
    '''
    This function creates a list of frequencies [rad/s] given the primary frequency
    and the number of requires Harmonics. 
    The user can select to create a 0 frequency related to the static mode by turning
    the static Boolean variable to True. Also, if the displacement in time in complex,
    the user can create negative frequencies to represent complex data.
    
    Parameters:
    ----------
        f0 : float
            primary frequency for the harmonic balance
        number_of_harm: int, default=1
            number of harmonic to be considered
        
        static : Boolean, default=False
            create a 0 frequency for the static mode representation
        complex_data : Boolean, default=False
            create negative frequencies to represent complex data.
            This function only work with static mode is True.
            
    Returns
    -------
        freq_list : list
            list of frequencies in [rad/s]
    
    '''
    two_pi = 2.0*np.pi
    w0 = two_pi*f0
    T = two_pi/w0

    if complex_data:
        freq_list = w0*np.arange(-number_of_harm,number_of_harm+1,1)
    else:
        if static:
            freq_list = w0*np.arange(0,number_of_harm+1,1)
        else:
            freq_list = w0*np.arange(1,number_of_harm+1,1)
    
    return freq_list
    
    
def hbm_complex_bases(f0,number_of_harm=1,n_points=100,static=True,complex_data=False,normalized=True):
    ''' create a harmonic bases [1, exp(jw0t), exp(j2w0t), ...., exp(j(number_of_harm)w0t)]
    
     Parameters:
    ----------
        f0 : float
            primary frequency for the harmonic balance
        number_of_harm: int, default=1
            number of harmonic to be considered
        n_points: int, default=100
            number of points to the harmonic bases
        static : Boolean, default=False
            create a 0 frequency for the static mode representation
        complex_data : Boolean, default=False
            create negative frequencies to represent complex data.
            This function only work with static mode is True.
        normalized : Boolean, default=False
            Define if bases are orthonormal or not
            
    Returns
    -------
        phi : np.array
            bases of a Fourier Expansion 
            the size of the bases is 1 + 2*n_harm
            where n_harm is the size of freq list
    
    '''
    t0 = 0
    two_pi = 2.0*np.pi
    w0 = two_pi*f0
    T = two_pi/w0
    
    x = np.linspace(t0,T,n_points)
    freq_list = hbm_freq(f0,number_of_harm=number_of_harm,complex_data=complex_data,static=static)
    
    if normalized:
        mult = 1.0
        
    else:
        mult =np.sqrt(2.0)
        
    
    nH = len(freq_list)
    phi_dynamic = np.zeros([n_points,nH],dtype=np.complex)
    
    for k,omega in enumerate(freq_list): 
        for i,dt in enumerate(x):     
            phi_dynamic[i,k] += np.exp(1J*omega*dt) 
            
    return mult*np.array(phi_dynamic)/np.sqrt(n_points)    
    

def assemble_hbm_operator(ndofs,f0=1,number_of_harm=1,n_points=100,static=True,complex_data=False,normalized=True):
    ''' create a harmonic bases [1, exp(jw0t), exp(j2w0t), ...., exp(j(number_of_harm)w0t)]
    
     Parameters:
    ----------
        ndofs : int
            number of dofs for the multidimensional bases
        
        f0 : float : default = 1.0
            primary frequency for the harmonic balance
        number_of_harm: int, default=1
            number of harmonic to be considered
        n_points: int, default=100
            number of points to the harmonic bases
        static : Boolean, default=False
            create a 0 frequency for the static mode representation
        complex_data : Boolean, default=False
            create negative frequencies to represent complex data.
            This function only work with static mode is True.
        normalized : Boolean, default=False
            Define if bases are orthonormal or not
    Returns
    -------
        Q : HBMOperator
            Truncated Fourier Multidimensional operator 
            the size of the bases is 1 + 2*n_harm
            where n_harm is the size of freq list
    
    '''
    
    q = hbm_complex_bases(f0=f0,number_of_harm=number_of_harm,
                          n_points=n_points,static=static,complex_data=complex_data,normalized=normalized)

    Q = HBMOperator(ndofs,q)

    return Q


 
def linear_harmonic_force(a, f0 = 1, n_points=100, cos=True):
    t0 = 0
    two_pi = 2.0*np.pi
    w0 = two_pi*f0
    
    
    if f0==0.0:
        return a*np.ones(n_points, dtype = np.complex)
    
    #T = two_pi/w0
    T = 1.0/f0
    
    x = np.linspace(t0,T,n_points)
    
    if cos:
        f = a*np.cos(w0*x)
    else:
        f = a*np.sin(w0*x)
    return f    


def duffing_force_in_freq(B,beta=1,power=3):  
    ''' This function returns the duffing force in frequency
    
    paramenters:
        B : np.array
            matrix with Discrite Forier Bases
        beta : float
            Duffing stiffness paramenters
        power : float
        Duffing's power, usually 3
        
        returns:
            lambda function based on amplitutes
    '''

    u_in_time = lambda u_ : np.array(B.dot(u_)).flatten().real
    f_ = lambda u_ :  B.conj().T.dot(beta*(u_in_time(u_)**power))
    
    return f_    



class  Test_Operators(TestCase):
    def setUp(self):
        pass

    def test_build_hbm_operator(self):

        amplitude_dof_1 = 5.0
        amplitude_dof_2 = 0.0
        P = np.array([amplitude_dof_1, amplitude_dof_2], dtype = np.complex)
        beta = 5.0
        m1 = 1.0
        m2 = m1
        k1 = 1.0
        k2 = k1
        c1 = 0.05
        c2 = c1

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


        ndofs = 2
        nH = 5
        n_points = 2000

        q = hbm_complex_bases(1,number_of_harm=nH,n_points=n_points,static=False,normalized=False)
        Ro = ReshapeOperator(ndofs,n_points)
        Qp = assemble_hbm_operator(ndofs,f0=1,number_of_harm=nH ,n_points=n_points,static=False,complex_data=False,normalized=False)

        I = np.eye(ndofs)
        I_harm = np.eye(nH)
        Q = np.kron(I,q[:,0])
        for i in range(1,nH):
            Q = np.concatenate([Q,np.kron(I,q[:,i])])
        Q = Q.T
        Tc = H.dot(B_delta)

        self.assertAlmostEqual(abs(Qp.Q.toarray() - Q).max(), 0.0, places=12)

        P_aug = list(0*P)*nH
        P_aug[0:ndofs] = list(P)
        P_aug = np.array(P_aug)
        fl = Q.dot(P_aug).real


        

        fnl = lambda u : beta*(Tc.dot(u)**3)
        
        fnl_ = lambda u_ : Q.conj().T.dot(Ro.T.dot(fnl(Ro.dot(Q.dot(u_).real)))) 
        new_fnl_ = lambda u_ : Qp.H.dot(fnl(Qp.dot(u_))) 

        u_ = 100.0*np.random.rand(ndofs*nH*2)
        u_.dtype = np.complex

        u_desired = Ro.dot(Q.dot(u_).real)
        u_actual = Qp.dot(u_)

        np.testing.assert_array_almost_equal(u_actual, u_desired ,  decimal=12 )

        fnl_desired = fnl_(u_)
        fnl_actual = new_fnl_(u_)

        np.testing.assert_array_almost_equal(fnl_actual, fnl_desired ,  decimal=10)


if __name__ == '__main__':
    main()    
    #test_obj = Test_Operators()
    #test_obj.test_build_hbm_operator()