import numpy as np

def rad2hertz(frequency):
    ''' Converts frequency given in rad/s
    to Hertz
    
    paramenters:
        frequency : float or np.array
            frequency in rad/s
            
    return 
        frequency in Hertz
    '''
    return frequency/(2.0*np.pi)

def hertz2rad(frequency):
    ''' Converts frequency given in
    Hertz to rad/s
    
    paramenters:
        frequency : float or np.array
            frequency in Hertz
            
    return 
        frequency in rad/s
    '''
    return frequency*2.0*np.pi
    
def radsquared2hertz(frequency_squared):
    ''' Converts squared frequency given in (rad/s)^2
    to Hertz. This is usually a result of the Generalized 
    Eigen value problem:
    
    K x = \omega^2 M x 
    
    where omega is given in (rad/s)^2
    
    paramenters:
        frequency : float or np.array
            frequency in rad/s)^2
            
    return 
        frequency in Hertz
    '''
    return np.sqrt(frequency_squared)/(2.0*np.pi)