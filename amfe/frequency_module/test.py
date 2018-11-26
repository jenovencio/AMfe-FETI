from frequency import *

N = 1000
dt = 0.001
T = N*dt


f = Fourier(T,dt)
time_list = f.time_list
phi = f.compute_bases()
phi_sin =  phi.imag
phi_cos =  phi.real