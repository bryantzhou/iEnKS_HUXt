# Important necessary packages
import numpy as np
from tools import rk

# External forcing
F = 8.0

# Define dynamics
# For all m, n, return s(x,n)=x[m+n]
def s(x,n):
    a = x.ndim-1
    return np.roll(x,-n,axis=a)

# dx[m]/dt = (x[m+1]-x[m-2])*x[m-1]-x[m]+F
def dxdt(x):
    return (s(x,1)-s(x,-2))*s(x,-1)-x+F

# Using rk4 to integrate the dynamics forward one step
def step(x, t, dt):
    return rk(lambda t,x: dxdt(x), x, np.nan, dt, order=4)




