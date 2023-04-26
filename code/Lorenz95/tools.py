import tqdm
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

# Runge-Kutta method to solve for ODE
def rk(f, x, t, dt, order):
    if order >= 1: k1 = f(t, x) * dt
    if order >= 2: k2 = f(t+dt/2, x+k1/2) * dt
    if order == 3: k3 = f(t+dt, x+k2*2-k1) *dt
    if order == 4: 
        k3 = f(t+dt/2, x+k2/2) * dt
        k4 = f(t+dt, x+k3) * dt
    
    if order == 1: return x + k1
    elif order == 2: return x + k2
    elif order == 3: return x + (k1 + 4*k2 + k3)/6
    elif order == 4: return x + (k1 + 2*(k2+k3) + k4)/6
    else: return print("Udefined Runge-Kutta order!")

# Progress bar
def progbar(iterable, desc=None, leave=1):
    "Prints a nice progress bar in the terminal"
    return tqdm.tqdm(iterable,desc=desc,leave=leave,smoothing=0.3,dynamic_ncols=True)

# Center the ensemble
def center(E):
    x_mean = np.mean(E,axis=0,keepdims=True)
    E_center = E - x_mean
    x_mean = x_mean.squeeze()
    return x_mean, E_center

# Make a square matrix by appending 0
def pad0(ss,N):
  out = np.zeros(N)
  out[:len(ss)] = ss
  return out

# Singular value decomposition
def svd0(A):
  """
  Compute the 
   - full    svd if nrows > ncols
   - reduced svd otherwise.
  As in Matlab: svd(A,0),
  except that the input and output are transposed, in keeping with DAPPER convention.
  It contrasts with scipy.linalg's svd(full_matrice=False) and Matlab's svd(A,'econ'),
  both of which always compute the reduced svd.
  For reduction down to rank, see tsvd() instead.
  """
  M,N = A.shape
  if M>N: return linalg.svd(A, full_matrices=True)
  else:   return linalg.svd(A, full_matrices=False)

# Calculate RMSE
def rmse(x,y):
   error_sq = np.square(x-y)
   error_mean_sq = np.mean(np.mean(error_sq,axis=0))
   x_rmse = np.sqrt(error_mean_sq)
   return x_rmse

# Live plots
def freshfig(num,figsize=None,*args,**kwargs):
  """Create/clear figure.
  - If the figure does not exist: create figure it.
    This allows for figure sizing -- even on Macs.
  - Otherwise: clear figure (we avoid closing/opening so as
    to keep (potentially manually set) figure pos and size.
  - The rest is the same as:
    >>> fig, ax = suplots()
  """
  fig = plt.figure(num=num,figsize=figsize)
  fig.clf()
  _, ax = plt.subplots(num=fig.number,*args,**kwargs)
  return fig, ax
