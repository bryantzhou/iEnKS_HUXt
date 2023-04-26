# Import necessary packages
import numpy as np
import math
from tools import *
import matplotlib.pyplot as plt
import datetime
import astropy.units as u
import os

import huxt as H
import huxt_analysis as HA
import huxt_inputs as Hin

#####################################################
# Model FACT:
# v_grid = [time, radial, longitudinal]
# radial grids = 141
# longitudinal grids = 128
#####################################################

#####################################################
# Experiment setup:
# This is an initial attempt in applyting the iEnKS DA method to the space weather HUXt model.
# Here, no CME effect is included in the model, and "artificial" observation is made up from 
# perturbing the true states.
# DA updates the inner boundary states by assimilating observations downstream.
# In this twin experiment, a truth trajectory and a prior trajectory are generated. They come from
# different bounary conditions, which have the same mean but with different perturbation.
# The initial ensemble is from the boundary condition of the prior, and the goal is to assimilate
# observations to update the boundary condition. 
# The prior trajectory and the trajectory generated from the "posterior" boundary condition are 
# compared to the truth trajectory.
#####################################################

# Define parameters
simT = 10*u.day
obsT = 4 # =4*dt, dt~520s, it is roughly half an hour
lon = 0.0 # longitudianl point for 1-D plot
M = 128 # state at inner boundary
N = 20 # ensemble number
eps = 1e-4 
e = 1e-3 # stopping criteria for Gauss-Newton
j_max = 50 # max iteration of Gauss-Newton
infl = 1.02 # inflation

# Boundary condition 
v_boundary_prior = (np.ones(M) * 400 + np.random.normal(0,40,M)) * (u.km/u.s)
v_boundary_true = (np.ones(M) * 400 + np.random.normal(0,40,M)) * (u.km/u.s)

########################################################################
# Twin Experiment Setup (simulate two trajectores and take observations)
########################################################################
# Build two models, one for prior, the other for truth
model_prior = H.HUXt(v_boundary=v_boundary_prior, cr_num = 2000, lon_out=lon*u.deg, simtime=simT, dt_scale=obsT)
model_true = H.HUXt(v_boundary=v_boundary_true, cr_num = 2000, lon_out=lon*u.deg, simtime=simT, dt_scale=obsT)
print(model_true.v_grid.shape)

# Solve the model
cme_list = []
model_prior.solve(cme_list)
model_true.solve(cme_list)

# Extract data
x_pri = model_prior.v_grid[:,:,int(lon)].value # first coordinate is time, second is wind speed
xTrue = model_true.v_grid[:,:,int(lon)].value

# Take observations
model_obs = H.HUXt(v_boundary=v_boundary_true, cr_num = 2000, latitude = 0*u.deg, simtime=simT, dt_scale=obsT)
cme_list = []
model_obs.solve(cme_list)
t_out = model_obs.time_out.value
r = model_obs.r.value
y = np.zeros((len(t_out),len(r)))
L = len(t_out) # DAW length
beta = np.ones(L)

for i in progbar(np.arange(L),'Take Observations'): # calculate the observation error covariance, following Lang et al. paper
    R_kk = (0.1*np.mean(model_obs.v_grid[i,:,:].value,axis=1))**2 # std deviation is 10% of the mean prior solar wind speed at the obervation radius
    R_o = np.diag(R_kk)   
    y[i,:] = xTrue[i,:] + np.random.multivariate_normal(np.zeros(len(r)),R_o,1)

################################################################################
# Data Assimilation
################################################################################
# Create ensemble
E = np.random.multivariate_normal(v_boundary_prior,np.eye(M),N) 
E_1 = np.zeros((20,128))

# Initialize Gauss-Newton iteration
ens_mean, ens_center = center(E) # respectively: ensemble mean, anomaly matrix
w = np.zeros(N) # initialize control vector

# Iteration
for j in progbar(np.arange(j_max),'iEnKS Progress:'):
    x = ens_mean + w@ens_center # current solution of the state in the ensemble space: x = x_mean + A*w
    E = x + eps*ens_center # reconstruct the ensemble
    grad_incre = 0
    Hessian_incre = 0

    for i in progbar(np.arange(L),'L'): # propogate L cycles ahead
        y_pred = np.zeros((N,len(r)))
        for k in np.arange(N): # loop through each ensemble member
            v_boundary_k = E[k,:] * (u.km/u.s)
            model_k = H.HUXt(v_boundary=v_boundary_k, cr_num = 2000, lon_out=lon*u.deg, simtime=simT, dt_scale=obsT)
            cme_list = []
            model_k.solve(cme_list)
            y_pred[k,:] = model_k.v_grid[i,:,int(lon)].value

        ens_o_mean, ens_o_center = center(y_pred) # respectively: expected observation from ensemble, anomaly matrix
        ens_o_center = ens_o_center/eps
    
        # Gauss-Newton optimization
        grad_incre = grad_incre + beta[i]*ens_o_center@np.linalg.inv(R_o)@np.transpose(y[i,:]-ens_o_mean) # gradient increment after each time step
        Hessian_incre = Hessian_incre + beta[i]*ens_o_center@np.linalg.inv(R_o)@ens_o_center.T # Hessian increment after each time step

    grad = (N-1)*w - grad_incre
    Hessian = (N-1)*np.eye(N) + Hessian_incre
    D, V = np.linalg.eig(Hessian) # eigenvalue decomposition to calculate inverse
    Hessian_inv = V@np.diag(1/D)@V.T
    dw = grad@Hessian_inv

    w = w - dw
    j = j + 1

    if np.linalg.norm(dw) <= e:
        break
    
H_inv_sqrt = V@np.diag(1/np.sqrt(D))@V.T
E = ens_mean + w@ens_center + (np.sqrt(N-1)*ens_center.T@H_inv_sqrt@np.eye(N)).T
for k in np.arange(N): # loop through each ensemble member
    v_boundary_k = E[k,:] * (u.km/u.s)
    model_k = H.HUXt(v_boundary=v_boundary_k, cr_num = 2000, simtime=simT, dt_scale=obsT)
    cme_list = []
    model_k.solve(cme_list)
    id_r = np.argmin(np.abs(model_k.r - 30 * u.solRad))
    E_1[k,:] = model_k.v_grid[412,id_r,:].value
x_1 = center(E_1)[0]
E_1 = np.ones((N,1)).dot(x_1) + infl*(np.copy(E_1)-x_1)
# x_a = center(E_1)[0]
x_a = np.mean(E_1,axis=0,keepdims=True).squeeze()

# Test 
v_boundary_post = x_a * (u.km/u.s)
model_post = H.HUXt(v_boundary=v_boundary_post, cr_num = 2000, lon_out=lon*u.deg, simtime=simT, dt_scale=obsT) 
cme_list = []
model_post.solve(cme_list)
x_post = model_post.v_grid[:,:,int(lon)].value

print(x_pri[:,100])
t_day = model_true.time_out.to(u.day)
fig, ax = plt.subplots()
ax.set_xlim(0, 10)
ax.set_ylim(200, 600)
ax.set_ylabel('Solar Wind Speed')
ax.set_xlabel('Time (days)')
ax.grid(True)
ax.plot(t_day,xTrue[:,100],'k-',label='True')
ax.plot(t_day,x_pri[:,100],'r-',label='Prior')
ax.plot(t_day,x_post[:,100],'g-',label='Posterior')
ax.legend()
plt.show()


"""
# RMSE
x_a_rmse_t = rmse(x_a[:DAW_left,:],xTrue[BurnInCyc+1:BurnInCyc+1+DAW_left,:])
#print(x_a_rmse_t)


# Calculate RMSE
x_a_rmse = rmse(x_a,xTrue[BurnInCyc+1:len(xTrue)-L+1,:])
x_p_rmse = rmse(x_p[BurnInCyc+1:len(xTrue)-L+1,:],xTrue[BurnInCyc+1:len(xTrue)-L+1,:])
print("posterior RMSE=",x_a_rmse)
print("prior RMSE=",x_p_rmse)
"""


