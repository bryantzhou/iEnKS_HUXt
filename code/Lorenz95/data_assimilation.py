# Import necessary packages
import numpy as np
import math
from tools import *
import matplotlib.pyplot as plt
from Lorenz95 import *

# Define parameters
dt = 0.05
L = 1 # DAW length
#L_total = np.linspace(20,20,1) # DAW length
M = 40 # state numer
H = np.eye(M) # truth run is fulling observed
R_o = np.eye(M) # observation error covariance
R_x = np.zeros((M,M))
#N = 20 # ensemble number
N_total = np.array([16,17,18,19,20,25,30,35,40,50])
eps = 1e-4 
e = 1e-3 # stopping criteria for Gauss-Newton
j_max = 50 # max iteration of Gauss-Newton
T = 1e4*dt # DA runs extend over 1e5 cycles
BurnIn = 5e3*dt # burn-in time
infl = 1.02 # inflation
#beta = np.ones(L)
method = 'SDA' # specify iEnKS method: SDA or MDA

BurnInCyc = int(BurnIn/dt)
EndCyc = int(T/dt)

# Live plot
LivePlot = False

# Resulting plot
finalPlot = False

# Initial state
x0Prior = np.random.normal(0,0.01,M)
x0True = np.random.normal(0,0.01,M)

# Posterior state
# x_a = np.zeros((int(T/dt-BurnIn/dt-L),M))

# Log RMSE
x_a_rmse = []
x_p_rmse = []

#############################################################
# Twin Experiment Setup (simulate two trajectores and take observations)
#############################################################
tt = np.linspace(0,T,EndCyc)
x_p = np.zeros((EndCyc,M)) # prior states
xTrue = np.zeros((EndCyc,M)) # truth trajectory
y = np.zeros((EndCyc-BurnInCyc,M)) # observations
xTrue[0]=x0True
x_p[0] = x0Prior

t = 0
for i in progbar(np.arange(len(tt)-1),'Truth & Obs'):
    xTrue[i+1] = step(xTrue[i],t,dt) + math.sqrt(dt)*np.random.multivariate_normal(np.zeros(M),R_x,1) # simulate truth states
    x_p[i+1] = step(x_p[i],t,dt) + math.sqrt(dt)*np.random.multivariate_normal(np.zeros(M),R_x,1) # simulate prior states
    t = t + dt
    if t > BurnIn:
        y[int((t-BurnIn)/dt)-1] = H@xTrue[i+1] + np.random.multivariate_normal(np.zeros(M),R_o,1) # take observations

""""
# Plot truth states in the Lorenz'95 model 
plt.figure(figsize=(7,4))
    
plt_t = 8 # only plot the last several time steps
colors = plt.cm.cubehelix(0.1+0.6*np.linspace(0,1,plt_t))
for k in range(plt_t,0,-1):
    plt.plot(xTrue[len(xTrue)-k],c=colors[plt_t-k])
plt.ylim(-10,20)
plt.show()
"""

################################################################################
# Data Assimilation
################################################################################
# Create ensemble
for N in N_total:
    print('current N=',N)
    L = int(L)
    beta = np.ones(L)
    x_a = np.zeros((int(T/dt-BurnIn/dt-L),M))
    E = np.random.multivariate_normal(xTrue[BurnInCyc+1,:],0.01*np.eye(M),N) # perturb initial state after burn-in

    if LivePlot:
    # Create empty figure
        fig, ax = plt.subplots()
        ax.set_xlim(0, 40)
        ax.set_ylim(-10, 10)
        ax.set_ylabel('Value')
        ax.set_xlabel('State Index')
        ax.set_gid(True)

    # Slide DAW from burn-in time
    for DAW_left in progbar(np.arange(len(y)-L),'iEnKS Progress'): # left of first DAW starts right after burn-in 
        DAW_right = DAW_left+L # right of DAW is L cycles apart from left of DAW

        # Initialize Gauss-Newton iteration
        ens_mean, ens_center = center(E) # respectively: ensemble mean, anomaly matrix
        w = np.zeros(N) # initialize control vector
        j = 0 # iteration index
        dw = 1e5*np.ones(N) # initialize to enter while loop

        # Iteration
        while j <= j_max and np.linalg.norm(dw) > e:
            x = ens_mean + w@ens_center # current solution of the state in the ensemble space: x = x_mean + A*w
            E = x + eps*ens_center # reconstruct the ensemble
            grad_incre = 0
            Hessian_incre = 0

            if method == 'SDA': # if the selected method is simple simple data assimilation
                for i in np.arange(L): # propogate L cycles ahead
                    E = step(E,t,dt)

                # Analysis of observations at L cycles ahead
                E_o = E@H # observation operator is I
                y_L = y[DAW_right] # observation is always at the end of DAW
                ens_o_mean, ens_o_center = center(E_o) # respectively: expected observation from ensemble, anomaly matrix
                ens_o_center = ens_o_center/eps
                U, s, VT = svd0(ens_o_center@np.sqrt(np.linalg.inv(R_o))) # SVD for calculating the approx. Hessian inverse

                # Gauss-Newton optimization
                grad = (N-1)*w - ens_o_center@np.linalg.inv(R_o)@np.transpose(y_L-ens_o_mean) # approx. gradient
                Hessian_inv = (U*(pad0(s**2,N)+(N-1))**-1)@U.T # approx. Hessian
                dw = grad@Hessian_inv

            if method == 'MDA': # if the selected method is multiple data assimilation
                for i in np.arange(L): # propogate L cycles ahead
                    E = step(E,t,dt)

                    # Analysis of observations at L cycles ahead
                    E_o = E@H # observation operator is I
                    y_L = y[DAW_left+i+1] # observation is always at the end of DAW
                    ens_o_mean, ens_o_center = center(E_o) # respectively: expected observation from ensemble, anomaly matrix
                    ens_o_center = ens_o_center/eps
                
                    # Gauss-Newton optimization
                    grad_incre = grad_incre + beta[i]*ens_o_center@np.linalg.inv(R_o)@np.transpose(y_L-ens_o_mean) # approx. gradient
                    Hessian_incre = Hessian_incre + beta[i]*ens_o_center@np.linalg.inv(R_o)@ens_o_center.T

                grad = (N-1)*w - grad_incre
                Hessian = (N-1)*np.eye(N) + Hessian_incre
                D, V = np.linalg.eig(Hessian)
                Hessian_inv = V@np.diag(1/D)@V.T
                dw = grad@Hessian_inv

            w = w - dw
            j = j + 1

        if method == 'SDA':
            H_inv_sqrt = (U*(pad0(s**2,N)+(N-1))**-(1/2))@U.T
        if method == 'MDA':
            H_inv_sqrt = V@np.diag(1/np.sqrt(D))@V.T
        E = ens_mean + w@ens_center + (np.sqrt(N-1)*ens_center.T@H_inv_sqrt@np.eye(N)).T
        x_a[DAW_left,:] = center(E)[0]
        E_plot = np.copy(E)
        E = step(E,t,dt)
        ens_mean, ens_center = center(E)
        E = ens_mean + infl*ens_center

        # RMSE
        x_a_rmse_t = rmse(x_a[:DAW_left,:],xTrue[BurnInCyc+1:BurnInCyc+1+DAW_left,:])
        #print(x_a_rmse_t)   
        
        if LivePlot:
        # Plot true states, observations, and ensemble states
            ax.clear()
            ax.plot(np.arange(M),xTrue[BurnInCyc+DAW_left+1],'k-',label='Truth')
            ax.plot(np.arange(M),y[DAW_left],'g*',label='Observation')
            ax.plot(np.arange(M),E_plot.T[:,1],'-',color='grey',alpha=0.2,label='Ensemble')
            ax.plot(np.arange(M),E_plot.T,'-',color='grey',alpha=0.2)
            ax.text(0.05, 0.05, 't_index= {}'.format(DAW_left+1), transform=ax.transAxes, fontsize=12)

            ax.legend(loc='upper right')

            # Redraw the plot
            plt.draw()
            plt.pause(0.05)

    if LivePlot:
        # Show the final plot
        plt.show()

    if finalPlot:
        error_post = np.abs(x_a-xTrue[BurnInCyc+1:len(xTrue)-L+1,:])
        error_pri = np.abs(x_p[BurnInCyc+1:len(xTrue)-L+1,:]-xTrue[BurnInCyc+1:len(xTrue)-L+1,:])
        fig, ax = plt.subplots(2,1)
        ax[0].set_xlim(0, 5000)
        ax[0].set_ylim(-1, 15)
        ax[0].set_ylabel('Error')
        ax[0].set_xlabel('T')
        ax[0].set_gid(True)
        ax[0].plot(np.arange(4980),error_post[:,0],'r-',label='Posterior')
        ax[0].plot(np.arange(4980),error_pri[:,0],'g-',label='Prior')
        ax[0].legend(loc='upper right')

        ax[1].set_xlim(0, 5000)
        ax[1].set_ylim(-10, 10)
        ax[1].set_ylabel('Error')
        ax[1].set_xlabel('T')
        ax[1].set_gid(True)
        ax[1].plot(np.arange(error_post.shape[0]),x_a[:,0],'r-',label='Posterior')
        ax[1].plot(np.arange(error_post.shape[0]),x_p[BurnInCyc+1:len(xTrue)-L+1,0],'g-',label='Prior')
        ax[1].plot(np.arange(error_post.shape[0]),xTrue[BurnInCyc+1:len(xTrue)-L+1,0],'k-',label='Truth')
        ax[1].legend(loc='upper right')

        plt.show()


    # Calculate RMSE
    x_a_rmse = np.append(x_a_rmse,rmse(x_a,xTrue[BurnInCyc+1:len(xTrue)-L+1,:]))
    x_p_rmse = np.append(x_p_rmse,rmse(x_p[BurnInCyc+1:len(xTrue)-L+1,:],xTrue[BurnInCyc+1:len(xTrue)-L+1,:]))
    print("posterior RMSE=",x_a_rmse)
    print("prior RMSE=",x_p_rmse)

fig, ax = plt.subplots()
ax.set_xlim(10, 50)
ax.set_ylim(0.05, 4)
ax.set_ylabel('RMSE')
ax.set_xlabel('Lag')
ax.grid(True)
ax.plot(N_total,x_a_rmse,'g-',marker='.')

plt.show()




    



















