# Import necessary packages
import numpy as np
import math
from tools import *
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
import datetime
import astropy.units as u
import os
from random import *

import huxt as H

#####################################################
# Model FACT:
# v_grid = [time, radial, longitudinal]
# radial grids = 141
# longitudinal grids = 128
#####################################################

#####################################################
# Experiment setup:

#####################################################

# Define parameters
initDate = '04032007' # in the format of DD/MM/YYYY
DAW = 27 # DAW length in days
DAWNum = 102 # number of DAW, with 102 DAW, the experiment covers 03/04/2007 - 11/11/2014
lonNum = 128 # number of longitude points
locRad = 15 # localization radius, 0 means no localization
deltaPhiDeg=float(360.0/lonNum)
dt_years = 27/129 *u.day
dt_s = dt_years.to(u.s)
dt_scale = dt_s.value/521.775 # dt to have 128 output from HUXt
beta = np.ones(lonNum) # weight in MDA scheme

M = 128 # state at inner boundary
N = 30 # ensemble number
eps = 1e-4 # scaling factor for finite difference method
e = 1e-1 # stopping criteria for Gauss-Newton
j_max = 20 # max iteration of Gauss-Newton
infl = 1.02 # inflation

seed(20000) # seed for random number generator

#####################################################
# Initialization
#####################################################

# Initialize plotting variables
plotPriorSta = np.zeros((DAWNum,lonNum))
plotPriorStb = np.zeros((DAWNum,lonNum))

plotMASSta = np.zeros((DAWNum,lonNum))
plotMASStb = np.zeros((DAWNum,lonNum))

postStaEns = np.zeros((N,lonNum))
postStbEns = np.zeros((N,lonNum))

plotPostSta = np.zeros((DAWNum,lonNum))
plotPostStb = np.zeros((DAWNum,lonNum))

# Initialize RMSE variable
squareDiffMASSta = np.zeros(DAWNum)
squareDiffPriSta = np.zeros(DAWNum)
squareDiffPostSta = np.zeros(DAWNum)

squareDiffMASStb = np.zeros(DAWNum)
squareDiffPriStb = np.zeros(DAWNum)
squareDiffPostStb = np.zeros(DAWNum)

RMSEMASSta = np.zeros(DAWNum)
RMSEPriSta = np.zeros(DAWNum)
RMSEPostSta = np.zeros(DAWNum)

RMSEMASStb = np.zeros(DAWNum)
RMSEPriStb = np.zeros(DAWNum)
RMSEPostStb = np.zeros(DAWNum)

# Find current directory
currentDir = os.getcwd()
outputDir = currentDir + '/output/'
dataDir = currentDir + '/data/'

# Create the folder to store plots
plotDir = outputDir + 'Plots/'

if not os.path.isdir(plotDir):
    os.makedirs(plotDir+'STEA/')
    os.makedirs(plotDir+'STEB/')

# Create the folder to save RMSE
rmseDir = outputDir+'RMSE/'

if not os.path.isdir(rmseDir):
    os.makedirs(rmseDir)

###########################################################
# Convert from date to CR
###########################################################
initMJD = dateToMJD(initDate) # convert date to MJD
finalMJD=initMJD+(27*DAWNum)

print('Initial MJD: '+str(initMJD))
print('Final MJD  : '+str(finalMJD))
print('No. Windows: '+str(DAWNum))

MJD_all = []
CR_all = []
currMJD = []
currCR = []

# Extract the file that collects all MJD and the corresponding CR number
MJDToCRFile = dataDir+'CR_MJD/2039_2300CRMJDstart.csv'
MJDToCR = open(MJDToCRFile,'r')
MJDToCRLines = MJDToCR.readlines()
MJDToCR.close()

for l in range(len(MJDToCRLines)):
    fileLine = (MJDToCRLines[l].strip()).split(',')

    CR_all.append(int(fileLine[0]))
    MJD_all.append(float(fileLine[1]))

# Match MJD to the CR number for each DAW window
for i in range(DAWNum):
    currMJD.append(initMJD + DAW*i)
    CRIndex = next(id for id, val in enumerate(MJD_all) if val>currMJD[i])
    currCR.append(CR_all[CRIndex-1])

###########################################################
# construct prior ensemble from the MAS observations
###########################################################
# Make a file that contains the MJD corresponding to each longitudinal point
fileMJDLonDir = outputDir+'MJDfiles/'
fileMJDLon = []

for i in range(DAWNum):
    fileMJDLon.append(fileMJDLonDir+'MJD_'+str(int(currMJD[i]))+'.dat')

    if not os.path.isfile(fileMJDLon[i]):
        if not os.path.isdir(fileMJDLonDir):
            os.makedirs(fileMJDLonDir)
        makeMJDLon(currMJD[i],lonNum,fileMJDLon[i])

# Create ensemble files and save them
fileInitSpecDir=outputDir+'MJDfiles/MJDstart_'+str(int(initMJD))+'_nWindows_'+str(DAWNum)+'/' #Directory to hold solar wind ens files specific to this run
fileInit = []
MASObsDir = currentDir+'/data/MASens/'

for i in range(DAWNum):
    fileInit.append(fileInitSpecDir+'vin_ensemble_MJDstart'+str(int(currMJD[i]))+'.dat')

    if not os.path.isfile(fileInit[i]):
        if not os.path.isdir(fileInitSpecDir):
            os.makedirs(fileInitSpecDir)
        makeVinEns(currMJD[i],currCR[i],fileMJDLonDir,MASObsDir,fileInit[i],lonNum)

###########################################################
# Data Assimilation
###########################################################
# Open output file to store RMSE and covariance
outFile = open(outputDir+'RMSE.txt','w')
covFile = open(outputDir+'Covariance.txt','w')

for win in range(DAWNum-1):
    print('DAW Window No.: '+str(win+1)+'/'+str(DAWNum))

    ###########################################################
    # Compute Prior
    ###########################################################
    # Get the prior mean and covariance from the MAS observation for each 27-day window
    priorMean, priorCov = priorStat(fileInit[win],locRad,lonNum,deltaPhiDeg)

    # Construct prior
    prior = np.zeros(lonNum)
    B0InitState=np.copy(linalg.sqrtm(priorCov))
    normalVector=np.zeros(lonNum)
    for i in range(lonNum):
        normalVector[i]=gauss(0,1)
    initEns=(np.transpose(np.real(B0InitState))).dot(normalVector)

    for i in range(lonNum):                
        #If perturbation pushes ensemble outside of acceptable values for solar wind speed, adjust accordingly
        if (priorMean[i]+initEns[i]<200):
            prior[i]=200+0.1*(abs(200-abs(priorMean[i]+initEns[i])))
        elif (priorMean[i]+initEns[i]>800):
            prior[i]=800-0.1*(abs(800-(priorMean[i]+initEns[i])))
        else:
            prior[i]=priorMean[i]+initEns[i]

    ###########################################################
    # Compute Prior trajectory
    ###########################################################
    v_boundary = prior * (u.km/u.s)
    modelPrior = H.HUXt(v_boundary=v_boundary, cr_num=currCR[win+1], simtime=27*u.day, dt_scale=dt_scale)
    modelPrior.solve([])

    id_r_sta, id_lon_sta, id_r_stb, id_lon_stb = ObserverLoc(modelPrior)

    vPrior_sta_r = modelPrior.v_grid[:, id_r_sta, :].value

    vPrior_stb_r = modelPrior.v_grid[:, id_r_stb, :].value

    plotPriorSta[win,:] = modelPrior.v_grid[:, id_r_sta, id_lon_sta].value
    plotPriorStb[win,:] = modelPrior.v_grid[:, id_r_stb, id_lon_stb].value

    ###########################################################
    # Compute MAS Mean trajectory
    ###########################################################
    # Use prior mean to generate the prior trajectory at STEREO-A
    v_boundary = priorMean * (u.km/u.s)
    modelMAS = H.HUXt(v_boundary=v_boundary, cr_num=currCR[win+1], simtime=27*u.day, dt_scale=dt_scale)
    modelMAS.solve([])

    plotMASSta[win,:] = modelMAS.v_grid[:, id_r_sta, id_lon_sta].value
    plotMASStb[win,:] = modelMAS.v_grid[:, id_r_stb, id_lon_stb].value

    ###########################################################
    # Get Observations
    ###########################################################
    # Read Observations
    fileSTA = dataDir+'ObservationFiles/stereoa_2007_2020.lst'
    fileSTB = dataDir+'ObservationFiles/stereob_2007_2020.lst'
    fileACE = dataDir+'ObservationFiles/ace_2007_2020.lst'

    yA = readObsFile(fileSTA,fileMJDLon[win])
    yB = readObsFile(fileSTB,fileMJDLon[win])
    yACE = readObsFile(fileACE,fileMJDLon[win])

    # Make copies
    yAPlot = np.copy(yA)
    yBPlot = np.copy(yB)
    yACEPlot = np.copy(yACE)

    noOfObsA = lonNum
    noOfObsB = lonNum
    noOfObsACE = lonNum

    obsValidA = range(noOfObsA)
    obsValidB = range(noOfObsB)
    obsValidACE = range(noOfObsACE)

    obsNotValidA=[]
    obsNotValidB=[]
    obsNotValidACE=[]

    # Filter out observations that are too large or negative
    #Initialise temporary variables
    noOfObsATemp=np.copy(noOfObsA)
    noOfObsBTemp=np.copy(noOfObsB)
    noOfObsACETemp=np.copy(noOfObsACE)

    yATemp=list(np.copy(yA))
    yBTemp=list(np.copy(yB))
    yACETemp=list(np.copy(yACE))

    obsValidATemp=list(np.copy(obsValidA))
    obsValidBTemp=list(np.copy(obsValidB))
    obsValidACETemp=list(np.copy(obsValidACE))
    
    # Check if obs. speed negative or greater than 2000km/s remove observation
    for i in range(128):
        if ((abs(yA[i])>2000) or (yA[i]<0)):
            #Remove observation from list
            yATemp.remove(yA[i])
            
            #Update plotting variable as NaN
            yAPlot[i]=np.NaN
            
            #Reduce number of obs. to be taken and record which longitude obs is removed from (and which obs. are to be taken)
            noOfObsATemp=noOfObsATemp-1
            obsValidATemp.remove(obsValidA[i])
            obsNotValidA.append(obsValidA[i])

        if ((abs(yB[i])>2000) or (yB[i]<0)):
            #Remove observation from list
            yBTemp.remove(yB[i])
            
            #Update plotting variable as NaN
            yBPlot[i]=np.NaN
            
            #Reduce number of obs. to be taken and record which longitude obs is removed from (and which obs. are to be taken)
            noOfObsBTemp=noOfObsBTemp-1
            obsValidBTemp.remove(obsValidB[i])
            obsNotValidB.append(obsValidB[i])

        if ((abs(yACE[i])>2000) or (yACE[i]<0)):
            #Remove observation from list
            yACETemp.remove(yACE[i])
            
            #Update plotting variable as NaN
            yACEPlot[i]=np.NaN
            
            #Reduce number of obs. to be taken and record which longitude obs is removed from (and which obs. are to be taken)
            noOfObsACETemp=noOfObsACETemp-1
            obsValidACETemp.remove(obsValidACE[i])
            obsNotValidACE.append(obsValidACE[i])
        
    noOfObsA = np.copy(noOfObsATemp)
    noOfObsB = np.copy(noOfObsBTemp)
    noOfObsACE = np.copy(noOfObsACETemp)

    obsValidA = np.copy(obsValidATemp)
    obsValidB = np.copy(obsValidBTemp)
    obsValidACE = np.copy(obsValidACETemp)

    yA = np.copy(yATemp)
    yB = np.copy(yBTemp)
    yACE = np.copy(yACE)

    ###########################################################
    # Get Observation Covariance
    ###########################################################
    # Get valid prior based on valid observation
    vPrior_sta_valid = [vPrior_sta_r[:,i] for i in obsValidA]
    vPrior_stb_valid = [vPrior_stb_r[:,i] for i in obsValidB]

    # Taking avaerage of prior prediction over all longitudes
    vPrior_sta_avg = np.mean(vPrior_sta_valid,axis=1)
    vPrior_stb_avg = np.mean(vPrior_stb_valid,axis=1)

    R_kk_sta = (0.1*vPrior_sta_avg)**2
    R_kk_stb = (0.1*vPrior_stb_avg)**2

    R_sta = np.diag(R_kk_sta)
    R_stb = np.diag(R_kk_stb)

    ###########################################################
    # Perform iEnKS
    ###########################################################
    # Construct ensemble from prior mean and covariance
    E = np.random.multivariate_normal(priorMean,priorCov,N)

    E_1 = np.zeros(np.copy(E.shape)) # ensemble at the next time 

    # Initialize Gauss-Newton iteration
    ens_mean, ens_center = center(E) # respectively: ensemble mean, anomaly matrix
    w = np.zeros((1,N)) # initialize control vector
    dw_init = 0

    # Iteration
    for j in progbar(np.arange(j_max),'iEnKS Progress:'):
        x = ens_mean + w.dot(ens_center) # current solution of the state in the ensemble space: x = x_mean + A*w

        E = np.ones((N,1)).dot(x) + eps*ens_center # ensemble to be passed to the model

        y_pred_sta = np.zeros((N,lonNum))
        y_pred_stb = np.zeros((N,lonNum))

        for k in np.arange(N): # loop through each ensemble member, bc I can only pass one ensemble at a time into the HUXt model
            v_boundary_k = np.clip(E[k,:],a_min = 200, a_max = 1000) * (u.km/u.s)
            model_k = H.HUXt(v_boundary=v_boundary_k, cr_num = currCR[win], simtime=27*u.day, dt_scale=dt_scale)
            cme_list = []
            model_k.solve(cme_list)
            id_r_sta, id_lon_sta, id_r_stb, id_lon_stb = ObserverLoc(model_k)
            y_pred_sta[k,:] = model_k.v_grid[:,id_r_sta,id_lon_sta].value # Predicted SW condition from ensemble
            y_pred_stb[k,:] = model_k.v_grid[:,id_r_stb,id_lon_sta].value

        ens_o_sta_mean, ens_o_sta_center = center(y_pred_sta) # respectively: expected observation from ensemble, anomaly matrix
        ens_o_sta_center_scale = ens_o_sta_center/eps

        ens_o_stb_mean, ens_o_stb_center = center(y_pred_stb) # respectively: expected observation from ensemble, anomaly matrix
        ens_o_stb_center_scale = ens_o_stb_center/eps

        # Choose valid observations
        ens_o_sta_center_valid = ens_o_sta_center_scale[:,obsValidA]
        ens_o_stb_center_valid = ens_o_stb_center_scale[:,obsValidB]

        ens_o_sta_mean_valid = ens_o_sta_mean[:,obsValidA]
        ens_o_stb_mean_valid = ens_o_stb_mean[:,obsValidB]
    
        # Gauss-Newton optimization (the commented out part is for the version that requires inflation)
        # grad = (N-1)*w.T - (beta[i]*ens_o_sta_center_valid.dot(np.linalg.inv(R_sta)).dot(np.transpose(np.array(yA)-ens_o_sta_mean_valid)) + beta[i]*ens_o_stb_center_valid.dot(np.linalg.inv(R_stb)).dot(np.transpose(np.array(yB)-ens_o_stb_mean_valid)))# gradient increment after each time step
        grad = N * w.T/(1+w.dot(w.T)) - (beta[i]*ens_o_sta_center_valid.dot(np.linalg.inv(R_sta)).dot(np.transpose(np.array(yA)-ens_o_sta_mean_valid)) + beta[i]*ens_o_stb_center_valid.dot(np.linalg.inv(R_stb)).dot(np.transpose(np.array(yB)-ens_o_stb_mean_valid)))
        Hessian = (N-1)*np.eye(N) + beta[i]*ens_o_sta_center_valid.dot(np.linalg.inv(R_sta)).dot(ens_o_sta_center_valid.T) + beta[i]*ens_o_stb_center_valid.dot(np.linalg.inv(R_stb)).dot(ens_o_stb_center_valid.T) # Hessian increment after each time step

        D, V = np.linalg.eig(Hessian) # eigenvalue decomposition to calculate inverse
        Hessian_inv = V@np.diag(1/D)@V.T
        dw = grad.T.dot(Hessian_inv)

        if j == 0:
            dw_init = np.linalg.norm(dw)

        w = w - dw
        j = j + 1

        print('dw=',np.linalg.norm(dw))

        if np.linalg.norm(dw) <= max(e,np.linalg.norm(dw_init)/10):
            break

    Hessian_1 = N*((1+w.dot(w.T))*np.eye(N)-2*w.T.dot(w))/(1+w.dot(w.T))**2 + beta[i]*ens_o_sta_center_valid.dot(np.linalg.inv(R_sta)).dot(ens_o_sta_center_valid.T) + beta[i]*ens_o_stb_center_valid.dot(np.linalg.inv(R_stb)).dot(ens_o_stb_center_valid.T) 
    D, V = np.linalg.eig(Hessian_1) # eigenvalue decomposition to calculate inverse
    H_inv_sqrt = V.dot(np.diag(np.sqrt(1/D))).dot(V.T)

    E = np.ones((N,1)).dot(x) + (np.sqrt(N-1)*ens_center.T.dot(H_inv_sqrt).dot(np.eye(N))).T # reconstruct the ensemble based on the desired mean and covariance

    for k in np.arange(N): # loop through each ensemble member, bc I can only pass one ensemble at a time into the HUXt model
        v_boundary_k = np.clip(E[k,:],a_min=200,a_max=1000) * (u.km/u.s)
        model_k = H.HUXt(v_boundary=v_boundary_k, cr_num = currCR[win], simtime=27*u.day, dt_scale=dt_scale)
        cme_list = []
        model_k.solve(cme_list)
        id_r = np.argmin(np.abs(model_k.r - 30 * u.solRad))
        E_1[k,:] = model_k.v_grid[lonNum-1,id_r,:].value
    # x_1 = center(E_1)[0]
    # E_1 = np.ones((N,1)).dot(x_1) + infl*(np.copy(E_1)-x_1)

    if win >= 1:
        for i in np.arange(N):
            v_boundary_post = np.clip(E_1_prev[i,:],a_min=200, a_max=1000) * (u.km/u.s)
            modelPost = H.HUXt(v_boundary=v_boundary_post, cr_num = currCR[win], simtime=27*u.day, dt_scale=dt_scale)
            modelPost.solve([])

            id_r_sta, id_lon_sta, id_r_stb, id_lon_stb = ObserverLoc(modelPost)

            postStaEns[i,:] = modelPost.v_grid[:, id_r_sta, id_lon_sta].value
            postStbEns[i,:] = modelPost.v_grid[:, id_r_stb, id_lon_stb].value

        # Get the mean and covariance from the posterior ensemble
        postStaMean, postStaAnom = center(postStaEns)
        postStbMean, postStbAnom = center(postStbEns)

        postStaCov = 1/(N-1)*postStaAnom.T.dot(postStaAnom)
        postStbCov = 1/(N-1)*postStbAnom.T.dot(postStbAnom)

        # Compute the standard deviation from the covariance matrix
        postStaStd = np.sqrt(np.diag(postStaCov))
        postStbStd = np.sqrt(np.diag(postStbCov))

        plotPostSta[win,:] = postStaMean
        plotPostStb[win,:] = postStbMean


        figure(5000+win)
        plot(np.arange(0,359,deltaPhiDeg),yAPlot,color='k',linewidth=2.0,label='STEREO-A Data')
        plot(np.arange(0,359,deltaPhiDeg),plotMASSta[win-1,:],color='m',linewidth=3.0,label='MAS Mean')
        plot(np.arange(0,359,deltaPhiDeg),plotPriorSta[win-1,:],color='b',linewidth=3.0,label='Prior')
        plot(np.arange(0,359,deltaPhiDeg),plotPostSta[win,:], color='g',linewidth=2.0,label='Posterior')
        xlabel('Time (days)',fontsize=18)
        ylabel('Speed (km/s)',fontsize=18)
        xlim(0,360)
        ylim(240,800)
        yticks(np.arange(300,801,100),('300','400','500','600','700','800'),fontsize=18)
        xticks(np.append(np.arange(0,361,53.3),360), ('0','4','8','12','16','20','24','27'),fontsize=18)
        title(r'Solar wind speed at STEREO-A',fontsize=18)
        legend()
        tight_layout()
        savefig(plotDir+'STEA/SWspeedSTEREOA_MJDstart'+str(int(currMJD[win]))+'.png')

        figure(6000+win)
        plot(np.arange(0,359,deltaPhiDeg),yBPlot,color='k',linewidth=2.0,label='STEREO-B Data')
        plot(np.arange(0,359,deltaPhiDeg),plotMASStb[win-1,:],color='m',linewidth=2.0,label='MAS Mean')
        plot(np.arange(0,359,deltaPhiDeg),plotPriorStb[win-1,:],color='b',linewidth=2.0,label='Prior')
        plot(np.arange(0,359,deltaPhiDeg),plotPostStb[win,:],color='g',linewidth=2.0,label='Posterior')
        xlabel('Time (days)',fontsize=18)
        ylabel('Speed (km/s)',fontsize=18)
        xlim(0,360)
        ylim(240,800)
        yticks(np.arange(300,801,100),('300','400','500','600','700','800'),fontsize=18)
        xticks(np.append(np.arange(0,361,53.3),360), ('0','4','8','12','16','20','24','27'),fontsize=18)
        title(r'Solar wind speed at STEREO-B',fontsize=18)
        legend()
        tight_layout()
        savefig(plotDir+'STEB/SWspeedSTEREOB_MJDstart'+str(int(currMJD[win]))+'.png')

        # Calculate RMSE
        validA = 0
        validB = 0

        for i in range(lonNum):
            if np.isnan(yAPlot[i]) ==0:
                validA = validA + 1
                squareDiffMASSta[win] = squareDiffMASSta[win] + (plotMASSta[win-1,i]-yAPlot[i])**2
                squareDiffPriSta[win] = squareDiffPriSta[win] + (plotPriorSta[win-1,i]-yAPlot[i])**2
                squareDiffPostSta[win] = squareDiffPostSta[win] + (plotPostSta[win,i]-yAPlot[i])**2

            if np.isnan(yBPlot[i]) ==0:
                validB = validB + 1
                squareDiffPriStb[win] = squareDiffPriStb[win] + (plotPriorStb[win-1,i]-yBPlot[i])**2
                squareDiffMASStb[win] = squareDiffMASStb[win] + (plotMASStb[win-1,i]-yBPlot[i])**2
                squareDiffPostStb[win] = squareDiffPostStb[win] + (plotPostStb[win,i]-yBPlot[i])**2

        RMSEMASSta[win] = np.sqrt(squareDiffMASSta[win]/validA)
        RMSEPriSta[win] = np.sqrt(squareDiffPriSta[win]/validA)
        RMSEPostSta[win] = np.sqrt(squareDiffPostSta[win]/validA)

        RMSEMASStb[win] = np.sqrt(squareDiffMASStb[win]/validB)
        RMSEPriStb[win] = np.sqrt(squareDiffPriStb[win]/validB)
        RMSEPostStb[win] = np.sqrt(squareDiffPostStb[win]/validB)

        # Write RMSE into output file
        outFile.write('------------------------------------------------------\n')
        outFile.write('Window no. '+str(win)+'\n\n')
        outFile.write('RMSE STEREO A MASMean = '+str(RMSEMASSta[win])+'\n')
        outFile.write('RMSE STEREO A Prior = '+str(RMSEPriSta[win])+'\n')
        outFile.write('RMSE STEREO A Posterior = '+str(RMSEPostSta[win])+'\n')

        outFile.write('RMSE STEREO B MASMean = '+str(RMSEMASStb[win])+'\n')
        outFile.write('RMSE STEREO B Prior = '+str(RMSEPriStb[win])+'\n')
        outFile.write('RMSE STEREO B Posterior = '+str(RMSEPostStb[win])+'\n')
        outFile.write('------------------------------------------------------\n\n') 

        # # Write covariance into output file
        covFile.write('------------------------------------------------------\n')
        covFile.write('Window no. '+str(win)+'\n\n')
        covFile.write('Posterior Covariance at STEREO A'+'\n')
        np.savetxt(covFile, postStaCov, fmt='%.4f')
        covFile.write('\n')
        covFile.write('Posterior Covariance at STEREO B'+'\n')
        np.savetxt(covFile, postStbCov, fmt='%.4f')
        covFile.write('\n')
        covFile.write('Prior Covariance '+'\n')
        np.savetxt(covFile, priorCov, fmt='%.4f')
        covFile.write('\n')
        covFile.write('Observation at STEREO A Covariance '+'\n')
        np.savetxt(covFile, R_sta, fmt='%.4f')
        covFile.write('\n')
        covFile.write('Observation at STEREO B Covariance '+'\n')
        np.savetxt(covFile, R_stb, fmt='%.4f')
        covFile.write('\n')
        outFile.write('------------------------------------------------------\n\n') 

        figure(7000+win)
        plot(np.arange(1,win+1),RMSEMASSta[1:win+1],color='m',marker='o',markersize=4,label='MASMean RMSE')
        plot(np.arange(1,win+1),RMSEPriSta[1:win+1],color='b',marker='o',markersize=4,label='Prior RMSE')
        plot(np.arange(1,win+1),RMSEPostSta[1:win+1],color='g',marker='o',markersize=4,label='Posterior RMSE')
        xlabel('Window',fontsize=18)
        ylabel('RMSE (km/s)',fontsize=18)
        title(r'RMSE at $213r_S$',fontsize=18)
        xlim(0,win+1)
        legend(fontsize=8)
        tight_layout()
        grid()
        savefig(rmseDir+'RMSE_STEA.png')  

        figure(8000+win)
        plot(np.arange(1,win+1),RMSEMASStb[1:win+1],color='m',marker='o',markersize=4,label='MASMean RMSE')
        plot(np.arange(1,win+1),RMSEPriStb[1:win+1],color='b',marker='o',markersize=4,label='Prior RMSE')
        plot(np.arange(1,win+1),RMSEPostStb[1:win+1],color='g',marker='o',markersize=4,label='Posterior RMSE')
        xlabel('Window',fontsize=18)
        ylabel('RMSE (km/s)',fontsize=18)
        title(r'RMSE at $213r_S$',fontsize=18)
        xlim(0,win+1)
        legend(fontsize=8)
        tight_layout()
        grid()
        savefig(rmseDir+'RMSE_STEB.png')  

        print('RMSE STEREO A MASMean = '+str(RMSEMASSta[win]))
        print('RMSE STEREO A Forecast = '+str(RMSEPriSta[win]))
        print('RMSE STEREO A Posterior = '+str(RMSEPostSta[win]))

        print('RMSE STEREO B MASMean = '+str(RMSEMASStb[win]))
        print('RMSE STEREO B Forecast = '+str(RMSEPriStb[win]))
        print('RMSE STEREO B Posterior = '+str(RMSEPostStb[win]))
    
    E_1_prev = np.copy(E_1)

outFile.close()
covFile.close()
