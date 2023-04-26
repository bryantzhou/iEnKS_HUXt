import tqdm
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
from math import floor
import sys

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
    # x_mean = x_mean.squeeze()
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


#####################################################################################
# Tools dealing with observation files
#####################################################################################

#Convert between date (string in form DDMMYYYY) to MJD
def dateToMJD(date): 
    day=int(date[:2])
    month=int(date[2:4])
    year=int(date[4:])
    
    if not 0<day<32:
        print('Invalid day input in date, please check.')
        print('Day input='+str(day))
        sys.exit()
    
    if not 0<month<13:
        print('Invalid month input in date')
        print('Month input='+str(month))
        sys.exit()
        
    if not 2004<year<2025:
        print('This program only performs DA over the period of the STEREO missions and cannot assimilate observations from the future. Please check year input in date')
        print('Year input='+str(year))
        sys.exit()
        
    julianDayObs=(367*year)-floor(7*(year+floor((month+9)/12.0))*0.25)-floor(0.75*(floor(0.01*(year+((month-9)/7.0)))+1))+floor((275*month)/9.0)+day+1721028.5
    
    #julianTimeObs=julianDayObs+(hour)/24.0
    MJD=julianDayObs-2400000.5
    #sys.exit()
    #julian.to_jd(timeObs,format='mjd')
    
    return MJD

# Make a file that stores the MJD corresponds to each logitudinal point
# The reason of making this file is to locate the starting MJD precisely in terms of the longitudinal point
def makeMJDLon(MJDstart, lonNum, fileOut, daySolarRot=27):
    MJDStep = daySolarRot/lonNum
    
    MJD = np.zeros(lonNum)
    MJD[0] = MJDstart

    for i in range(lonNum-1):
        MJD[i+1] = MJD[i]+MJDStep

    file = open(fileOut,'w')
    for i in range(lonNum-1,-1,-1):
        file.write(str(MJD[i])+'\n')

    file.close()

# Make initial ensemble file from MAS observations
def makeVinEns(MJDstart, CRstart, MJDLonDir, MASObsDir,fileOut,numLon,numEns=576):
    
    fileMJDLonName = MJDLonDir+'MJD_'+str(int(MJDstart))+'.dat'

    fileMJDLon = open(fileMJDLonName,'r')
    fileMJDLonLines = fileMJDLon.readlines()
    fileMJDLon.close()

    startCoord = 0
    for i in range(len(fileMJDLonLines)):
        if MJDstart > float(fileMJDLonLines[i].strip()):
            startCoord = startCoord + 1
    
    endCoord = startCoord + numLon

    # We use the observations from two CR rotations because one DAW might cross from one CR to the next.
    # The starting the ending coordinates are determined from the "makeMJDLon" function.
    fileMASObsName1 = MASObsDir+'vin_ensemble_CR'+str(int(CRstart))+'.dat'
    fileMASObsName2 = MASObsDir+'vin_ensemble_CR'+str(int(CRstart)+1)+'.dat'

    file1 = open(fileMASObsName1,'r')
    file2 = open(fileMASObsName2,'r')

    file1Lines = file1.readlines()
    file2Lines = file2.readlines()

    file1.close()
    file2.close()

    vIn = np.zeros((numEns,len(file1Lines)+len(file2Lines)))
    vOut = np.zeros((numEns,numLon))

    for i in range(len(file1Lines)+len(file2Lines)):
        if i < len(file1Lines):
            vIn[:,i] = file1Lines[i].split('  ')[1:]
        else:
            vIn[:,i] = file2Lines[i-len(file1Lines)].split('  ')[1:]
    
    vOut = np.copy(vIn[:,startCoord:endCoord])

    fOut = open(fileOut,'w')
    np.savetxt(fOut,np.transpose(vOut))
    fOut.close()

# Compute the prior mean and covariance 
def priorStat(ensFile,locRad,numLon,deltaPhiDeg,numEns=576):
    priorEns = np.transpose(np.loadtxt(ensFile))
    priorMean = np.mean(priorEns,axis=0)

    priorAnom = priorEns-priorMean

    priorCov = 1/(numEns-1)*np.transpose(priorAnom).dot(priorAnom)

    if locRad!=0:
        locMat=np.zeros((np.shape(priorCov)))
    
        for i in range(numLon):
            for j in range(numLon):
                distBetweenPoints=min(abs((i-j)*deltaPhiDeg),360-abs((i-j)*deltaPhiDeg))
                locMat[i,j]=np.exp(-(distBetweenPoints**2)/(2*(locRad**2)))
                
        priorCov=locMat*np.copy(priorCov) 
      
    return priorMean, priorCov

# Convert between year, day of year and hour to MJD
def gregToMJD(year,dayOfYear,hour):
    #Find month and day of month from day of year
    #Define length of months
    if np.mod(year,4)==0:
        #dayMonths=[31,29,31,30,31,30,31,31,30,31,30,31]
        daysOfStartMonth=[0,31,60,91,121,152,182,213,244,274,305,335,366]
    else:
        #dayMonths=[31,28,31,30,31,30,31,31,30,31,30,31]
        daysOfStartMonth=[0,31,59,90,120,151,181,212,243,273,304,334,365]
    
    a=0
    m=1
    while ((a==0) or (m==14)):
        if (m==13):
            print('m=13: While loop overdone: Day no. not found')
            sys.exit()
        elif (daysOfStartMonth[m-1]<dayOfYear<=daysOfStartMonth[m]):
            month=m
            a=1
        
        m=m+1
    
    day=dayOfYear-daysOfStartMonth[month-1]
    
    #timeObs=dt.datetime(year,month,day,hour,0,0)
    
    #julianTimeObs=gcal2jd(year,month,day,hour)
    #Julian day init
    #JYear=-4713
    #JMonth=1
    #JDay=1
    #JHour=12
    #jdate=dt.datetime(JYear,JMonth,JDay,JHour,0,0)
    
    #year=2017
    #month=12
    #day=18
    #hour=21
    #timeObs=dt.datetime(year,month,day,hour,0,0)
    
    julianDayObs=(367*year)-floor(7*(year+floor((month+9)/12.0))*0.25)-floor(0.75*(floor(0.01*(year+((month-9)/7.0)))+1))+floor((275*month)/9.0)+day+1721028.5
    
    julianTimeObs=julianDayObs+(hour)/24.0
    MJD=julianTimeObs-2400000.5
    #sys.exit()
    #julian.to_jd(timeObs,format='mjd')
    
    return MJD


# Read observations
def readObsFile(obsFile,MJDLonFile):
    obs = open(obsFile,'r')
    obsLines = obs.readlines()
    obs.close()

    MJDLon = open(MJDLonFile,'r')
    MJDLonLines = MJDLon.readlines()
    MJDLon.close()

    MJDLonLines.reverse()

    MJDTimeAll = []
    obsAll = []
    observations = []

    for ln in obsLines:
        s = ln.split()
        MJDTime = gregToMJD(int(s[0].strip()),int(s[1].strip()),int(s[2].strip()))
        MJDTimeAll.append(MJDTime)

        obsAll.append(float(s[3].strip()))
    i = 0

    for ln in MJDLonLines:
        while MJDTimeAll[i]<float(ln.strip()) and i<(len(obsAll)+2):
            if i == (len(obsAll)+1):
                print('Date out of bounds')
                sys.exit()
            i = i+1
        observations.append(obsAll[i])

    return observations
        
# Provide STEREO-A and STEREO-B locations
def ObserverLoc(model):
    sta_r = np.mean(model.get_observer('sta').r)
    stb_r = np.mean(model.get_observer('stb').r)
    sta_lon = np.mean(model.get_observer('sta').lon)
    stb_lon = np.mean(model.get_observer('stb').lon)

    id_r_sta = np.argmin(np.abs(model.r - sta_r))
    if model.lon.size == 1:
        id_lon_sta= 0
    else:
        id_lon_sta = np.argmin(np.abs(model.lon - sta_lon))
    
    id_r_stb = np.argmin(np.abs(model.r - stb_r))
    if model.lon.size == 1:
        id_lon_stb= 0
    else:
        id_lon_stb = np.argmin(np.abs(model.lon - stb_lon))

    return id_r_sta, id_lon_sta, id_r_stb, id_lon_stb

    
    
