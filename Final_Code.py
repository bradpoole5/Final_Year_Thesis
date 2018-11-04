
# coding: utf-8

# In[ ]:

import numpy as np
import matplotlib.pyplot as plt
import timeit


# In[ ]:

def getGt (mu,delta,motion):
    theta = mu[2,0]
    Gt = np.array([[1,0,-((motion[0,0]/motion[1,0])*np.cos(theta))+((motion[0,0]/motion[1,0])*np.cos(theta+motion[1,0]*delta))],
                   [0,1,-((motion[0,0]/motion[1,0])*np.sin(theta))+((motion[0,0]/motion[1,0])*np.sin(theta+motion[1,0]*delta))],
                   [0,0,1]])
    return Gt


# In[ ]:

def getVt (mu,delta,motion):
    theta = mu[2,0]
    Vt = np.array([[(-np.sin(theta)+np.sin(theta +motion[1,0]*delta))/motion[1,0],
                    (motion[0,0]*(np.sin(theta)-np.sin(theta+motion[1,0]*delta))/(motion[1,0]**2))+(motion[0,0]*(np.cos(theta+motion[1,0]*delta))*delta/motion[1,0])],
                   [(np.cos(theta)-np.cos(theta +motion[1,0]*delta))/motion[1,0],
                    -(motion[0,0]*(np.cos(theta)-np.cos(theta+motion[1,0]*delta))/(motion[1,0]**2))+(motion[0,0]*(np.sin(theta+motion[1,0]*delta))*delta/motion[1,0])],
                   [0,delta]])
    return Vt


# In[ ]:

def getMt (motion,eParam):
    Mt = np.array([[(eParam[0]*(motion[0,0]**2))+(eParam[1]*(motion[1,0]**2)),0],[0,(eParam[2]*(motion[0,0]**2))+(eParam[3]*(motion[1,0]**2))]])
    return Mt


# In[ ]:

#Jacobian becomes 2x3 maxtrix because of exclusion of the signature
def getHt(mu,landM,index):
    q = (landM[0,index]- mu[0,0])**2 + (landM[1,index]-mu[1,0])**2
    Ht = np.array([[(-(landM[0,index]-mu[0,0])/np.sqrt(q)),(-(landM[1,index]-mu[1,0])/np.sqrt(q)),0],
                  [((landM[1,index]-mu[1,0])/np.sqrt(q)),(-(landM[0,index]-mu[0,0])/np.sqrt(q)),-1]])
    return Ht


# In[ ]:

#Initializes all parameters except sensors
def initNew(init = np.array([[0],[0],[0]]), trans = 1,rot = 0.5):
    
    
    truePos = init
    
    #motionControl - Vt = (vt wt).T with vt velocity in m/s & wt angular velicity in rad/s
    #Remark - randomised for simulation
    mot = np.zeros((2,1))
    mot[0,0] = trans
    mot[1,0] = rot

    
    # A mapping of 4 distinct landmarks [x1,x2,x3],[y1,y2,y3]
#     mt = np.array([[1,1,9,9],[1,9,1,9]])
    mt = np.array([[1,1,9,9,5,9,1,5],[1,9,1,9,9,5,5,1]])
    #measurement noise - set at 0.2 - omit signature
    noiseM = np.array([[0.2],[0.2]])
    
    #Error Parameters for motion - set by user/programmer
    eParams = np.array([0.15,0.15,0.15,0.15])
    
    #Correspondence with each landmark mapped 
#     ct = np.array([0,1,2,3])
    ct = np.array([0,1,2,3,4,5,6,7])
    
    return truePos,mot,mt,ct,noiseM,eParams


# In[ ]:

#Generates parameters for the velocity motion model
def getMotionValues(vel=np.random.uniform(0,1),omg=np.random.uniform(0,1)):
    motion = np.array([[vel],[omg]])
    return motion


# In[ ]:

#zt needs to be defined each time the robot is moved i.e. the size of zt changes
#sensor range - set to 5 m

#full circle sensor

# Robot senses all landmarks within 5m in a 360 degree view
#**********************************************************

#WORKING
# ***********************************************************************
def getSenseFull(mu,cov,landM,noise):
    
    temp = np.array([[0],[0]])
    flag = 0
    
    newCorresp = np.array([0])
    
    for i in range(landM[0,:].size):
        
        #Range Check
        rangeCheck = ((landM[0,i] - mu[0,0])**2) + ((landM[1,i] - mu[1,0])**2)
        rangeCheck = np.sqrt(rangeCheck)
        if(rangeCheck < 5):
            if flag ==0:
                
                temp[:,0] = np.array([[landM[0,i],landM[1,i]]])
                newCorresp[0] = i
                flag = 1
                
            else:
                temp = np.insert(temp,temp[0,:].size,[landM[0,i],landM[1,i]],axis = 1)
                newCorresp = np.hstack([newCorresp,i])
    
    tempFinal = getMeasurement(mu,cov,temp,noise)
                
    return tempFinal,newCorresp,temp    


# In[ ]:

#Remark - introduce noise here

#Converts measurements of landmark to appropriate coordinate system

def getMeasurement(mu,cov,landM,noise):
    
    
    temp = np.zeros((landM.shape))
    
    #picks value for measurement; an error value within the error ellipse
    convMu,convCov = convert(mu,cov)
    
    
    if landM[0,0] != 0 and landM[1,0] !=0:
        for i in range(landM[0,:].size):
            random,prob = noiseValueGenerator(landM[:,i].reshape((2,1)),np.diag(noise.reshape(2,)))
            temp[0,i] = np.sqrt(((-mu[0,0]+random[0])**2) + 
                                ((-mu[1,0]+random[1])**2))
            temp[1,i] = np.arctan2((random[1]-mu[1,0]),(random[0]-mu[0,0])) - mu[2,0]  
    return temp
        


# In[ ]:

#Restriction of movement needs to be implemented

def updateMu(mu,cov,motion,delta,eParam):
    newMu = mu + np.array([[((-motion[0,0]/motion[1,0])*np.sin(mu[2,0])) + ((motion[0,0]/motion[1,0])*np.sin(mu[2,0] + motion[1,0]*delta))]
                         ,[((motion[0,0]/motion[1,0])*np.cos(mu[2,0])) - ((motion[0,0]/motion[1,0])*np.cos(mu[2,0] + motion[1,0]*delta))]
                         ,[motion[1,0]*delta]])
    
    #Motion model Uncertainty
    
    #new covariance needed
    Mt = getMt(motion,eParam)
    convMu,convCov = convert(newMu,cov)
    convCov = convCov + Mt
    
#     temp,probo = noiseValueGenerator(convMu,convCov)
#     newMu = np.vstack((temp,newMu[2,0]))

    newCov = np.hstack((np.vstack((convCov,cov[2,:-1])),np.array([cov[:,2]]).T))
    return newMu,newCov


# In[ ]:

# Determines a value that can be used for noise in sensors etc
#Returns the randomly generated noise value within the covariance contour & the likelihhod of that point
def noiseValueGenerator(mu,cov):
    

    a,b = np.linalg.eigvals(cov)
    
    a = a**2
    b = b**2
    
    aRand = np.random.uniform(mu[0,0]-a,mu[0,0]+a)
    bRand = np.random.uniform(mu[1,0]-b,mu[1,0]+b)
    
    randVec = np.vstack((aRand,bRand))
    
    
    #Generating the gaussian distribution isnt necessary for the simulation
    #The gaussian gives the likelihood of that random point
    
    gauss = np.abs(np.linalg.det((2*np.pi*(cov**2))))**-0.5
    
    #Value needed for X
    #change
    gauss2 = np.exp(-0.5*np.dot(np.dot((randVec-mu).T,(np.linalg.inv(cov))),(randVec-mu)))
    
    prob = gauss*gauss2
    return randVec,prob


# In[ ]:

def convert(mu,cov):
    
    convMu = np.array([[mu[0,0]],[mu[1,0]]])
    convCov = np.array([cov[:-1,:-1]])
    convCov = convCov.reshape(2,2)
    
    return convMu,convCov


# In[ ]:

# #Ensure noise stays constant throughout simulation
def simulateData(initial = np.array([[0],[0],[0]]),delta=1,trans=1,rot=0.5):
    truePos,motion,mapping,corresp,noiseM,eParams = initNew(initial,trans,rot)
    trueCov = np.zeros((3,3))
    trackedTrue = np.zeros((1,3,1))
    covMock = np.zeros((3,3))
    #The trackedSense will hold 8 values, in the case all 8 lanmarks are measured - check for zero to get proper list
    #8 landmarks are given - any changes in the nuber; code needs to be changed accordingly
    trackedSense = np.zeros((1,2,8))
    preTrackSense = np.zeros((2,8))
    trackedCt = np.zeros((1,1,8))
    preTrackedCt = np.array([[-1,-1,-1,-1,-1,-1,-1,-1]])
    
    trackedU = np.zeros((1,2,1))
    
    

    #Robot senses landmarks within a 5 meter radius in 360 degrees
    #Conformation needed for accuracy

    truePos,covMock = updateMu(truePos,covMock,motion,delta,eParams)
    ztSense,ctNew,coords = getSenseFull(truePos,trueCov,mapping,noiseM)
    #Array of measurements; located at the correspondence of the landmark
    for i in range(8):
        for j in range(ztSense[0,:].size):
            if(ctNew[j] == i):
                preTrackSense[:,i] = ztSense[:,j]
            preTrackedCt[:,j] = ctNew[j]    
    trackedSense[0,:,:] = preTrackSense
    trackedCt[0,:,:] = preTrackedCt
    trackedTrue[0,:,:] = truePos
    trackedU[0,:,:] = motion
    
    truePos,covMock = updateMu(truePos,covMock,motion,delta,eParams) 
    ztSense,ctNew,coords = getSenseFull(truePos,trueCov,mapping,noiseM)
    ##############################
    #IMPORTANT
    preTrackSense = np.zeros((2,8))
    preTrackedCt = np.array([[-1,-1,-1,-1,-1,-1,-1,-1]])
    ###############################
    
#     for i in range(8):
#         if(8-ztSense[0,:].size > i):
#             preTrackSense[:,i] = np.zeros((2))
#         else:
#             preTrackSense[:,7-i] = ztSense[:,7-i]

    for i in range(8):
        for j in range(ztSense[0,:].size):
            if(ctNew[j] == i):
                preTrackSense[:,i] = ztSense[:,j]    
            preTrackedCt[:,j] = ctNew[j]

    trackedSense = np.vstack((trackedSense,preTrackSense.reshape((1,2,8))))
    trackedCt = np.vstack((trackedCt,preTrackedCt.reshape((1,1,8))))
    trackedTrue = np.vstack((trackedTrue,truePos.reshape((1,3,1))))
    trackedU = np.vstack((trackedU,motion.reshape((1,2,1))))
    
    #Need a better setup
    ############################################################
    for k in range(18):
        if k == 2:
            motion[1,0] = -motion[1,0]
        if k == 6:
            motion[1,0] = -motion[1,0]
        if k == 8:
            motion[1,0] = -1.5*motion[1,0]
        if k == 15:
            motion[1,0] = 0.8*motion[1,0]
    ###############################################################       
        
        truePos,covMock = updateMu(truePos,covMock,motion,delta,eParams)
        ztSense,ctNew,coords = getSenseFull(truePos,trueCov,mapping,noiseM)
        ##############################
        #IMPORTANT
        preTrackSense = np.zeros((2,8))
        preTrackedCt = np.array([[-1,-1,-1,-1,-1,-1,-1,-1]])
        ###############################     
        for i in range(8):
            for j in range(ztSense[0,:].size):
                if(ctNew[j] == i):
                    preTrackSense[:,i] = ztSense[:,j]    
                preTrackedCt[:,j] = ctNew[j]
                
        trackedSense = np.vstack((trackedSense,preTrackSense.reshape((1,2,8))))
        trackedCt = np.vstack((trackedCt,preTrackedCt.reshape((1,1,8))))
        trackedTrue = np.vstack((trackedTrue,truePos.reshape((1,3,1))))
        trackedU = np.vstack((trackedU,motion.reshape((1,2,1))))
        
    trackedCov = np.diagflat([0.00000000001,0.00000000001,0.00000000001]).reshape(1,3,3)
    for i in range(trackedTrue[:,0,0].size):
        trackedCov = np.vstack((trackedCov,np.diagflat([0.00000000001,0.00000000001,0.00000000001]).reshape(1,3,3)))
    return trackedTrue,trackedCov,trackedSense,trackedCt,trackedU,mapping,noiseM,eParams


# In[ ]:

def compLogLik(mu,mu1,cov,cov1,true,title = 'Log_Lik'):
    sumLog = 0
    loglik = np.zeros((1,1))
    p1 = np.abs(np.linalg.det((2*np.pi*(cov[0,:,:]**2))))**-0.5
    p1 = p1*np.exp(-0.5*np.dot(np.dot((true[0,:,:]-mu[0,:,:]).T,(np.linalg.inv(cov[0,:,:]))),(true[0,:,:]-mu[0,:,:])))
    p2 = np.abs(np.linalg.det((2*np.pi*(cov1[0,:,:]**2))))**-0.5
    p2 = p2*np.exp(-0.5*np.dot(np.dot((true[0,:,:]-mu1[0,:,:]).T,(np.linalg.inv(cov1[0,:,:]))),(true[0,:,:]-mu1[0,:,:])))
        
    loglik[0,0] = np.log(p1/p2)
    for i in range(mu[:,0,0].size-1):
        p1 = np.abs(np.linalg.det((2*np.pi*(cov[int(i)+1,:,:]**2))))**-0.5
        p1 = p1*np.exp(-0.5*np.dot(np.dot((true[int(i)+1,:,:]-mu[int(i)-1,:,:]).T,(np.linalg.inv(cov[int(i)+1,:,:]))),(true[int(i)+1,:,:]-mu[int(i)+1,:,:])))
        p2 = np.abs(np.linalg.det((2*np.pi*(cov1[int(i)+1,:,:]**2))))**-0.5
        p2 = p2*np.exp(-0.5*np.dot(np.dot((true[int(i)+1,:,:]-mu1[int(i)-1,:,:]).T,(np.linalg.inv(cov1[int(i)+1,:,:]))),(true[int(i)+1,:,:]-mu1[int(i)+1,:,:])))
        sumLog = sumLog + np.log(p1/p2)
        loglik = np.vstack((loglik,np.log(p1/p2)))
    
    timeSpace = np.linspace(0,20,20,False)
    plt.plot(timeSpace,loglik.reshape(20,))
    plt.xlabel('Time step')
    plt.ylabel('Log likelihood')
    plt.savefig(title)
    plt.show()
    return sumLog,loglik


# In[ ]:

def functionG (mu,delta,motion):
    theta = mu[2,0]
    temp = motion[0,0]/motion[1,0]
    newMu = mu + np.array([[(-temp*np.sin(theta))+(temp*np.sin(theta+motion[1,0]*delta))],
                               [(temp*np.cos(theta))-(temp*np.cos(theta+motion[1,0]*delta))],
                               [motion[1,0]*delta]]) 
    return newMu


# In[ ]:

def functionH (mapLand,corresp,newMu):
    
    temp = corresp
    q = (mapLand[0,temp] - newMu[0,0])**2 + (mapLand[1,temp]-newMu[1,0])**2
    
    belZt = np.array([[np.sqrt(q)],[np.arctan2(mapLand[1,temp]-newMu[1,0],mapLand[0,temp]- newMu[0,0]) - newMu[2,0]]])
    if(belZt[1,0] > 2*np.pi):
        while(belZt[1,0] > 2*np.pi):
            
            belZt[1,0] = belZt[1,0] - 2*np.pi
    if(belZt[1,0] < -2*np.pi):
        while(belZt[1,0] < -2*np.pi):
            belZt[1,0] = belZt[1,0] + 2*np.pi
            
    return belZt


# In[ ]:

class canonState:
    #At must be (3x3) and bt (3x1)
    def __init__(self,At = 0,bt = 0, noise = 0):
        self.K = np.dot(np.dot(np.vstack((np.vstack((np.identity(At[:,0].size),-At.T)),-np.identity(At[:,0].size).T))
                               ,np.linalg.inv(noise))
                        ,np.vstack((np.vstack((np.identity(At[:,0].size),-At.T)),-np.identity(At[:,0].size).T)).T)
        
        self.h = np.vstack((np.vstack((np.zeros((At[:,0].size,1)),np.zeros((At[:,0].size,1)))),np.zeros((At[:,0].size,1))))
        self.flag = False
        
    def evidenceBT(self,bt = 0):
        temp = np.zeros((6,3))
        for i in range (6):
            for j in range (3):
                temp[i,j] = self.K[i,j+6]
        
        
        self.h = self.h[:-3,0].reshape(6,1) - np.dot(temp,bt)
        self.K = self.K[:-3,:-3]
        self.flag = True
        
        
    def evidenceInitXT(self,Xt = 0):
        if(self.flag == True):
            temp = np.zeros((3,3))
            for i in range(3):
                for j in range(3):
                    temp[i,j] = self.K[i,j+3]
            self.h = self.h[:-3,0].reshape(3,1) - np.dot(temp,Xt)
            self.K = self.K[:-3,:-3]
        else:
            print("Reduce bt first")
            
    def returnMuCov(self):
        
        temp1 = np.linalg.inv(self.K)
        temp2 = np.dot(np.linalg.inv(self.K),self.h)
        
        return temp2,temp1


# In[ ]:

class newCanonMeasure:

    def __init__(self,Ct = 0,dt = 0,zt = 0,noise = 0):
        self.K = np.dot(np.dot(np.vstack((np.vstack((np.identity(zt[:,0].size),-Ct.T)),-np.identity(dt[:,0].size).T))
                               ,np.linalg.inv(noise))
                        ,np.vstack((np.vstack((np.identity(zt[:,0].size),-Ct.T)),-np.identity(dt[:,0].size).T)).T)
        self.h = np.vstack((np.vstack((np.zeros((zt[:,0].size,1)),np.zeros((Ct[0,:].size,1)))),np.zeros((dt[:,0].size,1))))
        self.flag1 = False
        self.flag2 = False
        
    def evidenceDT(self,dt = 0,zt=0,Ct = 0):
        self.flag1 = True
        row = zt[:,0].size + Ct.T[:,0].size
        col = dt[:,0].size

        K12 = np.zeros((row,col))
        for i in range(self.K[:,0].size - row):
            for j in range(self.K[0,:].size - col):
                
                #j refers to the row and i refers to the column
                
                K12[j,i] = self.K[j,i+row]
                
        self.h = self.h[:-col,:]- np.dot(K12,dt)
        self.K = self.K[:-col,:-col]

    def evidenceZT(self,zt = 0,Ct = 0):
        if (self.flag2==True):
            
            row = Ct.T[:,0].size
            col = zt[:,0].size
            K12 = np.zeros((row,col))
            for i in range (self.K[:,0].size - row):
                for j in range (self.K[0,:].size - col):
                    K12[j,i] = self.K[j,i+row]

            self.h = self.h[:-col,:] - np.dot(K12,zt)
            self.K = self.K[:-col,:-col]
            
            
        else:
            print('reduce first')

    def rearrange(self,zt=0,Ct=0):
        if (self.flag1 == True):
            self.flag2 = True
            K11 = np.zeros((zt[:,0].size,zt[:,0].size))
            K12 = np.zeros((zt[:,0].size,Ct.T[:,0].size))
            K21 = np.zeros((Ct.T[:,0].size,zt[:,0].size))
            K22 = np.zeros((Ct.T[:,0].size,Ct.T[:,0].size))
            h1 = np.zeros((zt[:,0].size,1))
            h2 = np.zeros((Ct.T[:,0].size,1))
            
            
            
            #Extraction of individual matrices
            for i in range(self.K[:,0].size - Ct.T[:,0].size):
                for j in range(self.K[0,:].size - Ct.T[:,0].size):
                    
                    K11[i,j] = self.K[i,j]
                    h1[i,0] = self.h[i,0]
            
            for i in range(self.K[:,0].size - Ct.T[:,0].size):
                for j in range(self.K[0,:].size - zt[:,0].size):
                    K12[i,j] = self.K[i,j+zt[:,0].size]
                    
            for i in range(self.K[:,0].size - zt[:,0].size):
                for j in range(self.K[0,:].size - Ct.T[:,0].size):
                    
                    K21[i,j] = self.K[i+zt[:,0].size,j]
                    
            for i in range(self.K[:,0].size - zt[:,0].size):
                for j in range(self.K[0,:].size - zt[:,0].size):
                    
                    K22[i,j] = self.K[i+zt[:,0].size,j+zt[:,0].size]
                    h2[i,0] = self.h[i+zt[:,0].size,0]
                    
            
            
            #Assignment of matrices to correct positions
            for i in range(self.K[:,0].size - zt[:,0].size):
                for j in range(self.K[0,:].size - zt[:,0].size):
                    
                    self.K[i,j] = K22[i,j]
                    self.h[i,0] = h2[i,0]
            
            for i in range(self.K[:,0].size - zt[:,0].size):
                for j in range(self.K[0,:].size - Ct.T[:,0].size):
                    self.K[i,j+Ct.T[:,0].size] = K21[i,j]
                    
            for i in range(self.K[:,0].size - Ct.T[:,0].size):
                for j in range(self.K[0,:].size - zt[:,0].size):
                    
                    self.K[i+Ct.T[:,0].size,j] = K12[i,j]
                    
            for i in range(self.K[:,0].size - Ct.T[:,0].size):
                for j in range(self.K[0,:].size - Ct.T[:,0].size):
                    
                    self.K[i+Ct.T[:,0].size,j+Ct.T[:,0].size] = K11[i,j]
                    self.h[i+Ct.T[:,0].size,0] = h1[i,0]
        else:
            print('reduce first')


# In[ ]:

def canonAdd(can1,can2):
    
    temp1 = can1.K + can2.K
    temp2 = can1.h + can2.h
    
    temp3 = canon(flag = False,dim=3)
    temp3.setKh(temp1,temp2)
    return temp3


# In[ ]:

def canonSub(can1,can2):
    temp1 = can1.K - can2.K
    temp2 = can1.h - can2.h
    
    temp3 = canon(flag = False)
    temp3.setKh(temp1,temp2)
    return temp3


# In[ ]:

def canonExtend1(can):
    temp =np.zeros((3,can.K[0,:].size))
    can.K = np.vstack((can.K,temp))
    temp = np.zeros((can.K[:,0].size,3))
    can.K = np.hstack((can.K,temp))
    
    can.h = np.vstack((can.h,np.zeros((3,1))))
    return can


# In[ ]:

def canonRearrange(can,dim=0,var=0):
    
    tempo = can
    temp1,temp2,temp3,temp4 = np.zeros((3,3)),np.zeros((3,3)),np.zeros((3,3)),np.zeros((3,3))
    temp5,temp6 = np.zeros((3,1)),np.zeros((3,1))
    for i in range(dim):
        for j in range(dim):
            
            temp1[i,j] = tempo.K[i,j]
            temp2[i,j] = tempo.K[i,j+3]
            temp3[i,j] = tempo.K[i+3,j]
            temp4[i,j] = tempo.K[i+3,j+3]
            temp5[i,0] = tempo.h[i,0]
            temp6[i,0] = tempo.h[i+3,0]
        
        
    for i in range(dim):
        for j in range(dim):
            
            can.K[i,j] = temp4[i,j]
            can.K[i,j+3] = temp3[i,j]
            can.K[i+3,j] = temp2[i,j]
            can.K[i+3,j+3] = temp1[i,j]
            can.h[i,0] = temp6[i,0]
            can.h[i+3,0] = temp5[i,0]
            
            
    return can
         


# In[ ]:

class canon:
    def __init__(self,mu = 0,cov = 0,flag = True, dim=3):
       
        if(flag == False):
            self.K = np.zeros((dim,dim))
            self.h = np.zeros((dim,1))
            
        else:
            self.cov = cov
            self.mu = mu
            self.K = np.linalg.inv(cov)
            self.h = np.dot(np.linalg.inv(cov),mu)
        
    def setKh (self,Kt = 0, ht = 0):
        self.K = Kt
        self.h = ht
        
    def returnMuCov(self):
        
        temp1 = np.linalg.inv(self.K)
        temp2 = np.dot(np.linalg.inv(self.K),self.h)
        
        return temp2,temp1


# In[ ]:

def canonReturnMuCov(can):
        
    temp1 = np.linalg.inv(can.K)
    temp2 = np.dot(np.linalg.inv(can.K),can.h)
        
    return temp2,temp1


# In[ ]:

def marginalize(can):
    #Note this function will marginalize over the variable at the bottom of the canonical form distribution 
    K11,K12,K21,K22 = np.zeros((3,3)),np.zeros((3,3)),np.zeros((3,3)),np.zeros((3,3))
    h1,h2 = np.zeros((3,1)),np.zeros((3,1))
    for i in range (3):
        for j in range (3):
            K11[i,j] = can.K[i,j]
            K12[i,j] = can.K[i,j+3]
            K21[i,j] = can.K[i+3,j]
            K22[i,j] = can.K[i+3,j+3]
            h1[i,0] = can.h[i,0]
            h2[i,0] = can.h[i+3,0]
    
    if (np.count_nonzero(K12) != 0 and np.count_nonzero(K21) != 0 and np.count_nonzero(K22) != 0):
    
        k_dash = K11 - np.dot(K12,np.dot(np.linalg.inv(K22),K21))
        h_dash = h1 - np.dot(K12,np.dot(np.linalg.inv(K22),h2))
        
    else:
        
        k_dash = K11
        h_dash = h1
    
    
    temp3 = canon(flag = False)
    temp3.setKh(k_dash,h_dash)
            
            
    return temp3


# In[ ]:

def getCanonG(can,dim):
    muT,covT = can.returnMuCov()
    g1 = (-1/2)*(np.dot(muT.T,np.dot(np.linalg.inv(covT),muT)))
    g2 = -np.log(np.sqrt(np.dot((2*np.pi)**(dim/2),np.linalg.det(covT))))
    g = g1+g2
    return g


# In[ ]:

#ASSUMING A 2-D dataset

#ASSUME timestep delta T = 1

#Localisation Algorithm with known correspondence of landmarks

#New EKF filter process must be run for each time step in ONE loop
#motionControl,measureSense & corresp are 3d datasets; similar to the datasets simulated in simulateData()

def EKF_Localization_with_correspondence_New(initialPos,motionControl,measureSense,corresp,mapLand,delta,eParam,noise):
    
    #Initial assumption of location of robot
    muPrev,covPrev = np.zeros((1,3,1)),np.zeros((1,3,3))
    ekfMu,ekfCov = np.zeros((1,3,1)),np.zeros((1,3,3))
    muPrev[0,:,:] = initialPos
    covPrev[0,:,:] = np.array([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,0.7]])
    
    #T = number of timesteps
    T = motionControl[:,0,0].size
    
    
    for k in range(T):
        
        #Prediction Step
        theta = muPrev[k,2,0]
        temp = motionControl[k,0,0]/motionControl[k,1,0]
        Gt = getGt(muPrev[k,:,:],delta,motionControl[k,:,:])
        Vt = getVt(muPrev[k,:,:],delta,motionControl[k,:,:])
        Mt = getMt(motionControl[k,:,:],eParam)
        newMu = muPrev[k,:,:] + np.array([[(-temp*np.sin(theta))+(temp*np.sin(theta+motionControl[k,1,0]*delta))],
                               [(temp*np.cos(theta))-(temp*np.cos(theta+motionControl[k,1,0]*delta))],
                               [motionControl[k,1,0]*delta]])
        newCov = (np.dot(np.dot(Gt,covPrev[k,:,:]),Gt.T)) + np.dot(np.dot(Vt,Mt),Vt.T)
        
        #Correction Step
        Qt = np.diag(noise.reshape(2,))
        index = 0
        while corresp[k,0,index] !=-1:
            temp = corresp[k,0,index]
            
            q = (mapLand[0,temp]- newMu[0,0])**2 + (mapLand[1,temp]-newMu[1,0])**2
            belZt = np.array([[np.sqrt(q)],[np.arctan2(mapLand[1,temp]-newMu[1,0],mapLand[0,temp]- newMu[0,0]) - newMu[2,0]]]) 
            
            Ht = getHt(newMu,mapLand,temp)
            
            St = np.dot(np.dot(Ht,newCov),Ht.T) + Qt
        
            Kt = np.dot(np.dot(newCov,Ht.T),np.linalg.inv(St))
        
            newMu = newMu + np.dot(Kt,(measureSense[k,:,corresp[k,0,index]].reshape(2,1) - belZt))
            newCov = np.dot((np.identity(newCov[:,0].size) - np.dot(Kt,Ht)),newCov)
            index=index+1
            
        if(k==0):
            
            ekfMu[k,:,:] = newMu
            ekfCov[k,:,:] = newCov
        else:
            ekfMu = np.vstack((ekfMu,newMu.reshape((1,3,1))))
            ekfCov = np.vstack((ekfCov,newCov.reshape((1,3,3))))
            
        muPrev = np.vstack((muPrev,newMu.reshape((1,3,1))))
        covPrev = np.vstack((covPrev,newCov.reshape((1,3,3))))
        muPrev[k+1,:,:],covPrev[k+1,:,:] = updateMu(muPrev[k+1,:,:],covPrev[k+1,:,:],motionControl[k,:,:],1,eParam)
    return ekfMu,ekfCov,muPrev,covPrev
    


# In[ ]:

#ASSUMING A 2-D dataset

#ASSUME timestep delta T = 1

#Localisation Algorithm without correspondence of landmarks

#New EKF filter process must be run for each time step in ONE loop
#motionControl,measureSense are 3d datasets; similar to the datasets simulated in simulateData()

#CHECK FOR ACCURACY

def EKF_Localization_without_correspondence_New(initialPos,motionControl,measureSense,mapLand,delta,eParam,noise):
    
    
    #Initial assumption of location of robot
    muPrev,covPrev = np.zeros((1,3,1)),np.zeros((1,3,3))
    ekfMu,ekfCov = np.zeros((1,3,1)),np.zeros((1,3,3))
    muPrev[0,:,:] = initialPos
    covPrev[0,:,:] = np.array([[1,0,0],[0,1,0],[0,0,0.7]])
    
    #T = number of timesteps
    T = motionControl[:,0,0].size
    
    belZt = np.zeros((1,2,1))
    St = np.zeros((1,2,2))
    Ht = np.zeros((1,2,3))
    flag = 0
    
    for l in range(T):
        
        #Prediction Step
        theta = muPrev[l,2,0]
        temp = motionControl[l,0,0]/motionControl[l,1,0]
        Gt = getGt(muPrev[l,:,:],delta,motionControl[l,:,:])
        Vt = getVt(muPrev[l,:,:],delta,motionControl[l,:,:])
        Mt = getMt(motionControl[l,:,:],eParam)
        newMu = muPrev[l,:,:] + np.array([[-(temp*np.sin(theta))+(temp*np.sin(theta+motionControl[l,1,0]*delta))],
                                   [(temp*np.cos(theta))-(temp*np.cos(theta+motionControl[l,1,0]*delta))],
                                   [motionControl[l,1,0]*delta]])
        newCov = (np.dot(np.dot(Gt,covPrev[l,:,:]),Gt.T)) + np.dot(np.dot(Vt,Mt),Vt.T)
        if(newMu[2,0] > 2*np.pi):
            while(angleE > 2*np.pi):
                newMu[2,0] = newMu[2,0] - 2*np.pi
        if(newMu[2,0] < -2*np.pi):
            while(newMu[2,0] < -2*np.pi):
                newMu[2,0] = newMu[2,0] + 2*np.pi
        #Correction Step
        Qt = np.diag(noise.reshape(2,))
    
        index = 0
        corrCheck = np.zeros(mapLand[0,:].size)
        for i in range(measureSense[l,0,:].size):
            if(measureSense[l,0,i] !=0.0 and measureSense[l,1,i] !=0.0):
                for j in range(mapLand[0,:].size):
                    if flag ==0:
                    
                        q = (mapLand[0,j]- newMu[0,0])**2 + (mapLand[1,j]-newMu[1,0])**2

                        belZt[j,:,:] = np.array([[np.sqrt(q)],[np.arctan2(mapLand[1,j]-newMu[1,0],mapLand[0,j]- newMu[0,0]) - newMu[2,0]]]) 
        
                        Ht[j,:,:] = getHt(newMu,mapLand,j)
        
                        St[j,:,:] = np.dot(np.dot(Ht[j,:,:],newCov),Ht[j,:,:].T) + Qt
            
                        flag = 1
                    else:
                    
                        q = (mapLand[0,j]- newMu[0,0])**2 + (mapLand[1,j]-newMu[1,0])**2
    
                        belZt= np.vstack((belZt,np.array([[np.sqrt(q)]
                                                          ,[np.arctan2(mapLand[1,j]-newMu[1,0],mapLand[0,j]- newMu[0,0]) - newMu[2,0]]]).reshape(1,2,1))) 
        
                        Ht = np.vstack((Ht,getHt(newMu,mapLand,j).reshape(1,2,3)))
        
                        St = np.vstack((St,(np.dot(np.dot(Ht[j,:,:],newCov),Ht[j,:,:].T) + Qt).reshape(1,2,2)))
                flag = 0
                temp = 0
                temp1 = 0
                index=0
                for k in range(mapLand[0,:].size):
                    temp = (np.abs(np.linalg.det(2*np.pi*St[k,:,:]))**-0.5)*np.exp(-0.5*np.dot(np.dot((measureSense[l,:,i].reshape(2,1) - belZt[k,:,:]).T
                                                                                                      ,(np.linalg.inv(St[k,:,:])))
                                                                                               ,(measureSense[l,:,i].reshape(2,1) - belZt[k,:,:])))

                    if(k==0):
                        temp1 = np.log(temp)
                    if np.log(temp) > temp1:
                        temp1 = np.log(temp)
                        index = k 
                        
                Kt = np.dot(np.dot(newCov,Ht[index,:,:].T),np.linalg.inv(St[index,:,:]))
                newMu = newMu + np.dot(Kt,(measureSense[l,:,i].reshape(2,1) - belZt[index,:,:]))
                newCov = np.dot((np.identity(newCov[:,0].size) - np.dot(Kt,Ht[index,:,:])),newCov)
                belZt = np.zeros((1,2,1))
                St = np.zeros((1,2,2))
                Ht = np.zeros((1,2,3))
        if(l==0):
            
            ekfMu[l,:,:] = newMu
            ekfCov[l,:,:] = newCov
        else:
            ekfMu = np.vstack((ekfMu,newMu.reshape((1,3,1))))
            ekfCov = np.vstack((ekfCov,newCov.reshape((1,3,3))))
            
        muPrev = np.vstack((muPrev,newMu.reshape((1,3,1))))
        covPrev = np.vstack((covPrev,newCov.reshape((1,3,3))))
        
        muPrev[l+1,:,:],covPrev[l+1,:,:] = updateMu(muPrev[l+1,:,:],covPrev[l+1,:,:],motionControl[l,:,:],1,eParam)

    return ekfMu,ekfCov,muPrev,covPrev


# In[ ]:

#Need to check for measurement error on initial state


def filterProcess_with_correspondence(initialPos,measureSense,motionControl,mapping,corresp,delta,noiseM,eParam,rightMessagesIn=0,check = False,UC = False):
    
    #Initial assumption of location of robot
    pgmNextState = 0
    pgmMeasureState = 0
    flag0 = True
    flag01 = True
    flag02 = True
    newCorresp = corresp
    muPrev,covPrev = np.zeros((3,1)),np.zeros((3,3))
    pgmMu,pgmCov = np.zeros((1,3,1)),np.zeros((1,3,3))
    pgmMu2,pgmCov2 = np.zeros((1,3,1)),np.zeros((1,3,3))
    muPrev = initialPos
    covPrev = np.array([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1]])
    pgmNextStates = 0
    rightMessIn = canon(flag=False,dim=3)
    rightMess = canon(flag=False,dim=3)
    upMess = canon(flag=False,dim=3)
    downMess = canon(flag=False,dim=3)
    rightMessages = 0
    leftMess = canon(flag=False,dim=3)
#Initial state measurement
#####################################################################################    
    pgmState = canon(muPrev,covPrev)
    leftMess = pgmState
    
    
#########################################################################################################

    #T = number of timesteps
    T = motionControl[:,0,0].size
    
    for k in range(T):
        #reset all messages except the left message
        rightMess = canon(flag=False,dim=3)
        upMess = canon(flag=False,dim=3)
        downMess = canon(flag=False,dim=3)
        if(check==True):
            
            rightMessIn = rightMessagesIn[k]
            
        else:
            rightMessIn = canon(flag=False,dim=3)
        
        #Prediction Step
        
        
        #Using the mean of the previous state to calculate Gt and bt
        Gt = getGt(muPrev,delta,motionControl[k,:,:])
        bg = functionG(muPrev,delta,motionControl[k,:,:])
        At = Gt
        bt = bg - np.dot(Gt,muPrev)

        pgmNextState = canonState(At,bt,covPrev)

        pgmNextState.evidenceBT(bt)

        leftMess = canonExtend1(leftMess)
        leftMess = canonRearrange(leftMess,3)

        rightMessIn = canonExtend1(rightMessIn)
        
        
        inMessages = canonAdd(leftMess,rightMessIn)

        almostMess = canonAdd(pgmNextState,inMessages)
        downMess = marginalize(almostMess)
            
        
        #calculating the mean and covariance of the downwards message - to calculate Ct and dt
        measureMu,measureCov = canonReturnMuCov(downMess)

        convMu,convCov = convert(measureMu,measureCov)
        
        
        #Measurement step - correction step
        
        #For each measurement
        Ct,dt,newNoise,mList,cIndex,newCorrespIndex = calculate_measure_poten(measureMu,measureCov,newCorresp,measureSense,mapping,k,noiseM,UC)
        newCorresp = newCorrespIndex
        pgmMeasureState = newCanonMeasure(Ct,dt,mList,newNoise)

        pgmMeasureState.evidenceDT(dt,mList,Ct)

        pgmMeasureState.rearrange(mList,Ct)

        pgmMeasureState.evidenceZT(mList,Ct)

        #The final cluster potential is a product of all the measurements each with their own evidence; no
        upMess = pgmMeasureState
        onlyMuMess = canon(flag=False,dim=3)
        onlyMuMess.K = upMess.K
#         onlyMuMess.h = downMess.h
        pgmNextState = canonState(At,bt,covPrev)
        pgmNextState.evidenceBT(bt)

#         upMess = canonExtend1(upMess)
        onlyMuMess = canonExtend1(onlyMuMess)
        messageAdd = canonAdd(leftMess,pgmNextState)
        eviMeasure = canonAdd(messageAdd,onlyMuMess)
        rightMess = marginalize(eviMeasure)
        rightMu,rightCov = canonReturnMuCov(rightMess)
            
            
        newMu,newCov = canonReturnMuCov(rightMess)
        checkMu,checkCov = newMu,newCov

        
        if (flag0 == True):
            pgmMu[k,:,:],pgmCov[k,:,:] = checkMu,checkCov
            pgmNextStates = pgmNextState
            flag0 = False
        else:
            pgmMu = np.vstack((pgmMu,checkMu.reshape((1,3,1))))
            pgmCov = np.vstack((pgmCov,checkCov.reshape((1,3,3))))
            pgmNextStates = np.append(pgmNextStates,pgmNextState)
            
        if(flag02 == True):
            rightMessages = rightMess
            flag02 = False
        else:
            rightMessages = np.append(rightMessages,rightMess)
            
        leftMess = canon(checkMu,checkCov)
        muPrev = checkMu
        covPrev = checkCov
        

    pgmMu,pgmCov = calibrateFilter(pgmMu,pgmCov)
    return pgmMu,pgmCov,rightMessages,pgmNextStates,newCorresp


# In[ ]:

def calculate_measure_poten(CMPmu,CMPcov,ct,ztSense,maps,timestep,noise,switchLoc):
    
    index = 0
    Ct,dt = 0,0
    measure = 0
    newCorr = ct
    temp=0
    CMPconvMu,CMPconvCov = convert(CMPmu,CMPcov)
    checko = 0
    UCindex = maximum_likelihood(CMPmu, CMPcov, ztSense, timestep, maps,noise)
    shortUC = 0
    for f in range(ct[timestep,0,:].size-1):
        if(ct[timestep,0,f] != -1):
            shortUC = np.hstack((shortUC,0))
    for h in range(UCindex.size):
        if(UCindex[h] != -1):
            shortUC[checko] = UCindex[h]
            checko=checko+1
    while ct[timestep,0,index] !=-1:
        if(switchLoc == False):
            temp = ct[timestep,0,index]
        else:
            temp = shortUC[index]
        
        Ht = getHt(CMPmu,maps,temp)
        dh = functionH(maps,temp,CMPmu)
        tempDt = dh - CMPconvMu
        
        if(tempDt[1,0] > 2*np.pi):
            while(tempDt[1,0] > 2*np.pi):
                tempDt[1,0] = tempDt[1,0] - 2*np.pi
        if(tempDt[1,0] < -2*np.pi):
            while(tempDt[1,0] < -2*np.pi):
                tempDt[1,0] = tempDt[1,0] + 2*np.pi
                
        if (index == 0):
            Ct = Ht
            dt = tempDt
            measure = ztSense[timestep,:,temp].reshape(2,1)
            
        else:
            Ct = np.vstack((Ct,Ht))
            dt = np.vstack((dt,tempDt)) 
            measure = np.vstack((measure,ztSense[timestep,:,temp].reshape(2,1)))
        if(switchLoc == True):
            newCorr[timestep,0,index] = temp
        
        index = index + 1
        
    final = noise
    for i in range(index-1):
        final = np.hstack((final,noise))
    finale = np.diagflat(final)
    return Ct,dt,finale,measure,index,newCorr
   
    


# In[ ]:

#where do i run this function? simulation or algorithm

def maximum_likelihood (newMu, newCov, measureSense, timestep, mapLand,noiseMeasure):
    belZt = np.zeros((1,2,1))
    St = np.zeros((1,2,2))
    Ht = np.zeros((1,2,3))
    Qt = np.diag(noiseMeasure.reshape(2,))
    indexArray = np.zeros((measureSense[timestep,0,:].size))
    index = 0
    tempVAR = 0
    temp1VAR = 0
    for i in range(measureSense[timestep,0,:].size):
        if(measureSense[timestep,0,i] !=0.0 and measureSense[timestep,1,i] !=0.0):
            for j in range(mapLand[0,:].size):
                if(j == 0):
                    q = (mapLand[0,j]- newMu[0,0])**2 + (mapLand[1,j]-newMu[1,0])**2
                    
                    belZt[j,:,:] = np.array([[np.sqrt(q)]
                                             ,[np.arctan2(mapLand[1,j]-newMu[1,0],mapLand[0,j]- newMu[0,0]) - newMu[2,0]]])
                    
                    Ht[j,:,:] = getHt(newMu,mapLand,j)
                    
                    St[j,:,:] = np.dot(np.dot(Ht[j,:,:],newCov)
                                       ,Ht[j,:,:].T) + Qt
                else:
    
                        belZt= np.vstack((belZt,np.array([[np.sqrt(q)]
                                                          ,[np.arctan2(mapLand[1,j]-newMu[1,0],mapLand[0,j]- newMu[0,0]) - newMu[2,0]]]).reshape(1,2,1))) 
        
                        Ht = np.vstack((Ht,getHt(newMu,mapLand,j).reshape(1,2,3)))
        
                        St = np.vstack((St,(np.dot(np.dot(Ht[j,:,:],newCov),Ht[j,:,:].T) + Qt).reshape(1,2,2)))

            tempVAR = 0
            temp1VAR = 0
            index = 0
            for k in range(mapLand[0,:].size):
                    tempVAR = (np.abs(np.linalg.det(2*np.pi*St[k,:,:]))**-0.5)*np.exp(-0.5*np.dot(np.dot((measureSense[timestep,:,i].reshape(2,1) - belZt[k,:,:]).T
                                                                                                      ,(np.linalg.inv(St[k,:,:])))
                                                                                               ,(measureSense[timestep,:,i].reshape(2,1) - belZt[k,:,:])))
                    if(k==0):
                        temp1VAR = np.log(tempVAR)
                    if np.log(tempVAR) > temp1VAR:
                        temp1VAR = np.log(tempVAR)
                        index = k 
            indexArray[i] = index         
        else:
            indexArray[i] = -1
    return indexArray


# In[ ]:

#right messages form the filtering process come in an array - denoted in this function as left Messages
# For last step ; k =0 ; use initial position as evidence ; no marginalization
def smoothProcess_with_correspondence_new(initMu,pgmPrevStates,leftMessagesIn,measureSense,mapping,corresp,delta,noiseM,eParam):

    
    flag0 = True
    flag01 = True
    flag02 = True
    #T = number of timesteps
    T = pgmPrevStates.size
    pgmMu,pgmCov = np.zeros((1,3,1)),np.zeros((1,3,3))
    pgmMu2,pgmCov2 = np.zeros((1,3,1)),np.zeros((1,3,3))
    leftMess = canon(flag=False,dim=3)
    rightMess = canon(flag=False,dim=3)
    upMess = canon(flag=False,dim=3)
    downMess = canon(flag=False,dim=3)
    leftMessIn = 0
    init = initMu
    initCov = np.array([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,0.7]])

    for k in range(T-1,-1,-1):
        
        #reset all messages except leftIn and right messages
        upMess = canon(flag=False,dim=3)
        downMess = canon(flag=False,dim=3)
        if(k==0):
            leftMessIn = canon(init,initCov) 
        else:
            leftMessIn = leftMessagesIn[k-1]
        
        leftMessIn = canonExtend1(leftMessIn)
        leftMessIn = canonRearrange(leftMessIn,3)

        rightMess = canonExtend1(rightMess)
        downMess = marginalize(canonAdd(pgmPrevStates[k],canonAdd(leftMessIn,rightMess)))
        
        measureMu,measureCov = canonReturnMuCov(downMess)
        #Measurement step - correction step
        
        #For each measurement
        Ct,dt,newNoise,mList,cIndex,dump = calculate_measure_poten(measureMu,measureCov,corresp,measureSense,mapping,k,noiseM,False)
    
        pgmMeasureState = newCanonMeasure(Ct,dt,mList,newNoise)
        pgmMeasureState.evidenceDT(dt,mList,Ct)

        pgmMeasureState.rearrange(mList,Ct)

        pgmMeasureState.evidenceZT(mList,Ct)
        
        upMess = pgmMeasureState
        onlyMuMess1 = canon(flag=False,dim=3)
        onlyMuMess1.K = upMess.K
#         onlyMuMess1.h = downMess.h
#         upMess = canonExtend1(upMess)
        onlyMuMess1 = canonExtend1(onlyMuMess1)
        messageAdd = canonAdd(rightMess,onlyMuMess1)
        eviMeasure = canonAdd(pgmPrevStates[k],messageAdd)
        
        leftMess = marginalize(eviMeasure)
        
        finalState = canonAdd(pgmPrevStates[k],canonAdd(onlyMuMess1,canonAdd(rightMess,leftMessIn)))
        finalState = marginalize(finalState)        
        
        
        finalMu,finalCov = canonReturnMuCov(finalState)
                                
        
        
        
        #reverse of  array
        if (flag0 == True):
            
            pgmMu[k-T+1,:,:],pgmCov[k-T+1,:,:] = finalMu,finalCov
            flag0 = False
        else:
            pgmMu = np.vstack((pgmMu,finalMu.reshape((1,3,1))))
            pgmCov = np.vstack((pgmCov,finalCov.reshape((1,3,3))))
            
            
        rightMess = leftMess

        if(flag02 == True):
            leftMessages = leftMess
            flag02 = False
        else:
            leftMessages = np.append(leftMessages,leftMess)
    
    pgmMu,pgmCov = calibrate(pgmMu,pgmCov)
    return pgmMu,pgmCov,leftMessages


# In[ ]:

def calibrate (checkMu,checkCov):

    temp = np.zeros((checkMu.shape))
    temp2 = np.zeros((checkCov.shape))

    for i in range(checkMu[:,0,0].size):
        temp[i,:,:] = checkMu[i,:,:]
        temp2[i,:,:] = checkCov[i,:,:]
    for i in range(checkMu[:,0,0].size):

        checkMu[i,:,:] = temp[checkMu[:,0,0].size-1-i,:,:]
        checkCov[i,:,:] = temp2[checkCov[:,0,0].size-1-i,:,:]
    
    for k in range(checkMu[:,0,0].size):
        checkMu[k,:,:] = checkMu[k,:,:] + tTrue[k,:,:]
    return checkMu,checkCov
########################################################


# In[ ]:

def calibrateFilter (checkMu,checkCov):
    for k in range(checkMu[:,0,0].size):
        checkMu[k,:,:] = checkMu[k,:,:] + tTrue[k,:,:]
    return checkMu,checkCov


# In[ ]:

def displayResult(init,mu,cov,truePos,motion,mapping, title = 'Figure'):
    for i in range(mu[:,0,0].size):
        
        convMu,convCov = convert(mu[i,:,:],cov[i,:,:])

        a,b = np.linalg.eigvals(convCov)
        val,vec = np.linalg.eig(convCov)
        angleE = np.arctan2((vec[0,1]),(vec[0,0]))
        if(angleE > 2*np.pi):
            while(angleE > 2*np.pi):
                angleE = angleE - 2*np.pi
        if(angleE < -2*np.pi):
            while(angleE < -2*np.pi):
                angleE = angleE + 2*np.pi
        t = np.linspace(0, 2*np.pi, 100)
        plt.plot(mapping[0,:],mapping[1,:], 'r*')
        plt.plot(convMu[0,:],convMu[1,:], 'yo')
        Ellipse = np.array([np.sqrt(np.abs(a))*np.cos(t) , np.sqrt(np.abs(b))*np.sin(t)])

        #2-D rotation matrix
        R_rot = np.array([[np.cos(angleE) , -np.sin(angleE)],[np.sin(angleE) , np.cos(angleE)]])  
     

        #Rotation of matrix
        Ell_rot = np.zeros((2,Ellipse.shape[1]))
        for j in range(Ellipse.shape[1]):
            Ell_rot[:,j] = np.dot(R_rot,Ellipse[:,j])

        #Vector of robot
        theta = mu[i,2,0]
    
        plt.quiver([convMu[0,0]], [convMu[1,0]], [motion[0,0]*np.cos(theta)], [motion[0,0]*np.sin(theta)])
    
        
        #initial position
        plt.plot(init[0,0],init[1,0],'bo')
        
        #Tracking of robot
        #Needs to be corrected
        plt.plot(truePos[i,0,0],truePos[i,1,0],'b^')
        plt.quiver([truePos[i,0,0]], [truePos[i,1,0]], [motion[0,0]*np.cos(truePos[i,2,0])], [motion[0,0]*np.sin(truePos[i,2,0])])
        
        plt.plot( convMu[0,0]+Ell_rot[0,:] , convMu[1,0]+Ell_rot[1,:],'darkorange' )

    plt.grid(color='lightgray',linestyle='--')
    plt.axis([-1,12,-1,12])
    plt.xlabel('X coordinates')
    plt.ylabel('Y coordinates')
    plt.savefig(title)
    plt.show()
    
    
    return


# In[ ]:

def display_combined(init,dispPGMMu,dispPGMCov,dispEKFMu,dispEKFCov,truePos,motion,mapping,title = 'Figure_1'):
    
    for k in range(2):
        for i in range(dispPGMMu[:,0,0].size):
            if k==0:
                convMu,convCov = convert(dispPGMMu[i,:,:],dispPGMCov[i,:,:])
            else:
                convMu,convCov = convert(dispEKFMu[i,:,:],dispEKFCov[i,:,:])
            
            a,b = np.linalg.eigvals(convCov)
            val,vec = np.linalg.eig(convCov)
            angleE = np.arctan2((vec[0,1]),(vec[0,0]))

            t = np.linspace(0, 2*np.pi, 100)
    
            plt.plot(mapping[0,:],mapping[1,:], 'r*')
            
            if(k==0):
                plt.plot(convMu[0,:],convMu[1,:], 'ko')
            else:
                plt.plot(convMu[0,:],convMu[1,:], 'yo')

            Ellipse = np.array([np.sqrt(np.abs(a))*np.cos(t) , np.sqrt(np.abs(b))*np.sin(t)])

            #2-D rotation matrix
            R_rot = np.array([[np.cos(angleE) , -np.sin(angleE)],[np.sin(angleE) , np.cos(angleE)]])  
     

            #Rotation of matrix
            Ell_rot = np.zeros((2,Ellipse.shape[1]))
            for j in range(Ellipse.shape[1]):
                Ell_rot[:,j] = np.dot(R_rot,Ellipse[:,j])

            #Vector of robot
            if(k==0):
                theta = dispPGMMu[i,2,0]
                plt.quiver([convMu[0,0]], [convMu[1,0]], [motion[0,0]*np.cos(theta)], [motion[0,0]*np.sin(theta)])
            else:
                theta = dispEKFMu[i,2,0]
                plt.quiver([convMu[0,0]], [convMu[1,0]], [motion[0,0]*np.cos(theta)], [motion[0,0]*np.sin(theta)])
    
            
            colour = 'blue'
            if(k==0):
                colour = 'darkgreen'
                plt.plot(convMu[0,0]+Ell_rot[0,:] , convMu[1,0]+Ell_rot[1,:],colour )
            else:
                colour = 'darkorange'
                plt.plot(convMu[0,0]+Ell_rot[0,:] , convMu[1,0]+Ell_rot[1,:],colour )
                
            
    
            #Tracking of robot
            #Needs to be corrected
            plt.plot(truePos[i,0,0],truePos[i,1,0],'b^')
            plt.quiver([truePos[i,0,0]], [truePos[i,1,0]], [motion[0,0]*np.cos(truePos[i,2,0])], [motion[0,0]*np.sin(truePos[i,2,0])])
    #initial position
    plt.plot(init[0,0],init[1,0],'bo')
    plt.grid(color='lightgray',linestyle='--')
    plt.axis([-1,12,-1,12])
    plt.xlabel('X coordinates')
    plt.ylabel('Y coordinates')
    plt.savefig(title)
    plt.show()
    
    
    return


# In[ ]:




# In[ ]:

iN = np.array([[0],[7.5],[-0.785398]])
deltaT = 1
tTrue,tTrueCov,tSense,tCt,tU,tMap,tNoise,tEparams = simulateData(iN,deltaT,0.7,0.25)


# In[ ]:

EKFMu,EKFCov,tMu,tCov = EKF_Localization_with_correspondence_New(iN,tU,tSense,tCt,tMap,deltaT,tEparams,tNoise)


# In[ ]:

EKFMu1,EKFCov1,tMu1,tCov1 = EKF_Localization_without_correspondence_New(iN,tU,tSense,tMap,deltaT,tEparams,tNoise)


# In[ ]:

PGMMu,PGMCov,PGMMessages,nextStates,newTCt = filterProcess_with_correspondence(iN,tSense,tU,tMap,tCt,deltaT,tNoise,tEparams,check=False,UC = True)


# In[ ]:

newPGMMu, newPGMCov, newPGMMessages = smoothProcess_with_correspondence_new(iN,nextStates,PGMMessages,tSense,tMap,newTCt,deltaT,tNoise,tEparams)


# In[ ]:

displayResult(iN,newPGMMu,newPGMCov,tTrue,tU,tMap, 'PGM_localisation_wo_1')


# In[ ]:

displayResult(iN,EKFMu,EKFCov,tTrue,tU,tMap, 'EKF_Localisation_1')


# In[ ]:

display_combined(iN,newPGMMu,newPGMCov,EKFMu1,EKFCov1,tTrue,tU,tMap, title= 'Combined_wo')


# In[ ]:

logSumo,ll = compLogLik(newPGMMu,EKFMu1,newPGMCov,EKFCov1,tTrue,'log_lik_comb_wo')


# In[ ]:

logSumo/20


# In[ ]:

logSumo = 0
for nm in range(ll[:,0].size):
    if(nm==1):
        logSumo=logSumo
    else:
        logSumo = logSumo+ll[nm,0]
    
logSumo/ll[:,0].size
        


# In[ ]:

ll


# In[ ]:



