import numpy as np
from numpy import linalg as LA
import re
import os
import inspect

class functions:
#__init__(self,sig,c)
#HBkernel(self,r_i,r_r)
#PBCkernel(self,r_i,r_r)
#computeV(mapping,currentposition)
#overlap(matA,matB)
#distance(r_1,r_2)
#transfer(h,th,g)
#positive_mean(v,th)
#fix_parameters(V1,h1,th,a,a2)
#dynamic(position,J,mapping)
#attractor_distrib(self,side,J,grid,iterations,subcells)
#random_trajectory(self,L,n,t)
#ordered_trajectory(self,L,n,t)
#sample_regular_trajectory(self,L,n,epochs)

    
    def __init__(self,sig):
        self.sig=sig

    def HBkernel(self,r_i,r_r):#r_i=position on the grid,r_r=rat position,sig=standard deviation
        V=0.0  
        dx=0
        dy=0
        d=0
        dx=np.abs(r_i[0]-r_r[0])
        dy=np.abs(r_i[1]-r_r[1])
        d=np.sqrt(pow(dx,2)+pow(dy,2))
        V=np.exp(-0.5*pow(d/self.sig,2))
        return V
    
    def PBCkernel(self,r_i,r_r):
        L=100
        periodicity=L/2
        V=0.0  
        dx=0
        dy=0
        d=0
        
        if np.abs(r_i[0]-r_r[0])<periodicity:
            dx=np.abs(r_i[0]-r_r[0])
        else:
            dx=L-np.abs(r_i[0]-r_r[0])
            
        if np.abs(r_i[1]-r_r[1])<periodicity:
            dy=np.abs(r_i[1]-r_r[1])
        else:
            dy=L-np.abs(r_i[1]-r_r[1])
        
        d=np.sqrt(pow(dx,2)+pow(dy,2))
        V=np.exp(-0.5*pow(d/self.sig,2))
        return V
    
    def Sparsify(self,a,f):
        vout=a
        th=np.percentile(a,(1.0-f)*100)
        for i in range(len(a)):
            if vout[i]<th:
                vout[i]=0
        return vout
    
    def transfer(self,h):
        if h>0:
            return h
        else:
            return 0
    
    def computeV(self,mapping,currentposition): #computes the activity of the network depending on the current position of the external input
        N=len(mapping)    
        V=np.zeros(N)
        for i in range(N):
            V[i]=self.PBCkernel(mapping[i],currentposition)
        return V

    def overlap(self,matA,matB):
        m=0
        for i in range(matA.shape[0]):
            for j in range(matA.shape[0]):
                m=m+matA[i][j]*matB[i][j]
        m=m/float(LA.norm(matA)*LA.norm(matB))
        return m

    def distance(self,r_1,r_2):
        dx=0
        dy=0
        d=0
        dx=np.abs(r_1[0]-r_2[0])
        dy=np.abs(r_1[1]-r_2[1])
        d=np.sqrt(pow(dx,2)+pow(dy,2))
        return d
        
    
    

    def dynamic(self,f,position,J,grid): #evolves the activity with J starting from a bump in current position
        #parameters
        N=len(grid)
        maxsteps=50
        #initialization

        V=self.computeV(grid,position)
        Vin=self.Sparsify(V,f)
        Vin=Vin/np.mean(Vin)
        for step in range(maxsteps):
            h=np.dot(J,V)
            V=np.asarray(list(map(lambda h: self.transfer(h),h)))
            V=self.Sparsify(V,f)
            V=V/np.mean(V)
            #print("Dynamic step: "+str(step)+" done, mean: "+str(np.mean(V))+" sparsity: "+str(pow(np.mean(V),2)/np.mean(pow(V,2))))
        return Vin,V
        
    
        
    def attractor_distrib(self,sparsity,J,grid,iterations,ncells):
        N=len(grid)
        spacing=1.0/(float(ncells))
        xcenters=np.linspace(spacing/2.0,1.0-spacing/2.0,ncells)
        ycenters=np.linspace(spacing/2.0,1.0-spacing/2.0,ncells)
        
        VinMat=np.zeros((ncells,ncells,iterations,N))  #matrix for storing initial activityes
        VoutMat=np.zeros((ncells,ncells,iterations,N)) #matrix for storing final activities
        for j in range(ncells):	     	
            for i in range(ncells):
                for t in range(iterations):
                       InitialLocation=np.zeros(2)
                       InitialLocation[0]=np.random.uniform(xcenters[i]-spacing/2.0,xcenters[i]+spacing/2.0)
                       InitialLocation[1]=np.random.uniform(ycenters[j]-spacing/2.0,ycenters[j]+spacing/2.0)
                       Vin,Vfin=self.dynamic(sparsity,InitialLocation,J,grid)
                       VinMat[i][j][t]=Vin
                       VoutMat[i][j][t]=Vfin
                print("Cell ("+str(i)+","+str(j)+") calculated")
        return VinMat,VoutMat

    


