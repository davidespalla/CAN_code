import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from functions import *
import math
import copy

class Jlearner:

    def __init__(self,n):
        self.n=n
        self.N=n*n #number of neurons
        self.gridA=np.zeros((pow(n,2),2)) #firing places' grid
        self.gridA=np.zeros((pow(n,2),2)) #firing places' grid
        self.J=np.zeros((n*n,n*n)) #connettivity matrix
        self.JA=np.zeros((n*n,n*n)) #hebbian J for map A
        self.JB=np.zeros((n*n,n*n))
        self.h=np.zeros(n*n)
        self.V=np.zeros(n*n) #activity of the network
        self.h0=0
        self.g=1

    def build_gridA(self):
        for i in range(self.n):
            for j in range(self.n):
                self.gridA[self.n*i+j][0]=float(i)/float(self.n)
                self.gridA[self.n*i+j][1]=float(j)/float(self.n)
        return      
    
    def build_gridB(self,Ncorr):
        self.gridB=partialshuffle(self.gridA,Ncorr)
        return         
    
    def buildJA(self,sigma,cutoff):
        for i in range(self.N):
            for j in range(self.N):
                if i != j:
                    self.JA[i][j]=Kgauss2D(self.gridA[i],self.gridA[j],sigma,cutoff)
        #self.JA=self.JA/LA.norm(self.JA)
        return
        
    def buildJB(self,sigma,cutoff):
        for i in range(self.N):
            for j in range(self.N):
                if i != j:
                    self.JB[i][j]=Kgauss2D(self.gridB[i],self.gridB[j],sigma,cutoff)
        #self.JB=self.JB/LA.norm(self.JB)
        return
                    
    def computeV(self,r,a,a2,sigma,cutoff): 
        maxsteps=50
        b=0.01
        tolerance=0.01
        maxiter=1000000
        for i in range(self.N):
                self.h[i]=Kgauss2D(self.gridB[i],r,sigma,cutoff)
        self.V=np.asarray(list(map(lambda h: f(h,self.h0,self.g),self.h)))
        self.h0=fix_parameters(self.V,self.h,self.g,self.h0,a,a2,b,tolerance,maxiter)
        self.V=np.asarray(list(map(lambda h: f(h,self.h0,self.g),self.h)))
        self.g=a/np.mean(self.V)
        self.V=self.g*self.V
        return
    
    def updateJ(self,eta,gamma): 
        #meanV=np.mean(self.V)
        for i in range(self.N):
            for j in range(self.N):
                if j!=i:
                    self.J[i][j]=self.J[i][j]+eta*(self.V[i])*(self.V[j])-gamma*self.J[i][j]
                else:
                    self.J[i][j]=0
                    
                if self.J[i][j]<0:
                    self.J[i][j]=0
        #self.J=self.J/LA.norm(self.J)
        return
            
    def LearningDynamics(self,eta,gamma,nepochs,a,a2,sigma,cutoff,simulation_name): #total learning of the CM in t steps of a path
        self.J=copy.deepcopy(self.JA)
        labels=np.asarray(range(self.N))
        print("initial ovelaps: mJA="+str(overlap(self.J,self.JA))+"  mJB="+str(overlap(self.J,self.JB))+"  mJAB="+str(overlap(self.J,self.JB+self.JA)))
        mJA=np.zeros(self.N)
        mJB=np.zeros(self.N)
        mJAB=np.zeros(self.N)
        for epoch in range(nepochs):
            np.random.shuffle(labels)
            for t in range(self.N):
                r=self.gridB[labels[t]]
                self.computeV(r,a,a2,sigma,cutoff) 
                self.updateJ(eta,gamma) 
                mJA[t]=overlap(self.J,self.JA)
                mJB[t]=overlap(self.J,self.JB)
                mJAB[t]=overlap(self.J,self.JB+self.JA)
                np.save(simulation_name+"/J"+str(t),self.J)
                print("Ovelaps: mJA="+str(overlap(self.J,self.JA))+"  mJB="+str(overlap(self.J,self.JB))+"  mJAB="+str(overlap(self.J,self.JB+self.JA)))
            print("Epoch "+str(epoch)+" completed.")
            
            
        return mJA,mJB,mJAB
           
            
    
       
        
    
        

#   
