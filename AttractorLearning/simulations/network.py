import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from functions import *
import math

class network:

    def __init__(self,n):
        self.n=n #linear dimension
        
        self.N=n*n #number of neurons
        self.gridA=np.zeros((pow(n,2),2)) #firing places grid A
        self.gridB=np.zeros((pow(n,2),2)) #firing places grid A
        self.J=np.zeros((n*n,n*n)) #connettivity matrix
        self.V=np.zeros(n*n) #activity of the network
        
        
    ### BUILD FUNCTIONS #########################################################################################
    
    def build_gridA(self):
        for i in range(self.n):
            for j in range(self.n):
                self.gridA[self.n*i+j][0]=float(i)/float(self.n)
                self.gridA[self.n*i+j][1]=float(j)/float(self.n)
        return      
    
    def build_gridB(self,Ncorr):
        self.gridB=partialshuffle(self.gridA,Ncorr)
        return      
    
    def buildJ(self,sigma,cutoff):
        for i in range(self.N):
            for j in range(self.N):
                if i != j:
                    self.J[i][j]=(1.0/float(self.N))*Kgauss2D(self.gridA[i],self.gridA[j],sigma,cutoff)
        return
    
    ### READ FUNCTIONS #########################################################################################
    
    def read_gridA(self,fielpath):
       self.gridA=np.load(filepath)
        return      
    
    def read_gridB(self,filepath):
        self.gridB=np.load(filepath)
        return      
    
    def readJ(self,filepath):
        self.J=np.load(filepath)
        return
    
    ### DYNAMICS & OTHER #########################################################################################
    
    def dynamics(self,Vin,sparsity,s,sigma,cutoff):
        #METAPARAMETERS
        maxsteps=50
        a=1
        a2=sparsity
        b=0.01
        tolerance=0.01
        maxiter=1000000
        centerB=[0.5,0.5]
        #########################
        self.V=Vin
        g=1
        h0=0
        for step in range(maxsteps):
            h=np.dot(self.J,self.V)
            for i in range(self.N):
                h[i]=h[i]+s*Kgauss2D(self.gridB[i],centerB,sigma,cutoff)
            self.V=np.asarray(list(map(lambda h: f(h,h0,g),h)))
            h0=fix_parameters(self.V,h,g,h0,a,a2,b,tolerance,maxiter)
            self.V=np.asarray(list(map(lambda h: f(h,h0,g),h)))
            g=a/np.mean(self.V)
            self.V=g*self.V
            #print("Dynamic step: "+str(step)+" done, mean: "+str(np.mean(self.V))+" sparsity: "+str(pow(np.mean(self.V),2)/np.mean(pow(self.V,2))))
        return self.V
        
    def calculate_mA(self,sigma,cutoff):
        mA=0
        meanV=np.mean(self.V)
        for i in range(self.N):
            for j in range(self.N):
                mA=mA+(self.V[i]-meanV)*(self.V[j]-meanV)*Kgauss2D(self.gridA[i],self.gridA[j],sigma,cutoff) 
        mA=np.sqrt(mA/pow(float(self.N),2))
        return mA
        
    def calculate_mB(self,sigma,cutoff):
        mB=0
        meanV=np.mean(self.V)
        for i in range(self.N):
            for j in range(self.N):
                mB=mB+(self.V[i]-meanV)*(self.V[j]-meanV)*Kgauss2D(self.gridB[i],self.gridB[j],sigma,cutoff) 
        mB=np.sqrt(mB/pow(float(self.N),2))
        return mB
    
    
        
