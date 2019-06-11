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
        return
    
    def dynamics(Vin,sparsity,s):
        return
        
    def calculate_mA(self):
        return
        
    def calculate_mB(self):
        return
        
