import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from functions import functions as fc
import math


class Network:

#__init__(self,n)
#ambient(self,l)
#computeV(self,currentposition)
#strenght(self,currentposition,eta)
#learn(self,path,eta,A)
#zerostrenght(self)
#hebbstrenght(self)
#random_trajectory(self,t)
#ordered_trajectory(self,t)
#sample_regular_trajectory(self,epochs)
#J_txt(A,l,eta,epochs)
#grid_txt(B,l)

    
    def __init__(self,n,sigma):
        self.n=n #linear dimension
        self.N=n*n #number of neurons
        self.sigma=sigma
        self.grid=np.zeros((pow(n,2),2)) #firing places' grid
        self.J=np.zeros((n*n,n*n)) #hebbian CM 
        self.V=np.zeros(n*n) #activity of the network
        self.func=fc(self.sigma)
    
           
    def gpf(self,x_c,y_c,sigma):
        grid=np.zeros((self.N,2))
        for i in range(math.floor(3*self.N/4)):
            grid[i][0]=np.random.uniform(0,self.L) 
            grid[i][1]=np.random.uniform(0,self.L)
        for i in range(math.ceil(3*self.N/4),self.N):
            condition=True
            while condition:
                grid[i][0]=np.random.normal(x_c,sigma)  
                condition= not (grid[i][0]>0 and grid[i][0]<self.L)
            condition=True
            while condition:
                grid[i][1]=np.random.normal(y_c,sigma)   
                condition= not (grid[i][1]>0 and grid[i][1]<self.L)
        return grid        
    
    def upf(self):
        grid=np.zeros((self.N,2))
        for i in range(self.N):
            grid[i][0]=np.random.uniform(0,1) 
            grid[i][1]=np.random.uniform(0,1)
        return grid
    
    def computeV(self,currentposition): #computes the activity of the network depending on the current position of the external input
        for i in range(self.N):
                self.V[i]=self.func.PBCkernel(self.grid[i],currentposition)
    
    
    def BuildJ(self):
        for i in range(self.N):
            for j in range(self.N):
                x=self.grid[i]
                y=self.grid[j]
                if i!=j:
                    self.J[i][j]=self.func.PBCkernel(x,y)
    
