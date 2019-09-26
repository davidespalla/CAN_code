#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 15:32:25 2019

@author: davide
"""

import functions 
import Jlearn
import numpy as np
import os

def upf(n):
    N=n*n
    grid=np.zeros((N,2))
    for i in range(N):
        grid[i][0]=np.random.uniform(0,1) 
        grid[i][1]=np.random.uniform(0,1)
    return grid



n=40 #linear number of neurons
sigma=[0.01,0.05,0.08,0.1,0.3] #linear size del kernel, relative to the environment
sparsity=[0.001,0.003,0.005,0.008,0.01,0.03,0.05,0.08,0.1,0.2,0.3,0.4] #sparsity of the dynamcis
ncells=2  #resolution of attractor dynamics
iterations=1 #number of dynamics starting from the same cell for each dynamics
SimulationName="Prova" #number of simulation
Qiters=10 #number of quenched re-assignment of place field distribution

print("Beginning simulation")
#CREATE SIMULATION FOLDER
if not os.path.exists(SimulationName):
    os.makedirs(SimulationName)
#SAVE PARAMETERS
np.save(SimulationName+"/Sparsity.npy",np.asarray(sparsity))
np.save(SimulationName+"/xi.npy",np.asarray(sigma))
f=open(SimulationName+"/Parameters.txt","w+")
f.write("n="+str(n)+"\n")
f.write("ncells="+str(ncells)+"\n")
f.write("Iterations per cell="+str(iterations)+"\n")
f.write("Quenched iterations="+str(Qiters)+"\n")
f.close()


for qi in range(Qiters):
    print("Starting quenched iteration: "+str(qi+1))
    Qgrid=upf(n) #defines place fields centers
    #CREATE FOLDER FOR SAVING DIFFERENT QUENCHED REALIZATION
    if not os.path.exists(SimulationName+"/Q"+str(qi+1)):
        os.makedirs(SimulationName+"/Q"+str(qi+1))
    np.save(SimulationName+"/Q"+str(qi+1)+"/grid.npy",Qgrid) #save grid to file
    for i in range(len(sigma)):
        print("Initializing Netowrk with xi="+str(sigma[i]))
        Network=Jlearn.Network(n,sigma[i]) #initializes network with N=n^2 neurons and kernel width=sigma
        func=functions.functions(Network.sigma) #imports function module
        Network.grid=Qgrid # assignes the quenched place fields
        Network.BuildJ() #Builds connectivity matrix with hebbian rule and kernel with width sigma
        #np.save(SimulationName+"/Q"+str(qi+1)+"/J"+str(i+1)+".npy",Network.J) #saves interction matrix
        if not os.path.exists(SimulationName+"/Q"+str(qi+1)+"/xi"+str(i+1)):
            os.makedirs(SimulationName+"/Q"+str(qi+1)+"/xi"+str(i+1))
        print("Starting dynamics with xi: "+str(sigma[i]))
        for j in range(len(sparsity)):
            Vin,Vout=func.attractor_distrib(sparsity[j],Network.J,Network.grid,iterations,ncells)
            np.save(SimulationName+"/Q"+str(qi+1)+"/xi"+str(i+1)+"/InitialConfigurations"+str(j+1)+".npy",Vin)
            np.save(SimulationName+"/Q"+str(qi+1)+"/xi"+str(i+1)+"/FinalConfigurations"+str(j+1)+".npy",Vin)
            print("Analysis done for sparsity: "+str(sparsity[j]))

print("Simulation terminated")

