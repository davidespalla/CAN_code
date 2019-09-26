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

n=50 #linear number of neurons
sigma=0.5 #linear size del kernel, relative to the environment
sparsity=[0.001,0.002,0.005,0.008,0.01,0.03,0.05,0.08,0.1,0.2,0.3,0.4] #sparsity of the dynamcis
ncells=10  #resolution of attractor dynamics
iterations=10 #number of dynamics starting from the same cell for each dynamics
SimulationName="SparsityStudy3" #number of simulation

print("Beginning simulation")
#CREATE SIMULATION FOLDER
if not os.path.exists(SimulationName):
    os.makedirs(SimulationName)
#SAVE PARAMETERS
np.save(SimulationName+"/Sparsity.npy",np.asarray(sparsity))
f=open(SimulationName+"/Parameters.txt","w+")
f.write("n="+str(n)+"\n")
f.write("xi="+str(sigma)+"\n")
f.write("ncells="+str(ncells)+"\n")
f.write("Iterations per cell="+str(iterations)+"\n")
f.close()

print("Initializing...")
Network=Jlearn.Network(n,sigma) #initializes network with N=n^2 neurons and kernel width=sigma
func=functions.functions(Network.sigma) #imports function module
Network.grid=Network.upf() # assignes each cell a uniform place fiels in the environment
np.save(SimulationName+"/grid.npy",Network.grid) #save grid to file
Network.BuildJ() #Builds connectivity matrix with hebbian rule and kernel with width sigma
np.save(SimulationName+"/J.npy",Network.J) #saves interction matrix

print("Starting simulation")
for i in range(len(sparsity)):
    Vin,Vout=func.attractor_distrib(sparsity[i],Network.J,Network.grid,iterations,ncells)
    np.save(SimulationName+"/InitialConfigurations"+str(i+1)+".npy",Vin)
    np.save(SimulationName+"/FinalConfigurations"+str(i+1)+".npy",Vout)
    print("Analysis done for sparsity: "+str(sparsity[i]))

print("Simulation terminated")

