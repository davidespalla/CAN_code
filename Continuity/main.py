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
sigma=0.2 #linear size del kernel, relative to the environment
sparsity=0.2 #sparsity of the dynamcis
ncells=10  #resolution of attractor dynamics
iterations=1 #number of dynamics starting from the same cell for each dynamics
SimulationName="Simulation1" #number of simulation

print("Beginning simulation")
#CREATE SIMULATION FOLDER
if not os.path.exists(SimulationName):
    os.makedirs(SimulationName)
#SAVE PARAMETERS

print("Initializing...")
Network=Jlearn.Network(n,sigma) #initializes network with N=n^2 neurons and kernel width=sigma
func=functions.functions(Network.sigma) #imports function module
Network.grid=Network.upf() # assignes each cell a uniform place fiels in the environment
np.save(SimulationName+"/grid.npy",Network.grid) #save grid to file
Network.BuildJ() #Builds connectivity matrix with hebbian rule and kernel with width sigma
np.save(SimulationName+"/J.npy",Network.J) #saves interction matrix

print("Starting dynamics")
Vin,Vout=func.attractor_distrib(sparsity,Network.J,Network.grid,iterations,ncells)
np.save(SimulationName+"/InitialConfigurations.npy",Vin)
np.save(SimulationName+"/FinalConfigurations.npy",Vout)

print("Simulation terminated")

