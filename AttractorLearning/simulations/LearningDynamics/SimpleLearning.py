#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from functions import * 
import Jlearner
import numpy as np
import os

#PARAMETERS
simulation_name="s16"
n=20 #linear number of cells
N=n*n
ksigma=0.5
kcut=5*ksigma
a=1
sparsity=0.5
Nuncorr=N
eta=0.000005
gamma=0.08
timesteps=1000
saveJs=True
normalization="hardL2" #choose betw: hardL2 -- softL2

print("Initializing...")

if not os.path.exists(simulation_name):
    os.makedirs(simulation_name)

save_parameters(simulation_name,N,ksigma,kcut,a,sparsity,Nuncorr,eta,gamma,timesteps,normalization)

Network=Jlearner.Jlearner(n)
Network.build_gridA()
Network.build_gridB(Nuncorr)
np.save(simulation_name+"/gridA",Network.gridA)
np.save(simulation_name+"/gridB",Network.gridB)
Network.buildJA(ksigma,kcut)
Network.buildJB(ksigma,kcut)
np.save(simulation_name+"/JA",Network.JA)
np.save(simulation_name+"/JB",Network.JB)
print("Initialization completed")

if normalization=="hardL2":
    print("Starting learning dynamics with hard L2 normalization")
    mJA,mJB,mJAB=Network.LearningDynamics_L2norm(eta,timesteps,a,sparsity,ksigma,kcut,simulation_name,saveJs)
elif normalization=="softL2":
    print("Starting learning dynamics with soft L2 normalization")
    mJA,mJB,mJAB=Network.LearningDynamics(eta,gamma,timesteps,a,sparsity,ksigma,kcut,simulation_name,saveJs)
    
else:
    print("Invalid value for normalization")
    
np.save(simulation_name+"/mJA",mJA)
np.save(simulation_name+"/mJB",mJB)
np.save(simulation_name+"/mJAB",mJAB)

print("End of simulation.")