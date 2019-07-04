#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 11:31:46 2019

@author: leonardo
"""

from functions import * 
import Jlearner
import numpy as np
import os

#PARAMETERS
simulation_name="s3"
n=20 #linear number of cells
N=n*n
ksigma=0.1
kcut=5*ksigma
a=1
sparsity=0.1
Nuncorr=N
eta=0.001
gamma=0.05
timesteps=500
s=0

print("Initializing...")

if not os.path.exists(simulation_name):
    os.makedirs(simulation_name)

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


mJA,mJB,mJAB=Network.LearningDynamicsRC(s,eta,gamma,timesteps,a,sparsity,ksigma,kcut,simulation_name)

np.save(simulation_name+"/mJA",mJA)
np.save(simulation_name+"/mJB",mJB)
np.save(simulation_name+"/mJAB",mJAB)

print("End of simulation.")