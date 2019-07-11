#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 14:40:40 2019

@author: davide
"""
################### COMPLETE RECURRENT DYNAMICS #################################

from functions import * 
import Jlearner
import numpy as np
import os

#PARAMETERS
simulation_name="s21"
n=20 #linear number of cells
N=n*n
ksigma=0.1
kcut=5*ksigma
a=1
sparsity=0.1
Nuncorr=N
eta=0.00001
timesteps=1000
t1=10 #number of step of input persistence, must divide timesteps
s=1
saveJs=True
normalization="hardL2" #choose betw: hardL2 -- softL2

print("Initializing...")

if not os.path.exists(simulation_name):
    os.makedirs(simulation_name)

save_parametersRC(simulation_name,N,ksigma,kcut,a,sparsity,Nuncorr,eta,timesteps,t1,s,normalization)

Network=Jlearner.Jlearner(n)
Network.build_gridA()
Network.build_gridB(Nuncorr)
np.save(simulation_name+"/gridA",Network.gridA)
np.save(simulation_name+"/gridB",Network.gridB)
Network.buildJA(ksigma,kcut)
Network.buildJB(ksigma,kcut)
np.save(simulation_name+"/JA",Network.JA)
np.save(simulation_name+"/JB",Network.JB)

if timesteps % t1 !=0:
    print("Invalid t1 value: please enter a number that divides the vairable timesteps")
print("Initialization completed")

print("Starting learning dynamics with recurrent connections")
mJA,mJB,mJAB=Network.RecurrentDynamics(s,eta,t1,timesteps,a,sparsity,ksigma,kcut,simulation_name,saveJs)
    
np.save(simulation_name+"/mJA",mJA)
np.save(simulation_name+"/mJB",mJB)
np.save(simulation_name+"/mJAB",mJAB)

print("End of simulation.")
