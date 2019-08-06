#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 11:31:46 2019

@author: Davide
"""

from functions import *
import network
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

#PARAMETERS
simulation_name="sigmaStudy1"
n=30 #linear number of cells
N=n*n
ksigma=[0.1,0.2,0.5]
kcut=5*ksigma
sparsity=ksigma
svalues=np.linspace(0,1,21)


if not os.path.exists(simulation_name):
    os.makedirs(simulation_name)

np.save(simulation_name+"/s_values",svalues)
np.save(simulation_name+"/sigma_values",ksigma)



MA=np.zeros((len(ksigma),len(svalues)))
MB=np.zeros((len(ksigma),len(svalues)))

#Initialize network
Network=network.network(n)
#Build and save gird A
Network.build_gridA()
np.save(simulation_name+"/gridA",Network.gridA)
    
#Build and save gird B
Network.build_gridB(N)
np.save(simulation_name+"/gridB",Network.gridB)

for i in range(len(ksigma)):
    
    #Build and save J
    Network.buildJ(ksigma[i],kcut[i])
    np.save(simulation_name+"/J"+str(i),Network.J)
    
    for j in range(len(svalues)):
        Vin=np.random.uniform(0,1,Network.N)
        Vin=Vin/np.mean(Vin)
        print("Starting dynamics number: "+str(len(svalues)*i+j))
        Network.dynamics(Vin,sparsity[i],svalues[j],ksigma[i],kcut[i])
        #np.save(simulation_name+"/V"+str(len(Nuncorrvalues)*i+j),Network.V)
        mA=Network.calculate_mA(ksigma[i],kcut[i])
        mB=Network.calculate_mB(ksigma[i],kcut[i])
        print("mA="+str(mA)+" mB="+str(mB))
        MA[i][j]=mA
        MB[i][j]=mB
        
np.save(simulation_name+"/MA",MA)
np.save(simulation_name+"/MB",MB)

print("Simulation ended")
