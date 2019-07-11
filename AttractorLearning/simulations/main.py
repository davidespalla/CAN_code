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
simulation_name="s1"
n=20 #linear number of cells
N=n*n
ksigma=0.1
kcut=5*ksigma
sparsity=0.1
svalues=np.linspace(0,1,21)
Nuncorrvalues=np.linspace(0,n*n,11)


if not os.path.exists(simulation_name):
    os.makedirs(simulation_name)

np.save(simulation_name+"/s_values",svalues)
np.save(simulation_name+"/Nuncorr_values",Nuncorrvalues)

Network=network.network(n)
Network.build_gridA()
np.save(simulation_name+"/gridA",Network.gridA)
Network.buildJ(ksigma,kcut)
np.save(simulation_name+"/J",Network.J)

MA=np.zeros((len(Nuncorrvalues),len(svalues)))
MB=np.zeros((len(Nuncorrvalues),len(svalues)))


for i in range(len(Nuncorrvalues)):
     Network.build_gridB(int(Nuncorrvalues[i]))
     np.save(simulation_name+"/gridB"+str(i),Network.gridB)
     for j in range(len(svalues)):
        Vin=np.random.uniform(0,1,Network.N)
        Vin=Vin/np.mean(Vin)
        print("Starting dynamics number: "+str(len(svalues)*i+j))
        Network.dynamics(Vin,sparsity,svalues[j],ksigma,kcut)
        np.save(simulation_name+"/V"+str(len(Nuncorrvalues)*i+j),Network.V)
        mA=Network.calculate_mA(ksigma,kcut)
        mB=Network.calculate_mB(ksigma,kcut)
        print("mA="+str(mA)+" mB="+str(mB))
        MA[i][j]=mA
        MB[i][j]=mB
        
np.save(simulation_name+"/MA",MA)
np.save(simulation_name+"/MB",MB)

print("Simulation ended")
