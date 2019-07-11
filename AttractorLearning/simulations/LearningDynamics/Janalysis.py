#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from functions import * 
import network
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
s=0
ndynamics=10

#steps=[0,20,40,60,80,100,120,140,160,180,200]
steps=[999]
print("Initializing...")

if not os.path.exists(simulation_name+"/Janalysis"):
    os.makedirs(simulation_name+"/Janalysis")

np.save(simulation_name+"/Janalysis/timesteps",steps)

Network=network.network(n)
Network.read_gridA(simulation_name+"/gridA.npy")
Network.read_gridB(simulation_name+"/gridB.npy")
MA=np.zeros((len(steps),ndynamics))
MB=np.zeros((len(steps),ndynamics))

print("Initialization completed")

for step in range(len(steps)):
    print("Analyzing dynamics produced by J"+str(steps[step]))
    Network.readJ(simulation_name+"/J"+str(steps[step])+".npy")
    for i in range(ndynamics):
        print("Starting dynamics number: "+str(i+1))
        center=[0.5,0.5]
        if i%2==0:
            print("Initial conditions correlated with A")
            Vin=np.zeros(N)
            for v in range(N):
                Vin[v]=Kgauss2D(center,Network.gridA[v],ksigma,kcut)
            Vin=Vin/np.mean(Vin)
        else:
            print("Initial conditions correlated with B")
            Vin=np.zeros(N)
            for v in range(N):
                Vin[v]=Kgauss2D(center,Network.gridB[v],ksigma,kcut)
            Vin=Vin/np.mean(Vin)
        Network.dynamics(Vin,sparsity,s,ksigma,kcut)
        mA=Network.calculate_mA(ksigma,kcut)
        mB=Network.calculate_mB(ksigma,kcut)
        print("mA="+str(mA)+" mB="+str(mB))
        np.save(simulation_name+"/Janalysis/V"+str(i),Network.V)
        MA[step][i]=mA
        MB[step][i]=mB

np.save(simulation_name+"/Janalysis/MA",MA)
np.save(simulation_name+"/Janalysis/MB",MB)



print("End of simulation.")