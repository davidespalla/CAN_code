#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 10:20:02 2019

@author: davide
"""

import scipy.integrate
import scipy.special
import numpy as np
import os

def main():
    #PARAMETERS
    output_folder="GKoutput"
    g=np.asarray([2.1,2.5,3.0,3.5,4.0,4.5])
    s=np.linspace(0,2,21)
    sig=np.linspace(0.01,0.5,10)
    stability=np.zeros((len(g),len(s),len(sig)))
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    np.save(output_folder+"/s.npy",s)
    np.save(output_folder+"/g.npy",g)
    np.save(output_folder+"/sigma.npy",sig)
    
    for i in range(len(g)):
        for j in range(len(s)):
            for k in range(len(sig)):
                stability[i][j][k]=Stability(sig[k],s[j],g[i])
                print("Done step "+str(k+j*len(sig)+i*len(s)*len(sig))+"/"+str(len(g)*len(s)*len(sig)))
    
    np.save(output_folder+"/stability.npy",stability)
    return




def L(r1,r2,sig):
    sig=float(sig)/np.sqrt(2.0)
    return np.exp(-np.power(r1 - r2, 2.) / (2 * np.power(sig, 2.)))

def K(r1,r2,sig):
    return np.exp(-np.power(r1 - r2, 2.) / (2 * np.power(sig, 2.)))

def PHI1(r1,r2,sig):
    x=(r1-r2)/(np.sqrt(2)*sig)
    out=0.5*scipy.special.erf(x)
    return out

def Integrand1(rA,rB,sig,s,w):
    h=s*(PHI1(rB,0.5,sig)-PHI1(rB,-0.5,sig))+w
    if h>0:
        return L(rA,0,sig)*h
    else:
        return 0

def Integrand2(rB,sig,s,w):
    h=s*(PHI1(rB,0.5,sig)-PHI1(rB,-0.5,sig))+w
    if h>0:
        return h
    else:
        return 0

def F1(sig,s,w,g):
    I=scipy.integrate.dblquad(Integrand1,-0.5,0.5,lambda x:-0.5,lambda x:0.5,args=(sig,s,w))[0]
    return 1-g*I

def F2(sig,s,w,g):
    I=scipy.integrate.quad(Integrand2,-0.5,0.5,args=(sig,s,w))[0]
    return g*I
    
def Find_w(sig,s,g):
    w=s
    maxt=1000
    tolerance=0.001
    converged=False
    exponent=0.5
    t=1
    while (not converged and t<maxt):
        eta=1.0/pow(t,exponent)
        x=F2(sig,s,w,g)
        deltaw=x-1
        w=w+eta*deltaw
        t=t+1
        converged= abs(deltaw)<tolerance
        #if t%100==0:
        #    print("step w: "+str(t))
    return w,x

def Stability(sig,s,g):
   w,x=Find_w(sig,s,g)
   return F1(sig,s,w,g)

if __name__ == '__main__':
    main()
