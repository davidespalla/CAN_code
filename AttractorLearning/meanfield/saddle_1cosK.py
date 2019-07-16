#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 15:04:31 2019

@author: davide
"""

import scipy.integrate
import numpy as np
import math
import os



def main():
    #PARAMETERS
    output_folder="1cosKoutput"
    eps=0 
    m1zero=0.5
    m2zero=0
    g=np.asarray([2.1,2.5,3.0,3.5,4.0,4.5])
    s=np.linspace(0,1,21)
    m1v=np.zeros((len(g),len(s)))
    m2v=np.zeros((len(g),len(s)))
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    np.save(output_folder+"/s.npy",s)
    np.save(output_folder+"/g.npy",g)
    
    for i in range(len(g)):
        for j in range(len(s)):
            print("Calgulating g: "+str(i+1)+"/"+str(len(g))+" and s: "+str(j+1)+"/"+str(len(s)))
            m1,m2,x=FindSaddlePoint(m1zero,m2zero,eps,s[j],g[i])
            print("m1="+str(m1)+" m2="+str(m2)+" x="+str(x))
        m1v[i][j]=m1
        m2v[i][j]=m2
    
    np.save(output_folder+"/mA.npy",m1v)
    np.save(output_folder+"/mB.npy",m2v)
    print("Integration Completed")
    
    return
    
    

def m1Integrand(theta1,theta2,m1,m2,eps,s,w):
    h=m1*np.cos(theta1)+(eps*m2+s)*np.cos(theta2)-w
    if h>0:
        return np.cos(theta1)*h
    else:
        return 0

def m2Integrand(theta1,theta2,m1,m2,eps,s,w):
    h=m1*np.cos(theta1)+(eps*m2+s)*np.cos(theta2)-w
    if h>0:
        return np.cos(theta2)*h
    else:
        return 0
    
def xIntegrand(theta1,theta2,m1,m2,eps,s,w):
    h=m1*np.cos(theta1)+(eps*m2+s)*np.cos(theta2)-w
    if h>0:
        return h
    else:
        return 0

def Fm1(m1,m2,eps,s,w,g):
    I=(scipy.integrate.dblquad(m1Integrand,0,(2*math.pi),lambda x:0,lambda x:2*math.pi,args=(m1,m2,eps,s,w)))[0]
    c=g/pow(2*math.pi,2)
    return c*I
    
def Fm2(m1,m2,eps,s,w,g):
    I=(scipy.integrate.dblquad(m2Integrand,0,(2*math.pi),lambda x:0,lambda x:2*math.pi,args=(m1,m2,eps,s,w)))[0]
    c=g/pow(2*math.pi,2)
    return c*I

def Fix_mean(m1,m2,eps,s,g):
    w=m1+eps*m2+s
    maxt=1000
    tolerance=0.001
    converged=False
    exponent=0.5
    t=1
    while (not converged and t<maxt) :
        eta=1.0/pow(t,exponent)
        x=g/pow(2*math.pi,2)*(scipy.integrate.dblquad(xIntegrand,0,(2*math.pi),lambda x:0,lambda x:2*math.pi,args=(m1,m2,eps,s,w)))[0]
        deltaw=x-1
        w=w+eta*deltaw
        t=t+1
        converged= abs(deltaw)<tolerance
        #if t%100==0:
        #    print("step w: "+str(t))
    return w,x

def FindSaddlePoint(m1,m2,eps,s,g):
    t=1
    tmax=1000
    exponent=0.5
    converged=False
    tolerance=0.001
    
    while (not converged and t<tmax):
        eta=1.0/pow(t,exponent)
        
        m1_old=m1
        m2_old=m2
        
        w,x=Fix_mean(m1,m2,eps,s,g)
        m1=(1.0-eta)*m1_old + eta*Fm1(m1_old,m2_old,eps,s,w,g)
        m2=(1.0-eta)*m2_old + eta*Fm2(m1_old,m2_old,eps,s,w,g)
        
        t=t+1
        dist=np.sqrt(pow(m1-m1_old,2)+pow(m2-m2_old,2))
        converged=(dist<tolerance)
        if t%100==0:
            print("step: "+str(t))
        
    return m1,m2,x
      

if __name__ == '__main__':
    main()