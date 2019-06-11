import numpy as np
from numpy import linalg as LA
import re
import os
import inspect
import copy
import random


def partialshuffle(inp,n):
    #randomly selects n position in the list
    indexes=list(range(len(inp)))
    random.shuffle(indexes)
    args=indexes[:n]
    #defines copy of input and 
    out= copy.deepcopy(inp)
    c = copy.deepcopy(args)
    #generate target position by shuffling copy of args
    random.shuffle(c)
    for i in range(len(args)):
        out[args[i]]=inp[c[i]]
    return out


def f(h,th,g):
    if (h-th)>0:
        return g*(h-th)
    else:
        return 0.0


def Kgauss2D(r_i,r_j,sigma,cutoff):
        if np.abs(r_i[0]-r_j[0])<0.5:
            dx=np.abs(r_i[0]-r_j[0])
        else:
            dx=1-np.abs(r_i[0]-r_j[0])
            
        if np.abs(r_i[1]-r_j[1])<0.5:
            dy=np.abs(r_i[1]-r_j[1])
        else:
            dy=1-np.abs(r_i[1]-r_j[1])
        
        d=np.sqrt(pow(dx,2)+pow(dy,2))
        
        if d<=cutoff:
            out=np.exp(-(pow(d,2)/(2*pow(sigma,2))))
        else:
            out=0    
        return out