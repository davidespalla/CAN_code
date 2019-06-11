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
n=50 #linear number of cells
ksigma=0.1
kcut=5*ksigma


Network=network.network(10)
Network.build_gridA()
Network.build_gridB(50)

