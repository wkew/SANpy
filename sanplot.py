# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 20:50:25 2022

@author: Will Kew
@email: will.kew@gmail.com
"""


import numpy as np
import matplotlib.pyplot as plt

dataloc = r'D:\Coding\SANpy/'
arr = np.load(dataloc+'test.npy')

# simple plot
fig,ax = plt.subplots()
ax.plot(arr)
ax.set_yscale('log')
ax.set_xscale('log')

# calculate log10 of the array
logy = np.log10(arr)
# generate an explicit x-axis
x = np.arange(len(arr))+1 #so it doesnt complain of log(0)
#log10 of the x-axis 
logx = np.log10(x)


# Calculate derivative of the log(y)?
dy = np.gradient(logy)

#plot again
fig,ax = plt.subplots()
ax.plot(logx,logy,c='k')
#twinx axis so the y-scales dont conflict
ax2 = ax.twinx()
ax2.plot(logx,dy,c='r')


