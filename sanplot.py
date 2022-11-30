# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 20:50:25 2022

@author: Will Kew
@email: will.kew@gmail.com
"""


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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



y,x = np.histogram(logy,bins=1000)
x = x[:-1]

from scipy.stats import rayleigh

fit = rayleigh.fit(logy)


fig,ax = plt.subplots()
ax.plot(x,y)
ax.axvline(np.log10(100_000),c='k',ls='--')
ax.axvline(np.log10(624_145),c='k',ls='--')
ax.axvline(np.log10(1_200_000),c='k',ls='--')

ax.axvline(fit[0]-1/3*(fit[1]),c='r',ls='--')
ax.axvline(fit[0]+5/3*(fit[1]),c='r',ls='--')
ax.axvline(fit[0]+7/3*(fit[1]),c='r',ls='--')


noise_thresh = 10**(fit[0]+7/3*(fit[1]))

red_arr = arr[arr<noise_thresh]
np.median(red_arr)
np.std(red_arr)*6


10**(fit[0]+fit[1])

10**fit[1]

'''
dist = np.random.rayleigh(fit[1],100000)
dist.sort()
dist = dist[::-1]
x = np.arange(len(dist))
logx = np.log10(x)
plt.plot(x, dist )

'''

loc = fit[0]
sca = fit[1]

fig,ax = plt.subplots()
ax.plot(x,y)
ax.axvline(sca)

ax.axvline( (loc+np.log(sca)**2 ))


ax.axvline(fit[0]+2*(fit[1]**2))


plt.plot((x/(sca**2))*np.exp((-x**2)/((2*sca)**2)))


from lmfit.models import GaussianModel, SkewedGaussianModel, SkewedVoigtModel


mod = SkewedVoigtModel()

pars = mod.guess(y, x=x)
out = mod.fit(y, pars, x=x)

out.plot()

