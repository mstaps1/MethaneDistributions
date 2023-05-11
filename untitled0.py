# -*- coding: utf-8 -*-
"""
Created on Fri May  5 13:43:56 2023

@author: marsh
"""

from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

mypath = 'CDF_data'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
permian_files = []
for file_ in onlyfiles:
    #
    if 'J - Permian Fall 2021Permian basin_production' in file_:
        #
        permian_files.append(file_)

# Now import the file into pandas
file_a = pd.read_csv(mypath+'/'+permian_files[0])
# file_a = pd.concat([pd.read_csv(mypath+'/'+permian_files[0]), pd.read_csv(mypath+'/'+permian_files[1])], ignore_index=True, sort=False)

# file_a.concat(pd.read_csv(mypath+'/'+permian_files[1]), ignore_index=True)

# build the sampling algorithm
# file_a.hist(0)

# file_a.hist('Log emission magnitude Rutherford [kgh]')
file_a.hist('Emission magnitude [kgh]')
mean_value = file_a.loc[:, 'Emission magnitude [kgh]'].mean()
#%%

column = 'Emission magnitude [kgh]'
n_bootstrap = 1000
n_samples_max = 3500
step = 20
means = np.zeros((n_samples_max//step, n_bootstrap))
n_samples_range = np.arange(n_samples_max-1,step=step)+1
j = 0
for n_samples in n_samples_range:#range(1, n_samples_max+1, 10):#n_samples_range[3122:]:
    for i in range(n_bootstrap):
        bootstrap_samples = file_a.loc[:, column].sample(n=n_samples).to_numpy()
        means[j, i] = np.mean(bootstrap_samples)
    j += 1
        
#%%

Mean_of_means = np.mean(means[:n_samples//step],axis=1)
p_dev_mean = (means[:n_samples//step] - np.expand_dims(Mean_of_means,axis=1))*np.expand_dims(Mean_of_means,axis=1)*100

#%%

# for sample_ in values:
#     sums.append(sample_.sum())

# plt.plot(np.arange(n_samples_max), means,'r',alpha=.5)
# plt.plot(np.arange(n_samples_max),np.percentile(means,[5,25,50,75,95,99],axis=1).transpose(),'b','-')
# p_dev_mean
ax = plt.gca()
plt.plot(np.arange(999),p_dev_mean,'b', alpha=0.01)
plt.plot(np.arange(999),np.percentile(p_dev_mean,[5],axis=1).transpose(),'k',label='5th %')
plt.plot(np.arange(999),np.percentile(p_dev_mean,[25],axis=1).transpose(),'k',label='25th %')
plt.plot(np.arange(999),np.percentile(p_dev_mean,[75],axis=1).transpose(),'k',label='75th %')
plt.plot(np.arange(999),np.percentile(p_dev_mean,[95],axis=1).transpose(),'k',label='95th %')

ax.legend()
# plt.plot(file_a.loc[:, 'Log emission magnitude Rutherford [kgh]'].mean(),'b','-')
plt.xlabel("n_samples")
plt.ylabel("percent deviation from mean site emissions symlog scale")
ax.set_yscale('symlog')
# plt.ylim(-1000,1000)
plt.show()

#%%
file_a.hist('Emission magnitude [kgh]')
