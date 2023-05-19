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

# 'A - DJ_Summer_2021Denver basin_production'
# 'B - NorthEast_2021Appalachian basin (eastern overthrust area)_production'
# 'CA 2016San Joaquin basin_production'
# 'Delaware_2019_F2021_boundsPermian basin_production'
Data_set = 'CA 2016San Joaquin basin_production'

for file_ in onlyfiles:
    #
    if Data_set in file_:
        #
        permian_files.append(file_)

# Now import the file into pandas
file_a = pd.read_csv(mypath+'/'+permian_files[1])
# for _ in range():
    
# file_a = pd.concat([pd.read_csv(mypath+'/'+permian_files[0]), pd.read_csv(mypath+'/'+permian_files[1])], ignore_index=True, sort=False)

# file_a.concat(pd.read_csv(mypath+'/'+permian_files[1]), ignore_index=True)

# build the sampling algorithm
# file_a.hist(0)

# file_a.hist('Log emission magnitude Rutherford [kgh]')
model_ = 'Emission magnitude [kgh]'
# model_ = 'Log emission magnitude Rutherford [kgh]'
# file_a.hist('Emission magnitude [kgh]')
mean_value = file_a.loc[:, model_].mean()
#%%

column = model_
n_bootstrap = 1000
n_samples_max = 4000
step = 25
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


percentile_list = [5, 25, 75, 95]
#%%
percentile_list = [5, 25, 75, 95]
x = file_a.loc[:, model_].to_numpy()
fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(20, 2.7*4))

ax1.hist(x,bins=20)
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel('Site Emissions [kg/hr]')
ax1.set_ylabel('number of sites')
ax1.set_title(f'Histogram of emission rates for {Data_set}', fontsize=16)

ax2.plot(n_samples_range[:-1],p_dev_mean,'b', alpha=0.01)
for percentile in percentile_list:
    plt.plot(n_samples_range[:-1],np.percentile(p_dev_mean,[percentile],axis=1).transpose(),'k',label=f'{percentile}th %')
ax2.set_title(f'% dev per n samples for {Data_set}', fontsize=16)
ax2.set_yscale('symlog')
ax2.set_xlabel('n_samples')
ax2.set_ylabel('% deviation from mean site emissions')
ax2.legend(loc='upper right')


#%%

# for sample_ in values:
#     sums.append(sample_.sum())

# plt.plot(np.arange(n_samples_max), means,'r',alpha=.5)
# plt.plot(np.arange(n_samples_max),np.percentile(means,[5,25,50,75,95,99],axis=1).transpose(),'b','-')
# p_dev_mean
ax = plt.gca()
plt.plot(n_samples_range[:-1],p_dev_mean,'b', alpha=0.01)
plt.plot(n_samples_range[:-1],np.percentile(p_dev_mean,[5],axis=1).transpose(),'k',label='5th %')
plt.plot(n_samples_range[:-1],np.percentile(p_dev_mean,[25],axis=1).transpose(),'k',label='25th %')
plt.plot(n_samples_range[:-1],np.percentile(p_dev_mean,[75],axis=1).transpose(),'k',label='75th %')
plt.plot(n_samples_range[:-1],np.percentile(p_dev_mean,[95],axis=1).transpose(),'k',label='95th %')

ax.legend(loc='right')
# plt.plot(file_a.loc[:, 'Log emission magnitude Rutherford [kgh]'].mean(),'b','-')
plt.xlabel("n_samples")
plt.ylabel("percent deviation from mean site emissions symlog scale")
ax.set_yscale('symlog')
# plt.ylim(-1000,1000)
plt.show()

#%%
file_a.hist('Emission magnitude [kgh]')
