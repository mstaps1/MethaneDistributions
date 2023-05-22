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

production_areas = ['A - DJ_Summer_2021Denver basin_production',
                    'B - NorthEast_2021Appalachian basin (eastern overthrust area)_production',
                    'CA 2016San Joaquin basin_production',
                    'Delaware_2019_F2021_boundsPermian basin_production',
                    'Permian_2021Permian basin_production',
                    'F - GAO_2020Permian basin_production',
                    'G - CA_2020San Joaquin basin_production',
                    'H - COVID_CA_2020San Joaquin basin_production',
                    'I - DJ Fall 2021Denver basin_production',
                    'J - Permian Fall 2021Permian basin_production',
                    'F - GAO_2020Uinta basin_production']

# 'K - CA Fall 2021San Joaquin basin_production'
# 'Kairos BarnettFort Worth basin_production'


Data_set = production_areas[-1]

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
step = 50

means = np.zeros((n_samples_max//step, n_bootstrap))
mdl_array_measured = np.zeros((n_samples_max//step, n_bootstrap))
mdl_array_unmeasured = np.zeros((n_samples_max//step, n_bootstrap))
total_emissions_bootstap = np.zeros((n_samples_max//step, n_bootstrap))

n_samples_range = np.arange(n_samples_max-1,step=step)+1
j = 0
mdl = 1

for n_samples in n_samples_range:#range(1, n_samples_max+1, 10):#n_samples_range[3122:]:
    for i in range(n_bootstrap):
        #
        bootstrap_samples = file_a.loc[:, column].sample(n=n_samples).to_numpy()
        
        #
        if model_ == 'Log emission magnitude Rutherford [kgh]':
            mdl_mask = bootstrap_samples>np.log10(mdl)
            mdl_array_measured[j, i] = np.sum(10**bootstrap_samples[mdl_mask])
            mdl_array_unmeasured[j, i] = np.sum(10**bootstrap_samples[~mdl_mask])
            total_emissions_bootstap[j, i] = np.sum(10**bootstrap_samples)
        elif model_ == 'Emission magnitude [kgh]':
            mdl_mask = bootstrap_samples>mdl
            mdl_array_measured[j, i] = np.sum(bootstrap_samples[mdl_mask])
            mdl_array_unmeasured[j, i] = np.sum(bootstrap_samples[~mdl_mask])
            total_emissions_bootstap[j, i] = np.sum(bootstrap_samples)
        #
        means[j, i] = np.mean(bootstrap_samples)
        
    j += 1
#%%    

# n_zero_sample = np.nonzero(total_emissions_bootstap)
# # Calculate the percent measured vs unmeasured
# percent_measured = (mdl_array_measured[n_zero_sample] - total_emissions_bootstap[n_zero_sample])/total_emissions_bootstap[n_zero_sample]*100
# percent_unmeasured = (mdl_array_measured[n_zero_sample] - total_emissions_bootstap[n_zero_sample])/total_emissions_bootstap[n_zero_sample]*100

# n_zero_sample = np.nonzero(total_emissions_bootstap)
# Calculate the percent measured vs unmeasured
percent_measured = -(mdl_array_measured - total_emissions_bootstap)/total_emissions_bootstap*100
percent_unmeasured = -(mdl_array_measured - total_emissions_bootstap)/total_emissions_bootstap*100




#%%
fig, ax0 = plt.subplots(1, 1,figsize=(20, 2.7*4))
ax0.plot(n_samples_range,percent_unmeasured)
# ax0.set_yscale('symlog')



#%%



fig, ax0 = plt.subplots(1, 1,figsize=(20, 2.7*4))
# ax0.plot(n_samples_range,mdl_array_measured)
ax0.set_title(f'Percentage of total emissions Measurable based on a MDL of {mdl} kg/hr', fontsize=16)
ax0.fill_between(n_samples_range,
                 np.amin(percent_measured,axis=1),
                 np.amax(percent_measured,axis=1),
                 alpha = 0.75,
                 label='min value to max value')

ax0.fill_between(n_samples_range,
                 np.percentile(percent_measured,[5],axis=1).flatten(),
                 np.percentile(percent_measured,[95],axis=1).flatten(),
                 alpha=0.75,
                 label='5th -95th percentile')

ax0.fill_between(n_samples_range,
                 np.percentile(percent_measured,[25],axis=1).flatten(),
                 np.percentile(percent_measured,[75],axis=1).flatten(),
                 alpha=0.75,
                 label='25th -75th percentile')
ax0.legend(loc='upper right')
    


# ax0.fill_between(n_samples_range, np.amin(mdl_array,axis=1), np.amax(mdl_array,axis=1),alpha = 0.5, label='min value to max value')


#%%

Mean_of_means = np.mean(means[:n_samples//step],axis=1)
p_dev_mean = (means[:n_samples//step] - np.expand_dims(Mean_of_means,axis=1))*np.expand_dims(Mean_of_means,axis=1)*100
# p_dev_mean = (means[:n_samples//step] -mean_value)*mean_value*100

percentile_list = [5, 25, 75, 95]
#%%
if model_ == 'Emission magnitude [kgh]':
    percentile_list = [5, 25, 75, 95]
    x = file_a.loc[:, model_].to_numpy()
    fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(20, 2.7*4))
    
    ax1.hist(x,bins=20)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Site Emissions [kg/hr]')
    ax1.set_ylabel('number of sites')
    ax1.set_ylim([None, 5*10**4])
    ax1.set_title(f'Histogram of emission rates for {Data_set}', fontsize=16)
    
    # ax2.plot(n_samples_range[:-1],p_dev_mean,'b', alpha=0.01)
    # for percentile in percentile_list:
    #     plt.plot(n_samples_range[:-1],np.percentile(p_dev_mean,[percentile],axis=1).transpose(),'k',label=f'{percentile}th %')
    ax2.fill_between(n_samples_range[:-1], np.amin(p_dev_mean,axis=1), np.amax(p_dev_mean,axis=1),alpha = 0.5, label='min value to max value')
    ax2.fill_between(n_samples_range[:-1], np.percentile(p_dev_mean,[5],axis=1).flatten(), np.percentile(p_dev_mean,[95],axis=1).flatten(),alpha=0.2,label='5th -95th percentile')
    ax2.fill_between(n_samples_range[:-1], np.percentile(p_dev_mean,[25],axis=1).flatten(), np.percentile(p_dev_mean,[75],axis=1).flatten(),alpha=0.4,label='25th -75th percentile')
    ax2.set_title(f'% dev per n samples for {Data_set}', fontsize=16)
    # ax2.set_yscale('symlog')
    ax2.set_xlabel('n_samples')
    ax2.set_ylabel('% deviation from mean site emissions')
    ax2.legend(loc='upper right')
    ax2.set_yscale('symlog')
    ax1.set_ylim([None, 5*10**5])
else:
    x = file_a.loc[:, model_].to_numpy()
    fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(20, 2.7*4))
    
    ax1.hist(x,bins=20)
    # ax1.set_xscale('log')
    # ax1.set_yscale('log')
    ax1.set_xlabel('Site Emissions [kg/hr]')
    ax1.set_ylabel('number of sites')
    # ax1.set_ylim([None, 5*10**4])
    ax1.set_title(f'Histogram of emission rates for {Data_set}', fontsize=16)
    ax2.fill_between(n_samples_range[:-1], np.amin(p_dev_mean,axis=1), np.amax(p_dev_mean,axis=1),alpha = 0.5, label='min value to max value')

# #%%

# # for sample_ in values:
# #     sums.append(sample_.sum())

# # plt.plot(np.arange(n_samples_max), means,'r',alpha=.5)
# # plt.plot(np.arange(n_samples_max),np.percentile(means,[5,25,50,75,95,99],axis=1).transpose(),'b','-')
# # p_dev_mean
# ax = plt.gca()
# plt.plot(n_samples_range[:-1],p_dev_mean,'b', alpha=0.01)
# plt.plot(n_samples_range[:-1],np.percentile(p_dev_mean,[5],axis=1).transpose(),'k',label='5th %')
# plt.plot(n_samples_range[:-1],np.percentile(p_dev_mean,[25],axis=1).transpose(),'k',label='25th %')
# plt.plot(n_samples_range[:-1],np.percentile(p_dev_mean,[75],axis=1).transpose(),'k',label='75th %')
# plt.plot(n_samples_range[:-1],np.percentile(p_dev_mean,[95],axis=1).transpose(),'k',label='95th %')

# ax.legend(loc='right')
# # plt.plot(file_a.loc[:, 'Log emission magnitude Rutherford [kgh]'].mean(),'b','-')
# plt.xlabel("n_samples")
# plt.ylabel("percent deviation from mean site emissions symlog scale")
# ax.set_yscale('symlog')
# # plt.ylim(-1000,1000)
# plt.show()

# #%%
# file_a.hist('Emission magnitude [kgh]')
