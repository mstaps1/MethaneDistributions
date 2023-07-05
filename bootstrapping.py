# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 15:45:43 2023

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


Data_set = production_areas[-2]
# basin = 'San Joaquin basin 2020'
basin = 'Permian basin 2021'
# basin = 'Denver basin 2021 '
# basin = 'Uinta basin 2020 '
# Data_set = production_areas[-3]

for file_ in onlyfiles:
    #
    if Data_set in file_:
        #
        permian_files.append(file_)
        
# Now import the file into pandas
file_a = pd.read_csv(mypath+'/'+permian_files[1])        
        
# model_ = 'Emission magnitude [kgh]'
model_ = 'Log emission magnitude Rutherford [kgh]'
# file_a.hist('Emission magnitude [kgh]')
if model_ == 'Log emission magnitude Rutherford [kgh]':
    sample_mean_value = file_a.loc[:, model_].mean()
    sample_var_value = file_a.loc[:, model_].var()
    log_mean_value = sample_mean_value + sample_var_value/2
elif model_ == 'Emission magnitude [kgh]':
    mean_value = file_a.loc[:, model_].mean()
    
    
column = model_

m_samples_max = 4000

# Make sure m is less than 1/2 of n
if m_samples_max > len(file_a.index)//2:
    m_samples_max = len(file_a.index)//2

step_size_m = 50
m_samples_list = np.arange(50,m_samples_max-1,step=step_size_m)


#Number of bootstraps B
B_bootstraps = 1000

store_means = dict()

# mdl_list = [1,10,50]

# total_emissions_bootstap = np.zeros((n_samples_max//step, n_bootstrap))
# mdl_array_unmeasured = np.zeros((n_samples_max//step, n_bootstrap))


# n_samples_range = np.arange(n_samples_max-1,step=step)+1
j = 0
mean_of_means = dict()

store_log_means = np.zeros((len(m_samples_list),B_bootstraps))
mean_of_m_bootstraps = np.zeros(len(m_samples_list))


for site_idex, n_sites in enumerate(m_samples_list):
    bootstrap_store_sample_mean = np.zeros(B_bootstraps)
    bootstrap_store_sample_variance = np.zeros(B_bootstraps)
    bootstrap_store_mean = np.zeros(B_bootstraps)
    
    mean_store = dict()
    just_store_stuff = dict()
    
    for i in range(B_bootstraps):
        
        bootstrap_samples = file_a.loc[:, column].sample(n=n_sites).to_numpy()
        store_dict = dict()
        
        if model_ == 'Log emission magnitude Rutherford [kgh]':
            # Calculate sample mean of log transformed data
            bootstrap_store_mean[i] = np.sum(bootstrap_samples)
            
            # Calculate sample variance of log transformed data
            bootstrap_store_sample_variance[i] = np.var(bootstrap_samples)
            bootstrap_store_sample_mean[i] = np.mean(bootstrap_samples)
            bootstrap_store_mean[i] = (bootstrap_store_sample_mean[i]+bootstrap_store_sample_variance[i]/2)
            
            # mean_store[i] = bootstap_store[i]
        elif model_ == 'Emission magnitude [kgh]':
            store_log_means[site_idex,i] = np.mean(bootstrap_samples)
            

    if model_ == 'Log emission magnitude Rutherford [kgh]':
        # calculate the variance
         # bootstrap_store_sample_variance = np.var(bootstrap_store_mean)
         # bootstrap_store_sample_mean = np.mean(bootstrap_store_mean)
         #  # 
         # mean_of_m_bootstraps[site_idex] = (bootstrap_store_sample_mean+bootstrap_store_sample_variance/2)
         # store_log_means[site_idex] = bootstrap_store_mean
         store_log_means[site_idex] = bootstrap_store_mean

        
        # mean_of_m_bootstraps[site_idex] = np.mean(bootstrap_store_mean)
#%%   
if model_ == 'Log emission magnitude Rutherford [kgh]':
    # log_mean_deviation = store_log_means - np.expand_dims(mean_of_m_bootstraps,axis=1)
    # percent_dev = log_mean_deviation/np.expand_dims(mean_of_m_bootstraps,axis=1)*100
    
    Sample_Mean_of_means = np.mean(store_log_means, axis=1)
    sample_var = np.var(store_log_means,axis=1)
    Mean_of_means = Sample_Mean_of_means + sample_var/2
    
    percent_dev = (store_log_means - np.expand_dims(Mean_of_means,axis=1))*np.expand_dims(Mean_of_means,axis=1)*100
    
    # percent_dev = (store_log_means - log_mean_value)*log_mean_value*100
    
    Mean_of_means = np.mean(10**store_log_means, axis=1)
    percent_dev = (10**store_log_means - np.expand_dims(Mean_of_means,axis=1))*np.expand_dims(Mean_of_means,axis=1)*100
    
    
elif model_ == 'Emission magnitude [kgh]':
    
    Mean_of_means = np.mean(store_log_means, axis=1)
    percent_dev = (store_log_means - np.expand_dims(Mean_of_means,axis=1))*np.expand_dims(Mean_of_means,axis=1)*100
    
    # log_mean_deviation = store_log_means - np.expand_dims(mean_of_m_bootstraps,axis=1)
    # percent_dev = log_mean_deviation/np.expand_dims(mean_of_m_bootstraps,axis=1)*100
    
    # mean_of_means[site_idex] = np.mean(bootstap_store)
    # p_deviation_mean = np.array(bootstap_store)
    # mean_of_means[site_idex]
    # store_means[site_idex] = just_store_stuff

#%%    

fig, ax2 = plt.subplots(1, 1,figsize=(20, 2.7*4))
ax2.boxplot(percent_dev[0::4].transpose(),labels=m_samples_list[:-1][0::4],whis=(5,95))
ax2.set_ylabel('Percent Deviation from mean')
ax2.set_xlabel('number of sites sampled')
ax2.set_yscale('symlog')
ax2.set_title(f'Box plot of % deviation from mean of log transformed site emissions  for the {basin}', fontsize=16)

# ax2.set_yticks(list( np.arange(-6.5,7,0.25) ) )

#%%
fig, ax0 = plt.subplots(1, 1, figsize=(20*1.5,20) )

_95_percentile_mean = np.percentile(store_log_means,[95],axis=1).flatten()
_5_percentile_mean = np.percentile(store_log_means,[5],axis=1).flatten()
_75_percentile_mean = np.percentile(store_log_means,[75],axis=1).flatten()
_25_percentile_mean = np.percentile(store_log_means,[25],axis=1).flatten()

ax0.fill_between(m_samples_list, _5_percentile_mean,_95_percentile_mean,alpha = 0.5,label='5th -95th percentile')
ax0.fill_between(m_samples_list, _25_percentile_mean,_75_percentile_mean,alpha = 0.5,label='25th -75th percentile')
ax0.set_xlabel('number of sites sampled')
ax0.set_ylabel('Mean Emissions [log(kg/hr)]')
ax0.legend(loc='upper right')
#%%
fig, ax0 = plt.subplots(1, 1,figsize=(15,20*4/3))
bar_width =  0.133
ax0.bar(store_log_means,bar_width,alpha=0.5)
#%%
fig, ax0 = plt.subplots(1, 1,figsize=(15,20*4/3))

X = m_samples_list
p_dev_5 = []
p_dev_95 = []
# Iterate
for item in store_means.keys():
    # print('item',item)
    p_dev_list = []
    # print(type(item))
    item = 0
    for m_idex in store_means[item]:
        # print(f'm index',type(m_idex))
        p_dev_list.append( store_means[item][m_idex]['p_dev'] )
    p_dev_array = np.array(list(p_dev_list))
    p_dev_5.append( np.percentile(p_dev_array,[5]) )
    p_dev_95.append( np.percentile(p_dev_array,[95]) )
    
    
ax0.fill_between(m_samples_list, np.array(p_dev_5).flatten(), np.array(p_dev_95).flatten())
ax0.set_yscale('symlog')

#%%
fig, ax0 = plt.subplots(1, 1,figsize=(15,20*4/3))

X = m_samples_list
p_dev_5 = []
p_dev_95 = []
# Iterate
for item in store_means.keys():
    # print('item',item)
    p_dev_list = []
    # print(type(item))
    item = 0
    for m_idex in store_means[item]:
        # print(f'm index',type(m_idex))
        p_dev_list.append( store_means[item][m_idex]['percent_error'] )
    p_dev_array = np.array(list(p_dev_list))
    p_dev_5.append( np.percentile(p_dev_array,[5]) )
    p_dev_95.append( np.percentile(p_dev_array,[95]) )
    
    
ax0.fill_between(m_samples_list, np.array(p_dev_5).flatten(), np.array(p_dev_95).flatten())
# ax0.set_yscale('symlog')
        
        
        
        
