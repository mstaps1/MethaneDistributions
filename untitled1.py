# -*- coding: utf-8 -*-
"""
Created on Wed May 24 11:42:58 2023

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
        
model_ = 'Emission magnitude [kgh]'
# model_ = 'Log emission magnitude Rutherford [kgh]'
# file_a.hist('Emission magnitude [kgh]')
mean_value = file_a.loc[:, model_].mean()
column = model_
n_bootstrap = 1000
n_samples_max = 4000
step = 50

means = np.zeros((n_samples_max//step, n_bootstrap))
mdl_list = [1,10,50]
mdl_array_measured = np.zeros((n_samples_max//step, n_bootstrap,len(mdl_list)))
total_emissions_bootstap = np.zeros((n_samples_max//step, n_bootstrap))
# mdl_array_unmeasured = np.zeros((n_samples_max//step, n_bootstrap))


n_samples_range = np.arange(n_samples_max-1,step=step)+1
j = 0


mdl = 10

for n_samples in n_samples_range:#range(1, n_samples_max+1, 10):#n_samples_range[3122:]:
    for i in range(n_bootstrap):
        #
        bootstrap_samples = file_a.loc[:, column].sample(n=n_samples).to_numpy()
        
        #
        if model_ == 'Log emission magnitude Rutherford [kgh]':
            for k, mdl in enumerate(mdl_list):
                mdl_mask = bootstrap_samples>np.log10(mdl)
                mdl_array_measured[j, i,k] = np.sum(10**bootstrap_samples[mdl_mask])
                # mdl_array_unmeasured[j, i] = np.sum(10**bootstrap_samples[~mdl_mask])
            total_emissions_bootstap[j, i] = np.sum(10**bootstrap_samples)
        elif model_ == 'Emission magnitude [kgh]':
            for k, mdl in enumerate(mdl_list):
                mdl_mask = bootstrap_samples>mdl
                mdl_array_measured[j, i,k] = np.sum(bootstrap_samples[mdl_mask])
            # mdl_array_unmeasured[j, i] = np.sum(bootstrap_samples[~mdl_mask])
            total_emissions_bootstap[j, i] = np.sum(bootstrap_samples)
        #
        means[j, i] = np.mean(bootstrap_samples)
        
    j += 1
#%%
#******************************************************************************
# Calculate the percent measured vs unmeasured
#******************************************************************************
# basin = 'Denver 2021'
# percent_measured = -(mdl_array_measured - total_emissions_bootstap)/total_emissions_bootstap*100
# percent_unmeasured = -(mdl_array_measured - total_emissions_bootstap)/total_emissions_bootstap*100

# fig, ax0 = plt.subplots(1, 1,figsize=(20, 2.7*4))
# # ax0.plot(n_samples_range,mdl_array_measured)
# ax0.set_title(f'Percentage of total emissions Measurable based on a MDL of {mdl} kg/hr {basin}', fontsize=16)
# ax0.fill_between(n_samples_range,
#                  np.amin(percent_measured,axis=1),
#                  np.amax(percent_measured,axis=1),
#                  alpha = 0.75,
#                  label='min value to max value')

# ax0.fill_between(n_samples_range,
#                  np.percentile(percent_measured,[5],axis=1).flatten(),
#                  np.percentile(percent_measured,[95],axis=1).flatten(),
#                  alpha=0.75,
#                  label='5th -95th percentile')

# ax0.fill_between(n_samples_range,
#                  np.percentile(percent_measured,[25],axis=1).flatten(),
#                  np.percentile(percent_measured,[75],axis=1).flatten(),
#                  alpha=0.75,
#                  label='25th -75th percentile')
# ax0.legend(loc='upper right')
# ax0.set_ylabel('Site Emissions [kg/hr]')
# ax0.set_xlabel('number of sites')

#%%
#******************************************************************************
# Calculate the percent measured vs unmeasured
#******************************************************************************
# Mean_of_means = np.mean(means[:n_samples//step],axis=1)
# p_dev_mean = (means[:n_samples//step] - np.expand_dims(Mean_of_means,axis=1))*np.expand_dims(Mean_of_means,axis=1)*100

# x = file_a.loc[:, model_].to_numpy()


#%%
from scipy.interpolate import UnivariateSpline
fig, a0 = plt.subplots(1, 1,figsize=(15,9))

x = file_a.loc[:, model_].to_numpy()

y, x = np.histogram(np.log10(x[np.nonzero(x)]), bins='auto', density=True)
# a0.hist(y,bins=x)
# a0.hist(x,bins=20)
# Get bin centers
x = x[:-1] + (x[1] - x[0])/2 
# n= 1000
# f = UnivariateSpline(x, y, s=n)
# a0.plot(x, f(x))
bar_width = 0.133
a0.bar(x,y,bar_width,alpha=0.5)
# a0.set_xscale('log')
# a0.set_yscale('log')
a0.set_xlabel('10^x Site Emissions [kg/hr]')
a0.set_ylabel('Probability')
# a0.set_ylim([None, 5*10**4])
a0.set_title(f'Probability Density Function for Emission Rates for {basin}', fontsize=16)



#%%
#******************************************************************************
# Percent Deviation from the mean
#******************************************************************************

# percent_measured_error = -(mdl_array_measured - total_emissions_bootstap)/total_emissions_bootstap*100
# percent_unmeasured = -(mdl_array_measured - total_emissions_bootstap)/total_emissions_bootstap*100


percent_measured = mdl_array_measured/np.expand_dims(total_emissions_bootstap,axis=2)*100

Mean_of_means = np.mean(means[:n_samples//step],axis=1)
p_dev_mean = (means[:n_samples//step] - np.expand_dims(Mean_of_means,axis=1))*np.expand_dims(Mean_of_means,axis=1)*100


if model_ == 'Emission magnitude [kgh]':
    percentile_list = [5, 25, 75, 95]
    x = file_a.loc[:, model_].to_numpy()
    
    # fig, ax0 = plt.subplots(1, 1,figsize=(20, 2.7*4))
    # ax0.boxplot(p_dev_mean[0::2].transpose(),labels=n_samples_range[:-1][0::2])
    # ax0.set_ylabel('Percent Deviation from Mean')
    # ax0.set_xlabel('number of sites sampled')
    # ax0.set_yscale('symlog')
    # ax0.set_title(f'Box plot of % deviation from mean site emissions for the {basin}', fontsize=16)
    
    
    
    fig, (ax1, ax2,ax3) = plt.subplots(3, 1,figsize=(15,20*4/3))
    
    ax1.hist(x,bins=20)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Site Emissions [kg/hr]')
    ax1.set_ylabel('number of sites')
    ax1.set_ylim([None, 5*10**4])
    ax1.set_title(f'Histogram of emission rates for {basin}', fontsize=16)
    
    
    # ax2.boxplot(p_dev_mean[0::4].transpose(),labels=n_samples_range[:-1][0::4])
    # ax2.set_ylabel('Percent Deviation from Mean')
    # ax2.set_xlabel('number of sites sampled')
    # ax2.set_yscale('symlog')
    # ax2.set_title(f'Box plot of % deviation from mean site emissions for the {basin}', fontsize=16)
    
    
    # ax2.plot(n_samples_range[:-1],p_dev_mean,'b', alpha=0.01)
    # for percentile in percentile_list:
    #     plt.plot(n_samples_range[:-1],np.percentile(p_dev_mean,[percentile],axis=1).transpose(),'k',label=f'{percentile}th %')
    ax2.fill_between(n_samples_range[:-1], np.amin(p_dev_mean,axis=1), np.amax(p_dev_mean,axis=1),alpha = 0.5, label='min value to max value')
    ax2.fill_between(n_samples_range[:-1], np.percentile(p_dev_mean,[5],axis=1).flatten(), np.percentile(p_dev_mean,[95],axis=1).flatten(),alpha=0.2,label='5th -95th percentile')
    ax2.fill_between(n_samples_range[:-1], np.percentile(p_dev_mean,[25],axis=1).flatten(), np.percentile(p_dev_mean,[75],axis=1).flatten(),alpha=0.4,label='25th -75th percentile')
    ax2.set_title(f'% dev per n samples for {basin}', fontsize=16)
    # ax2.set_yscale('symlog')
    ax2.set_xlabel('n_samples')
    ax2.set_ylabel('% deviation from mean site emissions')
    ax2.legend(loc='upper right')
    # ax2.set_xscale('log')
    ax2.set_yscale('symlog')
    # ax2.set_ylim([None, 5*10**5])
    
    # ax3.set_title(f'Percentage of total emissions Measurable based on a MDL of {mdl} kg/hr {basin}', fontsize=16)
    # ax3.fill_between(n_samples_range,
    #                  np.amin(percent_measured,axis=1),
    #                  np.amax(percent_measured,axis=1),
    #                  alpha = 0.75,
    #                  label='min value to max value')

    # ax3.fill_between(n_samples_range,
    #                  np.percentile(percent_measured,[5],axis=1).flatten(),
    #                  np.percentile(percent_measured,[95],axis=1).flatten(),
    #                  alpha=0.75,
    #                  label='5th -95th percentile')

    # ax3.fill_between(n_samples_range,
    #                  np.percentile(percent_measured,[25],axis=1).flatten(),
    #                  np.percentile(percent_measured,[75],axis=1).flatten(),
    #                  alpha=0.75,
    #                  label='25th -75th percentile')
    for i, mdl in enumerate(mdl_list):
        ax3.plot(n_samples_range, np.mean(percent_measured[:,:,i],axis=1),label=f'MDL {mdl} [kg/hr]')
        # ax3.plot(n_samples_range, np.mean(percent_measured[:,:,i],axis=1),c='k')
        # ax3.fill_between(n_samples_range,
        #                   np.percentile(percent_measured[:,:,i],[5],axis=1).flatten(),
        #                   np.percentile(percent_measured[:,:,i],[95],axis=1).flatten(),
        #                   alpha=0.15,
        #                   label='5th -95th percentile')
        # ax3.boxplot(percent_measured[:,:,i].transpose(),labels=f'MDL {mdl} [kg/hr]')
    ax3.legend(loc='upper right')
    ax3.set_ylabel('percentage of sites that are measureable for given MDL')
    ax3.set_xlabel('number of sites')
    ax3.set_title(f'Percentage of total emissions Measurable for {basin}', fontsize=16)
    
    
    
else:
    x = file_a.loc[:, model_].to_numpy()
    fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(20, 2.7*4))
    
    ax1.hist(x,bins=20)
    # ax1.set_xscale('log')
    # ax1.set_yscale('log')
    ax1.set_xlabel('Site Emissions [kg/hr]')
    ax1.set_ylabel('number of sites')
    # ax1.set_ylim([None, 5*10**4])
    ax1.set_title(f'Histogram of emission rates for {basin}', fontsize=16)
    ax2.fill_between(n_samples_range[:-1], np.amin(p_dev_mean,axis=1), np.amax(p_dev_mean,axis=1),alpha = 0.5, label='min value to max value')









# fig, ax0 = plt.subplots(1, 1,figsize=(20, 2.7*4))
# ax0.boxplot(percent_measured[:,:,0].transpose())
# ax0.boxplot(percent_measured[:,:,1].transpose(),color='r')
# ax0.boxplot(percent_measured[:,:,2].transpose(),color='b')


























