# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 14:07:12 2023

@author: marsh
"""

from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
from utils import data_loader
#


basin, year = 'Permian', '2021'
pandas_frame, file = data_loader(basin, year)
print(file)

model_ = 'Emission magnitude [kgh]'

column = model_

# 
n_sites = 1000

# number_distribution_changes_year
num_EED_year = 365
total_emissions_for_time_period = np.zeros(num_EED_year)

hours_per_year = 365*24

for day in range(num_EED_year):
    each_sources_emissions = pandas_frame.loc[:, column].sample(n=n_sites,replace=True).to_numpy()
    total_emissions_for_time_period[day] = np.sum(each_sources_emissions)
    
# True total annual emissions?
total_emissions_year = np.sum(total_emissions_for_time_period*24)

#number of Campaigns
samples_per_year = np.arange(12)+1

#
num_monte_samples = 5000

#
each_sources_emissions[::n_sites]
#%%
mean_emission_samples = np.zeros((len(samples_per_year)+1, num_monte_samples))
sample_var_emissions = np.zeros((len(samples_per_year), num_monte_samples))

# taking 12 samples per year
MDL = 0

for i_sample, num_samples in enumerate(samples_per_year):
    for sample in range(num_monte_samples):
        each_sources_emissions = []
        
        # loop through
        # for i in range(num_samples):
        #     each_sources_emissions.append( np.sum(file_a.loc[:, column].sample(n=n_sites).to_numpy()) )
        #
        each_sources_emissions = pandas_frame.loc[:, column].sample(n=n_sites*num_samples, replace=True).to_numpy()
        mask = each_sources_emissions>=MDL
        
        # get mean emission rate
        mean_emission_samples[i_sample, sample] =  np.mean(each_sources_emissions[mask])
        
        # use ddof=1 for sample variance
        sample_var_emissions[i_sample, sample] = np.sqrt( np.var(each_sources_emissions[mask], ddof=1)/(n_sites*num_samples) )

# 
measured_total_annual_emissions = mean_emission_samples*hours_per_year*n_sites/total_emissions_year*100-100

#%% Mean 
fig, ax0 = plt.subplots(1, 1)

#
mean_over_bootstraps = np.mean(measured_total_annual_emissions, axis=1)
mean_over_bootstraps = mean_over_bootstraps[:-1]

#
_05_percentile_mean = np.percentile(measured_total_annual_emissions, [5], axis=1).flatten()
_95_percentile_mean = np.percentile(measured_total_annual_emissions, [95], axis=1).flatten()


ax0.fill_between(samples_per_year, _05_percentile_mean[:-1], _95_percentile_mean[:-1], alpha = 0.5, label='5th -95th percentile mean')

ax0.plot(samples_per_year, mean_over_bootstraps)

ax0.set_xlabel('number of complete campaigns per year')
ax0.set_ylabel('Percent deviation of annual emissions')

ax0.legend(loc='lower right')

#%% Variance
fig, ax0 = plt.subplots(1, 1)

#
mean_over_bootstraps_var = np.mean(sample_var_emissions, axis=1)
# mean_over_bootstraps_var = mean_over_bootstraps_var[:-1]

#
_05_percentile_var = np.percentile(sample_var_emissions, [5], axis=1).flatten()
_95_percentile_var = np.percentile(sample_var_emissions, [95], axis=1).flatten()


ax0.fill_between(samples_per_year, _05_percentile_var, _95_percentile_var, alpha = 0.5, label='5th -95th percentile Var')

ax0.plot(samples_per_year, mean_over_bootstraps_var)

ax0.set_xlabel('Number of complete campaigns per year')
ax0.set_ylabel('Variance in site level emission rates [kg/h]')

ax0.legend(loc='lower right')  