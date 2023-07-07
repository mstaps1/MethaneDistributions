# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 15:56:48 2023

@author: marsh
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 10:11:23 2023

@author: marsh
"""

from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#
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

# 
Data_set = production_areas[-2]
# basin = 'San Joaquin basin 2020'
basin = 'Permian basin 2021'
# basin = 'Denver basin 2021 '
# basin = 'Uinta basin 2020 '
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
if model_ == 'Log emission magnitude Rutherford [kgh]':
    sample_mean_value = file_a.loc[:, model_].mean()
    sample_var_value = file_a.loc[:, model_].var()
    log_mean_value = sample_mean_value + sample_var_value/2
elif model_ == 'Emission magnitude [kgh]':
    mean_value = file_a.loc[:, model_].mean()

column = model_

# 
n_sites = 1000

# number_distribution_changes_year
num_EED_year = 365
total_emissions_for_time_period = np.zeros(num_EED_year)

hours_per_year = 365*24


"""
Sample 




"""

for day in range(num_EED_year):
    each_sources_emissions = file_a.loc[:, column].sample(n=n_sites,replace=True).to_numpy()
    total_emissions_for_time_period[day] = np.sum(each_sources_emissions)

total_emissions_year = np.sum(total_emissions_for_time_period*24)
# np.random.randint(0,365,size=12)
# sample_values = np.choose(total_emissions_for_time_period,2)

#%% monte_carlo simulation for sampling

# survey frequency
samples_per_year = np.arange(12)+1
num_monte_samples = 1000
mean_emission_samples = np.zeros((len(samples_per_year), num_monte_samples))
sample_var_emissions = np.zeros((len(samples_per_year), num_monte_samples))
# taking 12 samples per year
for num_samples in samples_per_year:
    
    
    #
    for sample in range(num_monte_samples):
        each_sources_emissions = file_a.loc[:, column].sample(n=n_sites,replace=True).to_numpy()
        
        standard_error = np.sqrt( np.var(each_sources_emissions, ddof=1)/n_sites )
        
        indicies = np.random.randint(0, 365, size=num_samples)
        # indicies = np.random.randint(0, 365, size=num_samples)
        mean_emission_samples[num_samples,sample] = np.mean(total_emissions_for_time_period[indicies])
        # Variance between each othe num_samples
        sample_var_emissions[num_samples,sample] = np.sqrt( np.var(total_emissions_for_time_period[indicies], ddof=1)/num_samples )
        
        # emission_samples = np.mean(total_emissions_for_time_period[indicies])
        # total_emissions
#%% monte_carlo simulation for sampling


#number of Campaigns
samples_per_year = np.arange(12)+1

#
num_monte_samples = 10000

#
each_sources_emissions[::n_sites]

mean_emission_samples = np.zeros((len(samples_per_year)+1, num_monte_samples))
sample_var_emissions = np.zeros((len(samples_per_year), num_monte_samples))
# taking 12 samples per year
MDL = 10

for i_sample, num_samples in enumerate(samples_per_year):
    for sample in range(num_monte_samples):
        each_sources_emissions = []
        
        # loop through
        # for i in range(num_samples):
        #     each_sources_emissions.append( np.sum(file_a.loc[:, column].sample(n=n_sites).to_numpy()) )
        #
        each_sources_emissions = file_a.loc[:, column].sample(n=n_sites*num_samples, replace=True).to_numpy()
        mask = each_sources_emissions>MDL
        
        # get mean emission rate
        mean_emission_samples[i_sample, sample] =  np.mean(each_sources_emissions[MDL])
        
        # use ddof=1 for sample variance
        sample_var_emissions[i_sample, sample] = np.sqrt( np.var(each_sources_emissions, ddof=1)/(n_sites*num_samples) )

# 
measured_total_annual_emissions = mean_emission_samples*hours_per_year*n_sites/total_emissions_year



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
ax0.set_ylabel('Extrapolated annual emissions estimate mean')

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
#%%


fig, ax0 = plt.subplots(1, 1)

mean_mean_value = np.mean(mean_emission_samples,axis=1)
mean_var_value = np.mean(sample_var_emissions,axis=1)

# ax0.plot(samples_per_year, mean_mean_value)
ax0.plot(samples_per_year, mean_var_value)

# _05_percentile_mean = np.percentile(mean_emission_samples,[5],axis=1).flatten()
# _95_percentile_mean = np.percentile(mean_emission_samples,[95],axis=1).flatten()
# ax0.fill_between(samples_per_year, _05_percentile_mean,_95_percentile_mean,alpha = 0.5,label='5th -95th percentile mean')


_05_percentile_var = np.percentile(sample_var_emissions,[5],axis=1).flatten()
_95_percentile_var = np.percentile(sample_var_emissions,[95],axis=1).flatten()
ax0.fill_between(samples_per_year, _05_percentile_var,_95_percentile_var, alpha = 0.5,label='5th -95th percentile variance')
ax0.set_xlabel('number of complete campaigns per year')
ax0.set_ylabel('Extrapolated annual emissions estimate varitaion')

ax0.legend(loc='lower right')    

#%%


fig, ax0 = plt.subplots(1, 1)
mean_mean_value = np.mean(mean_emission_samples,axis=1)
mean_var_value = np.mean(mean_emission_samples,axis=1)

# sample_var_2 = np.var(mean_emission_samples,axis=1, ddof=1)/samples_per_year
mean_anual_emissions = mean_value*hours_per_year/total_emissions_year

_95_percentile_mean = np.percentile(mean_emission_samples*hours_per_year/total_emissions_year,[95],axis=1).flatten()
_5_percentile_mean = np.percentile(mean_emission_samples*hours_per_year/total_emissions_year,[5],axis=1).flatten()
ax0.fill_between(samples_per_year, _5_percentile_mean,_95_percentile_mean,alpha = 0.5,label='5th -95th percentile mean')

# _95_percentile_mean = np.percentile(sample_var_emissions, [95],axis=1).flatten()
# _5_percentile_mean = np.percentile(sample_var_emissions, [5],axis=1).flatten()
# ax0.fill_between(samples_per_year, _5_percentile_mean,_95_percentile_mean,alpha = 0.5,label='5th -95th percentile var')

ax0.plot(samples_per_year, mean_anual_emissions)
# ax0.plot(samples_per_year, np.mean(sample_var_emissions, axis=1) )
ax0.set_xlabel('number of complete campaigns per year')
ax0.set_ylabel('Extrapolated annual emissions estimate varitaion')

ax0.legend(loc='lower right')
#%%
fig, ax0 = plt.subplots(1, 1)

# 
mean_value = np.mean(mean_emission_samples,axis=1)
mean_anual_emissions = mean_value*hours_per_year/total_emissions_year

sample_var_2 = np.var(mean_emission_samples,axis=1, ddof=1)/samples_per_year

# Calculate percentiles
_95_percentile_mean = np.percentile(sample_var_emissions, [95],axis=1).flatten()
_5_percentile_mean = np.percentile(sample_var_emissions, [5],axis=1).flatten()
ax0.fill_between(samples_per_year, _5_percentile_mean,_95_percentile_mean,alpha = 0.5,label='5th -95th percentile var')

ax0.plot(samples_per_year, sample_var_2)

ax0.set_xlabel('number of complete campaigns per year')
ax0.set_ylabel('Extrapolated annual emissions estimate varitaion')

ax0.legend(loc='lower right')

# monte_carlo simulation

# true_total_emissions = 

# get daily average emission rae
# np.mean(sample_values, axis=1)

# total yearly emissions for sites
































