# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 12:23:29 2023

@author: marsh

Monte Carlo simulations for various number of annual measurement campaigns and plotting.
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

# Select the permian basin as the data set.
Data_set = production_areas[-2]
basin = 'Permian basin 2021'
for file_ in onlyfiles:
    #
    if Data_set in file_:
        #
        permian_files.append(file_)
# Now import the file into pandas
file_a = pd.read_csv(mypath+'/'+permian_files[1])
# Select the column for data in this case choose the 'Emission magnitude [kgh]'
model_ = 'Emission magnitude [kgh]'
mean_value_A = file_a.loc[:, model_].mean()
column = model_

# number of sites the operator operates.
n_sites = 1000

# number of days sampled per year
num_EED_year = 365
total_emissions_for_time_period = np.zeros(num_EED_year)
hours_per_year = 365*24

# Generate the hypothetical population of emission rates
for day in range(num_EED_year):
    each_sources_emissions = file_a.loc[:, column].sample(n=n_sites).to_numpy()
    total_emissions_for_time_period[day] = np.sum(each_sources_emissions)


total_emissions_year = np.sum(total_emissions_for_time_period*24)

# 1 - 12 samples per year
samples_per_year = np.arange(12)+1

#
num_monte_samples = 5000

#
# each_sources_emissions[::n_sites]

list_of_stats = ['mean','median','standard deviation',]


mean_emission_samples = np.zeros((len(samples_per_year)+1, num_monte_samples))
media
sample_var_emissions = np.zeros((len(samples_per_year), num_monte_samples))
# taking 12 samples per year
MDL = 1 #kg/hr

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
    
    
