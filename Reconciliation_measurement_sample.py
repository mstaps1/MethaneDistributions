# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 13:57:02 2023

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
n_sites = 1000
number_bootstraps = 5000

MDL = 1

total_emissions_for_bootstrap = np.zeros(number_bootstraps)
# These should be sampled
mean_emission_rate = np.zeros(number_bootstraps)
median_emission_rate = np.zeros(number_bootstraps)
# loop over bootstraps
# for bootstrap in range(number_bootstraps):

n_sites_A = 50    
n_sites_B = 25 
n_sites_C = 25

rng = np.random.default_rng()

#bias
mu_A = 0.1
mu_B = 0
mu_C = 0.15
# measurement error
sigma_A = 0.35/2
sigma_B = 0.35/2
sigma_C = 0.35/2

Site_A_each_sources_emissions = file_a.loc[:, column].sample(n=n_sites_A).to_numpy() * ( rng.normal(mu_A, sigma_A, n_sites_A))
Site_B_each_sources_emissions = file_a.loc[:, column].sample(n=n_sites_B).to_numpy() * ( rng.normal(mu_A, sigma_B, n_sites_A))
Site_C_each_sources_emissions = file_a.loc[:, column].sample(n=n_sites_C).to_numpy() * ( rng.normal(mu_A, sigma_C, n_sites_A))



total_emission_rate_site_type_A = np.sum(Site_A_each_sources_emissions)
total_emission_rate_site_type_B = np.sum(Site_B_each_sources_emissions)
total_emission_rate_site_type_C = np.sum(Site_C_each_sources_emissions)

total_emissions_MDL_type_A = np.where(total_emission_rate_site_type_A<MDL, 0, total_emission_rate_site_type_A)
total_emissions_MDL_type_B = np.where(total_emission_rate_site_type_B<MDL, 0, total_emission_rate_site_type_B)
total_emissions_MDL_type_C = np.where(total_emission_rate_site_type_C<MDL, 0, total_emission_rate_site_type_C)

# 
each_sources_emissions_after_mdl = np.where(each_sources_emissions<MDL, 0, each_sources_emissions)

#
mean_emission_rate = np.mean(each_sources_emissions_after_mdl)
median_emission_rate = np.median(each_sources_emissions_after_mdl)
variance = np.var(each_sources_emissions_after_mdl)
    
    
    
    
    
    
    
    
    
    
    
    
    
    