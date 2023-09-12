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
from utils import data_loader
#
basin, year = 'Permian', '2021'
file_a, file = data_loader(basin, year)
print(file)

model_ = 'Emission magnitude [kgh]'
mean_value_A = file_a.loc[:, model_].mean()
column = model_

# number of sites that the operator operates
n_sites = 1000
n_campaigns = 2
# number of bootstraps to be performed
number_bootstraps = 5000

# Technology minimum detection limit
MDL = 1

#
rng = np.random.default_rng()

# There are 3 site types A:single seperator 1 well, B: single seperator 5 wells, C: 5 seperator 5 wells
# [A,B,C]
n_sites = np.array([50, 25, 25])
site_types = range(len(n_sites))
# use mean to increase emissions rate
mu = [0, 0.1, 0.15]

# Technology measurement error s.d.
# measurement error
sigma = [0.35/2, 0.35/2, 0.35/2]

# [site_A, site_B, site_C]
site_emissions_all = []
site_emissions_sampled = []
site_emissions_sampled_MDL = []
# how many times the emissions rates change in a year
true_emission_rate_discritization = 365
n_samples = n_sites*true_emission_rate_discritization

n_sites_sampled = n_sites*n_campaigns
site_type_statistics = dict()
site_type_statistics = {0:dict(),
                         1:dict(),
                         2:dict()}
for site_type in site_types:
    # Sample the emission rates
    all_emission_rates_single_site = file_a.loc[:, column].sample(n=n_sites[site_type], replace=True).to_numpy()
    # bias the results based on site type and add measurement error
    site_emissions_with_error = all_emission_rates_single_site*(1+rng.normal(mu[site_type], sigma[site_type], n_sites[site_type]))
    site_emissions_all.append(site_emissions_with_error)
    
    # Sample 
    site_emissions_sampled.append(rng.choice(site_emissions_with_error, n_sites_sampled))
    
    # Apply MDL
    site_emissions_sampled_MDL.append(np.where(site_emissions_sampled[site_type] < MDL, 0, site_emissions_sampled[site_type]))
    site_type_statistics[site_type]['mean'] = np.mean(site_emissions_sampled_MDL[site_type])
    site_type_statistics[site_type]['median'] = np.mean(site_emissions_sampled_MDL[site_type])
    site_type_statistics[site_type]['variance'] = np.mean(site_emissions_sampled_MDL[site_type])
    site_type_statistics[site_type]['max release rate'] = np.amax(site_emissions_sampled_MDL[site_type])
    site_type_statistics[site_type]['min release rate'] = np.amin(site_emissions_sampled_MDL[site_type])
#%%

#
import pprint

# merge all of the site emissions together
all_emissions_sampled_MDL = np.array(site_emissions_sampled_MDL).flatten()
campaign_stats = dict()
campaign_stats['mean'] = np.mean(all_emissions_sampled_MDL)
campaign_stats['median'] = np.mean(all_emissions_sampled_MDL)
campaign_stats['variance'] = np.mean(all_emissions_sampled_MDL)
campaign_stats['max release rate'] = np.amax(all_emissions_sampled_MDL)
campaign_stats['min release rate'] = np.amin(all_emissions_sampled_MDL)
print('Site type stats: \n', )
pprint.pprint(site_type_statistics)   
print('Measurement campaign stats: \n')
pprint.pprint(site_type_statistics)
#
#%%
bins = np.arange(-0.1,3.2,0.1)
x = all_emissions_sampled_MDL
# Get bin centers
x_in = np.hstack((np.log10(x[x.astype(bool)]).reshape(-1),x[~x.astype(bool)]))
# y, x = np.histogram(np.log10(x[np.nonzero(x)]), bins='auto', density=True)
y, x = np.histogram(x_in, bins=bins, density=False)
x = x[:-1] + (x[1] - x[0])/2
bar_width = 0.133
fig, a0 = plt.subplots(1, 1,figsize=(15,9))
a0.bar(x,y/np.sum(y),bar_width,alpha=1)
a0.set_xlabel('10^x Site Emissions [kg/hr]', fontsize=24)
a0.set_ylabel('Probability', fontsize=24)
# a0.set_yscale('log')
a0.tick_params(axis='both', labelsize=18)

    
    
    
    
    
    
    
    
    
    