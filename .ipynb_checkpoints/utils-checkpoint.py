# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 08:31:30 2023

@author: marsh
"""

from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


        
def data_loader(basin, year):
    #
    mypath = 'CDF_data'
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    basin_files = []

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
    data_set_name = production_areas[-2]
    for area in production_areas:
        if (basin in area) and (year in area):
            data_set_name = area
    for file_ in onlyfiles:
        #
        if data_set_name in file_:
            #
            basin_files.append(file_)
    pd_basin = pd.read_csv(mypath+'/'+basin_files[1]) 
    return pd_basin, data_set_name