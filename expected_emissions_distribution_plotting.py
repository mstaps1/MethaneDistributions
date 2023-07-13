# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 08:17:22 2023

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



fig, a0 = plt.subplots(1, 1,figsize=(15,9))

x = pandas_frame.loc[:, model_].to_numpy()

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
a0.set_xlabel('10^x Site Emissions [kg/hr]', fontsize=24)
a0.set_ylabel('Probability', fontsize=24)
a0.tick_params(axis='both', labelsize=18)
