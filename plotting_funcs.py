# -*- coding: utf-8 -*-
"""
Created on Wed May 24 13:02:13 2023

@author: marsh
"""
import numpy as np
import matplotlib.pyplot as plt

def plotter_box_plot(data, x_labels, x_axis_title, y_axis_title, title):
    fig, ax0 = plt.subplots(1, 1,figsize=(20, 2.7*4))

    ax0.boxplot(data,labels=x_labels)

    ax0.set_ylim(None,20)
    ax0.set_title(title, fontsize=16)
    ax0.set_ylabel(y_axis_title)
    ax0.set_xlabel(x_axis_title)