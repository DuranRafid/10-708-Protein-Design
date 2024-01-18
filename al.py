#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 15:56:37 2023

@author: youngjel
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import modal
import torch
import math

from functools import partial
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RationalQuadratic,ExpSineSquared,DotProduct
from modAL.models import BayesianOptimizer
from modAL.acquisition import optimizer_EI, max_EI

import matplotlib
matplotlib.rcParams.update({'font.size': 30})

def GenerateHeatmapFigure(x,y,c,sampled,figname,xlabel,ylabel):

    fig_size=(30,20)
    plt.figure(figsize=fig_size)
    ax = plt.gca()
    cbar_kw = {}
    cbarlabel=""
    # Plot the heatmap
    im = ax.scatter(x,y,c=c)
    for i in range(len(sampled)):
        plt.text(x[sampled[i]],y[sampled[i]],str(i),fontsize=15,color='red')
    
    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(figname)
    plt.clf()

def BayesianOptimization_GP(data,start_pos,kernel_name,num_test=20):
    
    data_agent=data['CROWDING AGENT'].to_list()
    data_agent=np.asarray([int(i) for i in data_agent]).reshape(-1,1)
    data_mat=data[['TEMPERATURE', 'CONCENTRATION', 'PH', 'IONIC STRENGTH']].to_numpy()
    data_mat=np.append(data_mat,data_agent,axis=1)
    y_value=data['LLPS PROBABILITY'].to_list()
    # index=data['PROTEIN NAME'].to_list()
    # print(np.argmin(y_value))
    X=data_mat
    y=np.asarray(y_value)
    
    start_pos=np.argmin(y_value)
    sampled=[start_pos]
    ###Gaussian process
    # assembling initial training set
    X_initial, y_initial = X[start_pos].reshape(1, -1), y[start_pos].reshape(1, -1)

    # defining the kernel for the Gaussian process
    if kernel_name=='Matern':
        kernel = Matern(length_scale=1.0)
    elif kernel_name=='RationalQuadratic':        
        kernel = RationalQuadratic(length_scale=1.0)
    elif kernel_name=='ExpSineSquared':
        kernel = ExpSineSquared(length_scale=1.0)
    elif kernel_name=='DotProduct':        
        kernel = DotProduct()



    regressor = GaussianProcessRegressor(kernel=kernel)

    optimizer = BayesianOptimizer(
        estimator=regressor,
        X_training=X_initial, y_training=y_initial,
        query_strategy=max_EI
    )
    # Bayesian optimization
    for n_query in range(num_test):
        query_idx, query_inst = optimizer.query(X)
        sampled.append(query_idx[0])
        optimizer.teach(X[query_idx].reshape(1, -1), y[query_idx].reshape(1, -1))

    y_pred, y_std = optimizer.predict(X, return_std=True)
    y_pred, y_std = y_pred.ravel(), y_std.ravel()
    X_max, y_max = optimizer.get_max()
    
    return y_pred, sampled, y_value

def GeneratePlot(sampled1,y_value1,all_sampled,kernels,xlabel,ylabel,savepath):

    matplotlib.rcParams.update({'font.size': 50})
    for i in range(len(sampled1)):
        plt.plot(i,y_value1[all_sampled[0][i]],'ro',markersize=30)
        plt.plot(i,y_value1[all_sampled[1][i]],'go',markersize=30)
        plt.plot(i,y_value1[all_sampled[2][i]],'bo',markersize=30)
        print(y_value1[all_sampled[0][i]],y_value1[all_sampled[1][i]],y_value1[all_sampled[2][i]])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim([0,1])
    plt.legend(kernels)
    plt.tight_layout()
    plt.savefig(savepath)
    plt.clf()


kernels=['Matern', 'RationalQuadratic','DotProduct']

f1='condition_exp/BO_condition_result.txt'
start_pos=58800

data=pd.read_csv(f1,sep='\t')
all_sampled=[]

for cur_kernel in kernels:
    y_pred1, sampled1, y_value1=BayesianOptimization_GP(data,start_pos,cur_kernel,10)
    all_sampled.append(sampled1)

    data_concentration=data['CONCENTRATION'].to_list()
    data_temperature=data['TEMPERATURE'].to_list()
    data_ph=data['PH'].to_list()
    data_ionic_strength=data['IONIC STRENGTH'].to_list()
    
    
    GenerateHeatmapFigure(data_concentration,data_ph,y_value1,sampled1,'BO_x_concen_y_ph_'+cur_kernel+'.pdf','Concentration','PH')
    GenerateHeatmapFigure(data_concentration,data_ionic_strength,y_value1,sampled1,'BO_x_concen_y_ionic_streng_'+cur_kernel+'.pdf','Concentration','IONIC STRENGTH')
    GenerateHeatmapFigure(data_concentration,data_temperature,y_value1,sampled1,'BO_x_concen_y_temp_'+cur_kernel+'.pdf','Concentration','TEMPERATURE')
    GeneratePlot(sampled1,y_value1,all_sampled,kernels,'Iteration','LLPS Probability','score_trend_gp_changes_'+cur_kernel+'.jpg')
########### Random case
f2='condition_exp/random_condition_result.txt'
start_pos=58800

data=pd.read_csv(f2,sep='\t')
all_sampled=[]

for cur_kernel in kernels:
    y_pred1, sampled1, y_value1=BayesianOptimization_GP(data,start_pos,cur_kernel,10)
    all_sampled.append(sampled1)

    data_concentration=data['CONCENTRATION'].to_list()
    data_temperature=data['TEMPERATURE'].to_list()
    data_ph=data['PH'].to_list()
    data_ionic_strength=data['IONIC STRENGTH'].to_list()
    
    
    GenerateHeatmapFigure(data_concentration,data_ph,y_value1,sampled1,'random_x_concen_y_ph_'+cur_kernel+'.pdf','Concentration','PH')
    GenerateHeatmapFigure(data_concentration,data_ionic_strength,y_value1,sampled1,'random_x_concen_y_ionic_streng_'+cur_kernel+'.pdf','Concentration','IONIC STRENGTH')
    GenerateHeatmapFigure(data_concentration,data_temperature,y_value1,sampled1,'random_x_concen_y_temp_'+cur_kernel+'.pdf','Concentration','TEMPERATURE')
    GeneratePlot(sampled1,y_value1,all_sampled,kernels,'Iteration','LLPS Probability','random_score_trend_gp_changes_'+cur_kernel+'.jpg')

