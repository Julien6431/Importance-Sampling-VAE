# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 10:41:09 2021

@author: jdemange
"""


#%% Modules

import numpy as np
import matplotlib.pyplot as plt


#%% Load data

correspondanceIS = {'theo_values' : {'label':'Reference value'},
                  'CE_vae' : {'label':'CE VAE','color':'#FFFF00','position':2},
                  'CE_vM' : {'label':'CE vMFNM','color':'#FF00FF','position':3},
                  'iCE_vae' : {'label':'iCE VAE','color':'#00FFFF','position':4},
                  'iCE_vM' : {'label':'iCE vMFNM','color':'#00FF00','position':5}}


#%% Ploting functions

def customize_boxplot(bp,face_color):
    #for patch, color in zip(bp['boxes'], face_color): 
        #patch.set_facecolor(face_color) 
        
    for patch in bp['boxes']:
        patch.set_facecolor(face_color) 
   
    # changing color and linewidth of 
    # whiskers 
    for whisker in bp['whiskers']: 
        whisker.set(color ='#8B008B', 
                    linewidth = 1.5) 

    # changing color and linewidth of 
    # caps 
    for cap in bp['caps']: 
        cap.set(color ='#8B008B', 
                linewidth = 2) 

    # changing color and linewidth of 
    # medians 
    for median in bp['medians']: 
        median.set(color ='black', 
                   linewidth = 3) 

    # changing style of fliers 
    for flier in bp['fliers']: 
        flier.set(marker ='D', 
                  color ='#e7298a', 
                  alpha = 0.5) 
        
        
        
def plot_boxplots(file,title,ref_value,ylim=(0,1)):
    """
    

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.

    Returns
    -------
    fig : TYPE
        DESCRIPTION.
    ax : TYPE
        DESCRIPTION.

    """
    
    data = np.load(file)
    data = dict(data)
    del data['N_tots']
    
    nb_data = len(data)
    

    correspondance = correspondanceIS
        
    reference_values = np.array([ref_value])#data["theo_values"]

    #positions = np.array([(nb_data+1)*i for i in range(dim)])
    
    fig,ax = plt.subplots(figsize=(15,15))
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"

    boxplots = np.zeros(nb_data,dtype=object)
    max_char = max([len(correspondance[dic]['label']) for dic in list(correspondance.keys())])
    legends = np.zeros(nb_data,dtype='<U'+str(max_char))
    
    bp_ref = ax.boxplot(reference_values.reshape((1,-1)),positions=[0.2*1],patch_artist = True)

    boxplots[0] = bp_ref["boxes"][0]    
    legends[0] = "Reference value"

    customize_boxplot(bp_ref,'red')
    for median in bp_ref['medians']: 
        median.set(color ='red', linewidth = 3)
        
    for label,dat in data.items():
        if label != "theo_values":
            dic_currlabel = correspondance[label]
            bp = ax.boxplot(dat.reshape((-1,1)),sym="k+",positions = [0.2*dic_currlabel['position']],patch_artist = True)
            customize_boxplot(bp,dic_currlabel['color'])
            
            boxplots[dic_currlabel["position"]-2] = bp["boxes"][0]
            legends[dic_currlabel["position"]-2] = dic_currlabel['label']

    ax.legend(list(boxplots),list(legends),loc='upper left',fontsize=30)

    #loc_xticks = [0.51 + 0.51*(nb_data//2) + (nb_data+1)*0.51*i for i in range(dim)]

    ax.set_xticks([])
    #ax.set_xticks(loc_xticks)
    ax.tick_params(axis='x',labelsize='xx-large')
    ax.tick_params(axis='y',labelsize='xx-large')
    ax.set_title(title,fontdict={'fontsize': 24,'fontweight' : 'bold','verticalalignment': 'baseline'})
    ax.set_xlim(0,1.2)
    ax.set_ylim(ylim[0],ylim[1])
    
    return fig,ax



#%%

plot_boxplots("Data/four_branches_failprob_estimations_100.npz", "Estimation of $p_f$ for the 4-branch problem in dimension 100",9.3e-4,ylim=(0,2e-3))
#plot_boxplots("Data/oscillator_failprob_estimations_200.npz", "Estimation of $p_f$ for the oscillator problem in dimension 200",4.28e-4,ylim=(0,1e-3))