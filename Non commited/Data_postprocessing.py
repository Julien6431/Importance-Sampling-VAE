# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 17:21:49 2023

@author: jdemange
"""

#%% Modules

import numpy as np


#%%

def get_numerical_results(example):
    if example == "four_branches":
        data1 = np.load("Data/four_branches_failprob_estimations_100.npz")
        data2 = np.load("Data/four_branches_failprob_estimations_100_3_5.npz")
        ce_vae = data1["CE_vae"]
        ce_vMNFM = data1["CE_vM"]
        ce_vMNFM_minus_1 = data2["CE_vM_3"]
        ce_vMNFM_plus_1 = data2["CE_vM_5"]
        ref = 9.3e-4
        
    elif example == "oscillator":
        data_osc = np.load("Data/oscillator_failprob_estimations_200.npz")
        ce_vae = data_osc["CE_vae"]
        ce_vMNFM = data_osc["CE_vM2"]
        ce_vMNFM_minus_1 = data_osc["CE_vM1"]
        ce_vMNFM_plus_1 = data_osc["CE_vM3"]
        ref = 4.28e-4
    
    print(np.mean(ce_vae))
    print(np.mean(ce_vMNFM))
    print(np.mean(ce_vMNFM_minus_1))
    print(np.mean(ce_vMNFM_plus_1))
    
    print("")
    
    print(np.std(ce_vae)/ref)
    print(np.std(ce_vMNFM)/ref)
    print(np.std(ce_vMNFM_minus_1)/ref)
    print(np.std(ce_vMNFM_plus_1)/ref)
    
    def compute_N(cv):
        return (1-ref)/(ref*cv**2)
    
    print("")
    
    if example == "four_branches":
        print(compute_N(np.std(ce_vae)/ref)/40000)
        print(compute_N(np.std(ce_vMNFM)/ref)/50000)
        print(compute_N(np.std(ce_vMNFM_minus_1)/ref)/200000)
        print(compute_N(np.std(ce_vMNFM_plus_1)/ref)/50000)
        
    elif example == "oscillator":

        N_tots = data_osc["N_tots"]       
        
        print(np.mean(N_tots,axis=0))

        print(compute_N(np.std(ce_vae)/ref)/30000)
        print(compute_N(np.std(ce_vMNFM)/ref)/40000)
        print(compute_N(np.std(ce_vMNFM_minus_1)/ref)/200000)
        print(compute_N(np.std(ce_vMNFM_plus_1)/ref)/40000)        

#%% Four branches

# data1 = np.load("Data/four_branches_failprob_estimations_100.npz")
# data2 = np.load("Data/four_branches_failprob_estimations_100_3_5.npz")

# ce_vae = data1["CE_vae"]
# ce_vMNFM = data1["CE_vM"]
# ce_vMNFM3 = data2["CE_vM_3"]
# ce_vMNFM5 = data2["CE_vM_5"]S


# print(np.mean(ce_vae))
# print(np.mean(ce_vMNFM))
# print(np.mean(ce_vMNFM3))
# print(np.mean(ce_vMNFM5))

# print("")

# ref = 9.3e-4
# print(np.std(ce_vae)/ref)
# print(np.std(ce_vMNFM)/ref)
# print(np.std(ce_vMNFM3)/ref)
# print(np.std(ce_vMNFM5)/ref)

# print("")

# def compute_N(cv):
#     return (1-ref)/(ref*cv**2)

# print(compute_N(np.std(ce_vae)/ref)/40000)
# print(compute_N(np.std(ce_vMNFM)/ref)/50000)
# print(compute_N(np.std(ce_vMNFM3)/ref)/200000)
# print(compute_N(np.std(ce_vMNFM5)/ref)/50000)

#%% Oscillator

get_numerical_results("oscillator")