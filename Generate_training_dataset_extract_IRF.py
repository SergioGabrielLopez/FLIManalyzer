# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 18:35:55 2024

@author: lopez
"""

# Importing the necessary libraries. 
import numpy as np
import pandas as pd
from scipy.fftpack import fft, ifft
import random

def create_decay(x,I,t,q):
    """This function creates a power-law fluorescence decay from the x-axis
    values (x), the initial fluorescence intensity value (I), the centre
    of the lifetime distribution (t), and the heterogeneity parameter (q)."""
    decay = ((2-q)/t)*(1-((1-q)*(x/t)))**(1/(1-q))
    decay = decay * I
    return decay

def convolve_IRF(irf,decay):
    """This function takes and Instrument Response Function and a fluorescence
    decay and performs a convolution of the two."""
    conv = ifft(fft(decay) * fft(irf)).real
    return conv

def create_gaussian_curve(x):
    """This function creates a Gaussian curve using the x-axis values (x),
    the center value (mu), and the sigma value (sig)."""
    sig = round(random.uniform(0.04,0.20),2)
    mu = round(random.uniform(0.80,2.50),2)
    gauss = np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
    return gauss

def add_Poisson_noise(decay):
    """This function adds Poison noise to a curve."""
    noise_mask = np.random.poisson(decay)
    decay = decay + noise_mask
    return decay

def create_decay_real(x,irf):
    """This function creates a realistic fluorescence decay. The inputs are 
    the x-axis values (x), and an IRF (irf). The initial fluorescence value 
    (I), the centre value of the lifetime distribution, and the heterogeneity
    parameter (q) are obtained randomly by generating values between 100 and
    500 thousand (I), between 0.5 and 8 (t), and between 0.51 and 1.8 (q).
    Gaussian noise is added to the decays. The decays are returned
    normalised.""" 
    I = random.randint(100,500000)
    t = round(random.uniform(1.250,8.000),3)
    q = round(random.uniform(0.950,1.990),3)
    if q != 1.000:
        decay_raw = create_decay(x,I,t,q)
        decay_conv = convolve_IRF(irf,decay_raw)
        decay_noisy = add_Poisson_noise(decay_conv)
        decay_noisy = decay_noisy/np.max(decay_noisy)
    return decay_noisy
    
# We create the x-axis values.
x = np.arange(0.04848485,49.6,0.09696969)

# We create the list of column names for the Dataframe.
columns = list(map(str,list(np.arange(0,1024,1))))

# We create an empty DataFrame using the columns above.
df = pd.DataFrame(columns=columns)

for i in range(20000):
    try:
        gauss = create_gaussian_curve(x)
        y = create_decay_real(x,gauss)
        datum = list(y) + list(gauss)
        df.loc[len(df)] = datum
    except:
        print('Pass')

# We check if there are rows with NaN and we drop those rows.
if df.isnull().values.any():
    df.dropna()
    
# We check if there are rows with values below zero and we drop those rows.
if (df < 0).values.any():
    df = df[df.min(axis=1) >= 0]
    
# We save the dataframe.
df.to_excel('training_extract_irf.xlsx')